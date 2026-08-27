import math
import re
import warnings
from typing import Optional
from collections.abc import Sequence

import numpy as np


_NODE_META_KEYS = {
    "peak_idx": ("peak_idx", int),
    "spectrum_id": ("spectrum_id", int),
    "intensity": ("intensity", float),
}


def _parse_node_meta(node_str: str) -> dict:
    """Pull peak_idx / spectrum_id / intensity out of a FlowNode's ``str``.

    These fields live only in the C++ ``to_string`` (e.g.
    ``EmpiricalNode(2, peak_idx: 0, intensity: 10.000000)``); parsing them here
    lets the visualization layer size nodes and fill tooltips without new
    bindings.  Source/sink nodes carry none of these keys and yield ``{}``.
    """
    meta = {}
    for token, (key, cast) in _NODE_META_KEYS.items():
        m = re.search(rf"{token}:\s*([-\d.]+)", node_str)
        if m:
            meta[key] = cast(m.group(1))
    return meta

from wnet.wnet_cpp import (
    CWassersteinNetwork,
    CWassersteinNetworkSubgraph,
    CWassersteinNetworkFactory,
    NetworkSimplex,
    CostScaling,
    CycleCanceling,
    CapacityScaling,
    SlopeDP,
    NSPivotRule,
    CSMethod,
    CCMethod,
)
from wnet.distribution import Distribution
from wnet.distances import DistanceMetric
from wnet.scaling import FineGridScaler
from wnet.visualization import show_graph


class WassersteinNetwork:
    """
    A network class for computing Wasserstein distances between a base distribution and multiple target distributions.

    The majority of functionality is implemented in the underlying C++ class `CWassersteinNetwork`, which this class extends.

    Args:
        base_distribution (Distribution): The base distribution from which the Wasserstein distance is computed.
        target_distributions (Sequence[Distribution]): A sequence of target distributions to which the Wasserstein distance is computed.
        distance (DistanceFunction): A callable that computes the distance between points in the distributions.
        max_distance (float | None): The maximum distance to consider. If None or infinity, it defaults to the maximum representable value.
        force_dense_1d (bool): In 1D, force the O(m*n) dense factory instead of the O(m+n) chain factory. Default False uses the chain factory in 1D. Note: max_distance semantics differ between factories — chain only uses it to split the chain into components, while dense also caps per-pair cost.
        p (float): Wasserstein transport order, any real number >= 1. Each matching edge costs ground_distance**p, so total_cost() and all derivatives are in W_p**p units; take the p-th root for the literal W_p distance (the high-level WassersteinDistance() does this). For p != 1 the cost is fractional, so the integer solver works in auto-scaled units (round(scale_factor() * d**p)); the public total_cost()/derivatives divide that back out. p == 1 is bit-exact with the legacy 1-Wasserstein (scale_factor() == 1, truncation). p != 1 always uses the dense factory (the 1D chain factory is invalid for p != 1, since exponentiated gap costs are not additive); in 1D with p != 1 the chain factory is bypassed automatically.
        solver: Solver configuration object. One of NetworkSimplex(), CostScaling(), CycleCanceling(), or CapacityScaling(). Defaults to NetworkSimplex() (warm restarts, BLOCK_SEARCH pivot).
    """

    _SOLVER_METHODS = {
        "network_simplex": NetworkSimplex,
        "cycle_canceling": CycleCanceling,
        "cost_scaling": CostScaling,
        "capacity_scaling": CapacityScaling,
        "slope_dp": SlopeDP,
    }

    def __init__(
        self,
        base_distribution: Distribution,
        target_distributions: Sequence[Distribution],
        distance: DistanceMetric,
        max_distance: Optional[float] = None,
        force_dense_1d: bool = False,
        p: float = 1.0,
        solver=None,
        method: str = None,
        intensity_scale: Optional[float] = None,
        round_max_distance: bool = True,
    ) -> None:
        if solver is None and method is None:
            solver = NetworkSimplex()
        elif solver is None:
            if method not in self._SOLVER_METHODS:
                raise ValueError(
                    f"Unknown method {method!r}. Choose from: {list(self._SOLVER_METHODS)}"
                )
            solver = self._SOLVER_METHODS[method]()
        if max_distance is None or max_distance == float("inf"):
            max_distance = CWassersteinNetwork.max_value()
        elif round_max_distance:
            # Default: with p == 1 cost truncation the matching threshold is
            # effectively integer, so round a fractional cap up (inclusive) and
            # warn — preserving the legacy behaviour.  Callers that opt into
            # cost scaling (real distances) pass round_max_distance=False and
            # keep the real threshold.
            md_int = int(math.ceil(max_distance))
            if md_int != max_distance:
                warnings.warn(
                    f"max_distance={max_distance!r} is not an integer; the solver "
                    f"uses an integer distance threshold, so it was rounded up to "
                    f"{md_int}. Pass an integer max_distance to avoid this.",
                    stacklevel=2,
                )
            max_distance = md_int
        p = float(p)
        if not (p >= 1.0) or not np.isfinite(p):
            raise ValueError(
                f"Wasserstein order p must be a real number >= 1, got {p!r}."
            )
        self._distance = distance
        self._p = p
        vec_base = base_distribution.vecdist
        vec_targets = [t.vecdist for t in target_distributions]
        # CostScaling / CapacityScaling cannot solve the 1D chain factory: the
        # bidirectional chain arcs carry unbounded (INF) capacity, on which these
        # two LEMON algorithms return INFEASIBLE (garbage total cost) or loop. Only
        # NetworkSimplex / CycleCanceling handle the chain. Force the dense
        # factory for them so 1D results stay correct.
        chain_incompatible = isinstance(solver, (CostScaling, CapacityScaling))
        # The 1D chain factory is only valid for p == 1; for p != 1 fall back to
        # the dense factory (whose per-pair d**p costs are the correct transport cost).
        use_chain = (
            base_distribution.dimension == 1
            and not force_dense_1d
            and p == 1.0
            and not chain_incompatible
        )
        # SlopeDP is chain-native only: it needs the 1D chain factory graph.
        if isinstance(solver, SlopeDP) and not use_chain:
            raise ValueError(
                "SlopeDP solver requires the 1D chain factory "
                "(1D data, p == 1, force_dense_1d=False)."
            )
        if use_chain:
            self.wnet = CWassersteinNetworkFactory.create_1d(
                vec_base, vec_targets, distance, max_distance, p
            )
        else:
            self.wnet = CWassersteinNetworkFactory.create(
                vec_base, vec_targets, distance, max_distance, p
            )
        # Intensities are carried as real doubles into the int64 solver, which
        # quantizes them to integer supplies as round(real * intensity_scale).
        # None => auto: 1.0 for integer-valued intensities (bit-compatible) or
        # p != 1 (the joint cost/intensity budget is future work), else a scale
        # that lifts fractional intensities onto a fine integer grid without
        # overflowing the cost accumulator. An explicit value (e.g. 1.0) is used
        # verbatim. Must be set before build().
        if intensity_scale is None:
            # Float-backend intensity scaling uses the FineGridScaler:
            # no position pre-scale, intensities mapped onto a ~2**30 total-flow
            # grid, capped by cost_bound**p so the network's own cost scale
            # stays >= 1. Returns 1.0 for integer-valued data.
            intensity_scale = FineGridScaler(
                base_distribution,
                list(target_distributions),
                distance,
                max_distance,
                trash_costs=[],
                p=p,
            ).sf_intensity()
        self.wnet.set_intensity_scale(float(intensity_scale))

        self.add_simple_trash = self.wnet.add_simple_trash
        self.add_experimental_trash = self.wnet.add_experimental_trash
        self.add_theoretical_trash = self.wnet.add_theoretical_trash
        # Opt-in p=1 cost scaling (lets a caller pass real distances instead of
        # pre-scaling positions). Must be called before build().
        self.set_cost_scaling = self.wnet.set_cost_scaling
        # Declared upper bound on point-scaled total flow (real intensity
        # units): build() sizes the cost scale so any point within the budget
        # stays inside the int64 cost accumulator, and solve() rejects points
        # past the ceiling. Must be called before build().
        self.set_flow_budget = self.wnet.set_flow_budget

        # Avoid capturing self in the lambda to prevent reference cycles that could lead to memory leaks.  The underlying C++ object should be freed when this wrapper is freed, but if we capture self in the lambda, the lambda's reference to self would keep it alive indefinitely.
        # Without this trick, the lambda would hold a reference to self, which holds a reference to the C++ object, which holds a reference back to the lambda, creating a cycle that prevents garbage collection.
        # Without this, the incremental GC introduced in Python3.14 can't collect WassersteinNetwork instances that are no longer needed, leading to memory leaks.
        _wnet = self.wnet  # avoid capturing self in the lambda (reference cycle).
        _solver = solver
        self.build = lambda: _wnet.build(_solver)

        self.solve = self.wnet.solve
        self.scale_factor = self.wnet.scale_factor
        self.intensity_scale_factor = self.wnet.intensity_scale_factor
        self.get_subgraph = self.wnet.get_subgraph
        self.no_subgraphs = self.wnet.no_subgraphs
        self.flows_for_target = self.wnet.flows_for_target
        self.count_empirical_nodes = self.wnet.count_empirical_nodes
        self.count_theoretical_nodes = self.wnet.count_theoretical_nodes
        self.matching_density = self.wnet.matching_density
        self.lemon_to_string = self.wnet.lemon_to_string
        self.count_matching_edges = self.wnet.count_matching_edges
        self.count_theoretical_to_sink_edges = self.wnet.count_theoretical_to_sink_edges
        self.count_src_to_empirical_edges = self.wnet.count_src_to_empirical_edges
        self.count_simple_trash_edges = self.wnet.count_simple_trash_edges
        self.count_chain_edges = self.wnet.count_chain_edges
        self.no_theoretical_spectra = self.wnet.no_theoretical_spectra
        self.theoretical_spectra_sizes = self.wnet.theoretical_spectra_sizes

    def __str__(self) -> str:
        """Returns a string representation of the Wasserstein network."""
        return self.wnet.__str__()

    def total_cost(self) -> float:
        """Total transport cost in real ``W_p**p`` units (= sum of d**p * flow).

        Internally the integer solver works in scaled units; this divides the raw
        scaled cost by the auto-chosen ``scale_factor()`` to recover the real
        value.  For ``p == 1`` the scale is 1, so this is the exact integer cost
        (as a float).  Take the p-th root for the literal ``W_p`` distance.
        """
        # Scaled cost is in cost_scale * intensity_scale units (flow carries the
        # intensity scale, edge costs carry the cost scale).
        return self.wnet.total_cost() / (
            self.wnet.scale_factor() * self.wnet.intensity_scale_factor()
        )

    def signal_part_derivatives(self) -> dict[int, dict[int, float]]:
        """Compute the marginal cost of increasing each theoretical signal by 1.

        Returns a nested dict mapping spectrum_id -> {peak_index -> derivative},
        in real ``W_p**p`` units (un-scaled by ``scale_factor()``).
        """
        s = self.wnet.scale_factor()
        ret: dict[int, dict[int, float]] = {}
        for spec_id, peak_idx, deriv in self.wnet.signal_part_derivatives():
            ret.setdefault(spec_id, {})[peak_idx] = deriv / s
        return ret

    def spectrum_proportion_derivatives(self) -> np.ndarray:
        """Gradient of total cost w.r.t. scaling each spectrum's proportion.

        Aggregates across all subgraphs.  Returns array of derivatives indexed
        by spectrum_id (0..n-1), in real ``W_p**p`` units (un-scaled).
        """
        # Only the cost scale divides out here: the C++ already weights the
        # per-supply marginal by the REAL (unscaled) theoretical intensity, so
        # the intensity scale is absent from this derivative (verified by finite
        # difference). This still equals d(total_cost)/dw because total_cost
        # carries 1/(cost_scale*intensity_scale) and P_cpp carries 1/intensity_scale.
        s = self.wnet.scale_factor()
        return np.array([v / s for _, v in self.wnet.spectrum_proportion_derivatives()])

    def signal_part_derivatives_fast_approx(self) -> dict[int, dict[int, float]]:
        """Fast, APPROXIMATE signal_part_derivatives().

        Uses the pure dual-potential difference instead of the residual
        shortest-path search: much faster, but a different (basis-dependent)
        gradient — a lower bound on the true marginal, exact only for peaks on
        the optimal flow support.  Opt-in; not a drop-in for the exact one.
        """
        s = self.wnet.scale_factor()
        ret: dict[int, dict[int, float]] = {}
        for spec_id, peak_idx, deriv in self.wnet.signal_part_derivatives_fast_approx():
            ret.setdefault(spec_id, {})[peak_idx] = deriv / s
        return ret

    def spectrum_proportion_derivatives_fast_approx(self) -> np.ndarray:
        """Fast, APPROXIMATE spectrum_proportion_derivatives() (see
        signal_part_derivatives_fast_approx for the accuracy caveat)."""
        s = self.wnet.scale_factor()
        return np.array(
            [v / s for _, v in self.wnet.spectrum_proportion_derivatives_fast_approx()]
        )

    def update_positions_and_get_gradient(
        self,
        new_base: Distribution,
        new_targets: Sequence[Distribution],
    ):
        """Update peak positions, re-solve, and return position gradients.

        Gradients are of the network objective total_cost() = sum d**p * flow
        (i.e. W_p**p, not the rooted W_p). For the gradient of the literal W_p,
        multiply by (1/p) * total_cost()**(1/p - 1).

        Returns
        -------
        emp_grad : np.ndarray, shape [N_emp, DIM], dtype float64
            Gradient of total cost w.r.t. each empirical peak position.
        theo_grads : list of np.ndarray, each shape [N_k, DIM], dtype float64
            Gradient w.r.t. each theoretical spectrum's peak positions.

        Raises
        ------
        logic_error
            If the network was built with the 1D chain factory (use force_dense_1d=True
            or DIM >= 2 to get gradients in 1D).
        """
        import numpy as np

        new_vec_base = new_base.vecdist
        new_vec_targets = [t.vecdist for t in new_targets]
        emp_grad_raw, theo_grads_raw = self.wnet.update_positions_and_get_gradient(
            new_vec_base, new_vec_targets, self._distance
        )
        emp_grad = np.asarray(emp_grad_raw)
        theo_grads = [np.asarray(g) for g in theo_grads_raw]
        return emp_grad, theo_grads

    def update_positions_and_solve(
        self,
        new_base: Distribution,
        new_targets: Sequence[Distribution],
    ) -> None:
        """Update peak positions and immediately re-solve (warm-restarting if possible).

        Keeps graph topology and intensities fixed; only edge costs change.
        The new distributions must have the same number of peaks as the originals.

        For 1D (chain) networks, peak sorted order must be preserved — raises
        ValueError if any peak has crossed another since construction.
        """
        new_vec_base = new_base.vecdist
        new_vec_targets = [t.vecdist for t in new_targets]
        self.wnet.update_positions_and_solve(
            new_vec_base, new_vec_targets, self._distance
        )

    def subgraphs(self) -> list["SubgraphWrapper"]:
        """
        Returns a list of SubgraphWrapper instances, each representing a subgraph of the network.
        Returns:
            List[SubgraphWrapper]: A list containing wrapped subgraph objects.
        """

        return [
            SubgraphWrapper(self.wnet.get_subgraph(i))
            for i in range(self.wnet.no_subgraphs())
        ]


class SubgraphWrapper:
    """
    A wrapper class for subgraph objects (implemented in C++), providing additional methods for visualization and conversion to NetworkX graphs.

    Args:
        obj: The subgraph object to wrap. Must implement `get_nodes()` and `get_edges()` methods.

    Attributes:
        _obj: The wrapped subgraph object.

    Methods:
        __getattr__(name):
            Delegates attribute access to the wrapped subgraph object.

        as_networkx():
            Converts the subgraph to a NetworkX directed graph (`DiGraph`), adding nodes and edges with relevant attributes.
            Node attributes: 'layer', 'type'.
            Edge attributes: 'capacity', 'weight'.

        show():
            Visualizes the subgraph using matplotlib and NetworkX.
            Nodes are colored based on their type ('source', 'sink', 'trash', or other).
            Edge labels display cost and capacity.
    """

    def __init__(self, obj: CWassersteinNetworkSubgraph) -> None:
        """
        Initializes the instance with a given CSubgraph object.
        Args:
            obj (CSubgraph): The subgraph object to be associated with this instance.
        """

        self._obj = obj

    def __getattr__(self, name):
        if name == "_obj":
            raise AttributeError("_obj")
        return getattr(self._obj, name)

    def as_networkx(self) -> "networkx.DiGraph":
        """Converts the subgraph to a NetworkX directed graph (DiGraph).
        If the graph is solved, it also includes flow information on the edges.

        Returns:
            networkx.DiGraph: A directed graph representation of the subgraph with nodes and edges.
        """
        import networkx as nx

        G = nx.DiGraph()
        for node in self.get_nodes():
            attrs = {"layer": node.layer(), "type": node.type_str()}
            attrs.update(_parse_node_meta(str(node)))
            G.add_node(node.get_id(), **attrs)
        if self.is_solved():
            flows = self.get_flow_map()
        for edge in self.get_edges():
            start = edge.get_start_node_id()
            end = edge.get_end_node_id()
            edge_data = {
                "capacity": edge.get_base_capacity(),
                "weight": edge.get_cost(),
            }
            if self.is_solved():
                edge_id = edge.get_id()
                if edge_id in flows:
                    edge_data["flow"] = flows[edge_id]
                else:
                    edge_data["flow"] = 0
            G.add_edge(start, end, **edge_data)
        return G

    def show(self) -> None:
        """Visualizes the subgraph using matplotlib and NetworkX.
        Nodes are colored based on their type ('source', 'sink', 'trash', or other).
        Edge labels display cost and capacity.
        """
        show_graph(self.as_networkx())

    def draw(self, **kwargs):
        """Interactive, draggable structure view (capacities + costs, no flow).

        See :func:`wnet.flow_viz.draw_network`.  Returns an object that renders
        inline in Jupyter; pass ``filename=...`` to save a standalone HTML file.
        """
        from wnet.flow_viz import draw_network

        return draw_network(self.as_networkx(), **kwargs)

    def draw_flow(self, **kwargs):
        """Interactive view of the solved flow (edge width/colour ∝ flow,
        saturated edges flagged).  See :func:`wnet.flow_viz.draw_flow`."""
        from wnet.flow_viz import draw_flow

        return draw_flow(self.as_networkx(), **kwargs)

    def draw_residual(self, **kwargs):
        """Interactive view of the residual network of the solved flow
        (forward vs. reverse/cancelling arcs).  See
        :func:`wnet.flow_viz.draw_residual`."""
        from wnet.flow_viz import draw_residual

        return draw_residual(self.as_networkx(), **kwargs)

    def residual_graph(self) -> "networkx.DiGraph":
        """Build the residual graph of the solved min-cost flow network.

        For each edge with remaining forward capacity, a forward residual edge
        is added.  For each edge with positive flow, a reverse residual edge
        (with negated cost) is added.  The resulting graph has no negative
        cycles (a property of optimal min-cost flow solutions).

        Raises:
            RuntimeError: If the subgraph has not been solved yet.
        """
        if not self.is_solved():
            raise RuntimeError(
                "residual_graph() requires a solved flow. Call solve() first."
            )
        import networkx as nx

        G = self.as_networkx()
        R = nx.DiGraph()
        for node, data in G.nodes(data=True):
            R.add_node(node, **data)
        for u, v, data in G.edges(data=True):
            if data["capacity"] is None or data["flow"] < data["capacity"]:
                R.add_edge(u, v, weight=data["weight"])
            if data["flow"] > 0:
                R.add_edge(v, u, weight=-data["weight"])
        return R

    def signal_part_derivatives(self) -> dict[int, dict[int, int]]:
        """Compute the marginal cost of increasing each theoretical signal by 1.

        Returns a nested dict mapping spectrum_id -> {peak_index -> derivative},
        where derivative is the change in total transport cost if that peak's
        intensity were increased by 1.  A negative derivative means increasing
        the signal *reduces* total cost (e.g. absorbing a trash unit).
        """
        ret: dict[int, dict[int, int]] = {}
        for spec_id, peak_idx, deriv in self._obj.signal_part_derivatives():
            ret.setdefault(spec_id, {})[peak_idx] = deriv
        return ret

    def spectrum_proportion_derivatives(self) -> np.ndarray:
        """Gradient of total cost w.r.t. scaling each spectrum's proportion.

        Returns array of derivatives indexed by spectrum_id (0..n-1).  The
        derivative approximates the cost change when every peak in the spectrum
        is scaled by (1 + eps): d(cost)/d(eps) at eps=0 =
        sum_i (peak_derivative_i * intensity_i).
        """
        return np.array([v for _, v in self._obj.spectrum_proportion_derivatives()])

    def signal_part_derivatives_fast_approx(self) -> dict[int, dict[int, int]]:
        """Fast, APPROXIMATE signal_part_derivatives() (dual-potential
        difference; different basis-dependent values, opt-in)."""
        ret: dict[int, dict[int, int]] = {}
        for spec_id, peak_idx, deriv in self._obj.signal_part_derivatives_fast_approx():
            ret.setdefault(spec_id, {})[peak_idx] = deriv
        return ret

    def spectrum_proportion_derivatives_fast_approx(self) -> np.ndarray:
        """Fast, APPROXIMATE spectrum_proportion_derivatives() (opt-in)."""
        return np.array(
            [v for _, v in self._obj.spectrum_proportion_derivatives_fast_approx()]
        )
