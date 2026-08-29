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

    The majority of functionality is implemented in the underlying C++ class `CWassersteinNetwork`, which this class wraps.
    The C++ network is created lazily in :meth:`build` (trash costs, declared via ``add_*_trash()`` between construction
    and ``build()``, feed both the factory choice and the automatic intensity scale).

    Distance-cap semantics — exactly one of two named parameters:

    * ``max_distance`` (dense semantics, the default): a **per-pair matching threshold**.  Mass is never transported
      between an empirical and a theoretical peak farther apart than ``max_distance``, guaranteed.  Internally the O(m+n)
      1D chain factory may still be used, but only when provably equivalent to the dense factory: when the cap is
      None/infinite, or when both-side trash exists (simple, or experimental+theoretical) and ``max_distance`` is at
      least the sum of the two per-unit trash costs (simple trash ``t`` counts as ``2t``; asymmetric as ``t_exp +
      t_theo``) — beyond that distance transport is dominated by trashing both sides, so the cap is provably inactive.
      Otherwise the dense factory is used.
    * ``split_distance``: **explicit chain semantics**.  The merged 1D peak sequence is split into independent components
      wherever the gap between *consecutive* peaks exceeds ``split_distance``; within a component mass may legally ride
      the chain arbitrarily far (multi-hop), i.e. this is a component-splitting radius, not a per-pair cap.  Requires 1D
      data, ``p == 1``, ``force_dense_1d=False`` and a chain-compatible solver (NetworkSimplex, CycleCanceling, SlopeDP);
      anything else raises ValueError — there is no silent dense fallback, since chain semantics were requested by name.

    Passing both ``max_distance`` and ``split_distance`` raises ValueError.

    Args:
        base_distribution (Distribution): The base distribution from which the Wasserstein distance is computed.
        target_distributions (Sequence[Distribution]): A sequence of target distributions to which the Wasserstein distance is computed.
        distance (DistanceFunction): A callable that computes the distance between points in the distributions.
        max_distance (float | None): Per-pair matching threshold (dense semantics, see above). If None or infinity, no cap.
        force_dense_1d (bool): In 1D, force the O(m*n) dense factory instead of the O(m+n) chain factory; the chain factory is never used. Incompatible with split_distance.
        p (float): Wasserstein transport order, any real number >= 1. Each matching edge costs ground_distance**p, so total_cost() and all derivatives are in W_p**p units; take the p-th root for the literal W_p distance (the high-level WassersteinDistance() does this). For p != 1 the cost is fractional, so the integer solver works in auto-scaled units (round(scale_factor() * d**p)); the public total_cost()/derivatives divide that back out. p == 1 is bit-exact with the legacy 1-Wasserstein (scale_factor() == 1, truncation). p != 1 always uses the dense factory (the 1D chain factory is invalid for p != 1, since exponentiated gap costs are not additive).
        solver: Solver configuration object. One of NetworkSimplex(), CostScaling(), CycleCanceling(), CapacityScaling(), or SlopeDP(). Defaults to NetworkSimplex() (warm restarts, BLOCK_SEARCH pivot). SlopeDP is chain-native: it requires either split_distance or a max_distance for which the chain factory is provably equivalent (see above).
        intensity_scale (float | None): None => auto (FineGridScaler, chosen at build() time so it sees the declared trash costs); an explicit value is used verbatim.
        round_max_distance (bool): Round a fractional cap up to an integer (legacy p == 1 truncation behaviour) with a warning. Applied to whichever of max_distance/split_distance was given.
        split_distance (float | None): Explicit chain-semantics split radius (see above).
    """

    _SOLVER_METHODS = {
        "network_simplex": NetworkSimplex,
        "cycle_canceling": CycleCanceling,
        "cost_scaling": CostScaling,
        "capacity_scaling": CapacityScaling,
        "slope_dp": SlopeDP,
    }

    # C++ methods reachable directly on the wrapper once build() has run
    # (previously bound-method aliases assigned in __init__; delegated lazily
    # now that the C++ network is created in build()).
    _CPP_DELEGATES = frozenset({
        "solve",
        "scale_factor",
        "intensity_scale_factor",
        "get_subgraph",
        "no_subgraphs",
        "flows_for_target",
        "count_empirical_nodes",
        "count_theoretical_nodes",
        "matching_density",
        "lemon_to_string",
        "count_matching_edges",
        "count_theoretical_to_sink_edges",
        "count_src_to_empirical_edges",
        "count_simple_trash_edges",
        "count_chain_edges",
        "no_theoretical_spectra",
        "theoretical_spectra_sizes",
    })

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
        split_distance: Optional[float] = None,
    ) -> None:
        # Must exist before anything that could trigger __getattr__.
        self._wnet_obj = None
        if solver is None and method is None:
            solver = NetworkSimplex()
        elif solver is None:
            if method not in self._SOLVER_METHODS:
                raise ValueError(
                    f"Unknown method {method!r}. Choose from: {list(self._SOLVER_METHODS)}"
                )
            solver = self._SOLVER_METHODS[method]()
        elif method is not None:
            raise ValueError(
                "Pass either solver= or method=, not both "
                f"(got solver={solver!r} and method={method!r})."
            )
        if max_distance is not None and split_distance is not None:
            raise ValueError(
                "Pass either max_distance (per-pair matching threshold, dense "
                "semantics) or split_distance (chain component-splitting "
                "radius), not both."
            )
        self._explicit_split = split_distance is not None
        cap = split_distance if self._explicit_split else max_distance
        cap_name = "split_distance" if self._explicit_split else "max_distance"
        if cap is None or cap == float("inf"):
            self._cap_is_inf = True
            cap = CWassersteinNetwork.max_value()
        else:
            self._cap_is_inf = False
            if round_max_distance:
                # Default: with p == 1 cost truncation the matching threshold is
                # effectively integer, so round a fractional cap up (inclusive)
                # and warn — preserving the legacy behaviour.  Callers that opt
                # into cost scaling (real distances) pass
                # round_max_distance=False and keep the real threshold.
                md_int = int(math.ceil(cap))
                if md_int != cap:
                    warnings.warn(
                        f"{cap_name}={cap!r} is not an integer; the solver "
                        f"uses an integer distance threshold, so it was rounded up to "
                        f"{md_int}. Pass an integer {cap_name} to avoid this.",
                        stacklevel=2,
                    )
                cap = md_int
        p = float(p)
        if not (p >= 1.0) or not np.isfinite(p):
            raise ValueError(
                f"Wasserstein order p must be a real number >= 1, got {p!r}."
            )
        self._distance = distance
        self._p = p
        self._cap = cap
        self._base_distribution = base_distribution
        self._target_distributions = list(target_distributions)
        self._force_dense_1d = bool(force_dense_1d)
        self._solver = solver
        self._intensity_scale_arg = intensity_scale
        # Trash costs / pre-build settings are recorded here and replayed onto
        # the C++ network in build() — the factory gate and the auto intensity
        # scale both need the trash costs, which arrive between __init__ and
        # build().
        self._simple_trash_cost: Optional[float] = None
        self._exp_trash_cost: Optional[float] = None
        self._theo_trash_cost: Optional[float] = None
        self._independent_trash_costs: Optional[tuple] = None
        self._cost_scaling_arg: Optional[int] = None
        self._flow_budget_arg: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Pre-build configuration (recorded, replayed onto the C++ network in
    # build()).  Validation mirrors the C++ methods so errors surface at the
    # call site, not at build().
    # ------------------------------------------------------------------ #

    @property
    def wnet(self):
        """The underlying C++ network. Available after build()."""
        if self._wnet_obj is None:
            raise RuntimeError(
                "The C++ network does not exist yet: call build() first "
                "(trash edges and pre-build settings are applied at build time)."
            )
        return self._wnet_obj

    def _check_not_built(self, what: str) -> None:
        if self._wnet_obj is not None:
            raise RuntimeError(f"{what}() must be called before build(), not after.")

    def add_simple_trash(self, cost: float) -> None:
        self._check_not_built("add_simple_trash")
        if self._simple_trash_cost is not None:
            raise RuntimeError("Simple trash edge already added.")
        if self._exp_trash_cost is not None or self._theo_trash_cost is not None:
            raise RuntimeError(
                "add_simple_trash() is exclusive with experimental/theoretical trash."
            )
        if self._independent_trash_costs is not None:
            raise RuntimeError(
                "add_simple_trash() is exclusive with independent trash."
            )
        self._simple_trash_cost = float(cost)

    def add_experimental_trash(self, cost: float) -> None:
        self._check_not_built("add_experimental_trash")
        if self._simple_trash_cost is not None:
            raise RuntimeError(
                "add_experimental_trash() is exclusive with simple trash."
            )
        if self._independent_trash_costs is not None:
            raise RuntimeError(
                "add_experimental_trash() is exclusive with independent trash."
            )
        if self._exp_trash_cost is not None:
            raise RuntimeError("Experimental trash already added.")
        self._exp_trash_cost = float(cost)

    def add_theoretical_trash(self, cost: float) -> None:
        self._check_not_built("add_theoretical_trash")
        if self._simple_trash_cost is not None:
            raise RuntimeError(
                "add_theoretical_trash() is exclusive with simple trash."
            )
        if self._independent_trash_costs is not None:
            raise RuntimeError(
                "add_theoretical_trash() is exclusive with independent trash."
            )
        if self._theo_trash_cost is not None:
            raise RuntimeError("Theoretical trash already added.")
        self._theo_trash_cost = float(cost)

    def add_independent_asymmetric_trash(self, C_exp: float, C_theo: float) -> None:
        """Independent asymmetric trash (dualdeconv4 semantics): every
        discarded empirical unit costs C_exp and every phantom-filled
        theoretical unit C_theo, charged independently — an excess pair costs
        C_exp + C_theo, never the annihilating model's min(C_exp, C_theo).
        Requires the dense factory (forced automatically); exclusive with the
        other trash models and with explicit chain semantics. Must be called
        before build()."""
        self._check_not_built("add_independent_asymmetric_trash")
        if (
            self._simple_trash_cost is not None
            or self._exp_trash_cost is not None
            or self._theo_trash_cost is not None
        ):
            raise RuntimeError(
                "add_independent_asymmetric_trash() is exclusive with the "
                "simple/experimental/theoretical trash models."
            )
        if self._independent_trash_costs is not None:
            raise RuntimeError("Independent trash already added.")
        if self._explicit_split:
            raise ValueError(
                "Independent asymmetric trash requires the dense factory; it "
                "cannot be combined with explicit chain semantics "
                "(split_distance)."
            )
        self._independent_trash_costs = (float(C_exp), float(C_theo))

    def set_cost_scaling(self, scale: int = 0) -> None:
        """Opt-in p=1 cost scaling (lets a caller pass real distances instead
        of pre-scaling positions). scale <= 0 => auto. Must be called before
        build()."""
        self._check_not_built("set_cost_scaling")
        self._cost_scaling_arg = int(scale)

    def set_flow_budget(self, flow: float) -> None:
        """Declared upper bound on point-scaled total flow (real intensity
        units): build() sizes the cost scale so any point within the budget
        stays inside the int64 cost accumulator, and solve() rejects points
        past the ceiling. Must be called before build()."""
        if self._wnet_obj is not None:
            raise RuntimeError("set_flow_budget() must be called before build().")
        flow = float(flow)
        if not np.isfinite(flow) or flow < 0.0:
            raise ValueError("set_flow_budget: flow must be finite and >= 0.")
        self._flow_budget_arg = flow

    # ------------------------------------------------------------------ #
    # Factory decision + build
    # ------------------------------------------------------------------ #

    def _active_trash_costs(self) -> list:
        costs = [
            c
            for c in (
                self._simple_trash_cost,
                self._exp_trash_cost,
                self._theo_trash_cost,
            )
            if c is not None
        ]
        if self._independent_trash_costs is not None:
            costs.extend(self._independent_trash_costs)
        return costs

    def _decide_chain(self) -> bool:
        """True => use the 1D chain factory.  Raises for impossible requests."""
        solver = self._solver
        chain_capable_solver = not isinstance(solver, (CostScaling, CapacityScaling))
        if self._explicit_split:
            # Chain semantics requested by name: no silent dense fallback.
            problems = []
            if self._base_distribution.dimension != 1:
                problems.append(
                    f"data is {self._base_distribution.dimension}D (need 1D)"
                )
            if self._p != 1.0:
                problems.append(f"p={self._p} (need p == 1)")
            if self._force_dense_1d:
                problems.append("force_dense_1d=True")
            if not chain_capable_solver:
                # CostScaling / CapacityScaling cannot solve the chain: the
                # bidirectional chain arcs carry unbounded (INF) capacity, on
                # which these two LEMON algorithms return INFEASIBLE (garbage
                # total cost) or loop.
                problems.append(
                    f"solver {type(solver).__name__} cannot solve the chain "
                    "(use NetworkSimplex, CycleCanceling or SlopeDP)"
                )
            if problems:
                raise ValueError(
                    "split_distance requests explicit chain semantics, but: "
                    + "; ".join(problems)
                    + "."
                )
            return True
        # max_distance semantics: dense per-pair threshold, GUARANTEED.  The
        # chain factory is an invisible implementation detail, allowed ONLY
        # when the cap is absent (None/INF): then dense components and chain
        # runs coincide and the LPs are identical.  A finite cap always builds
        # dense.  A trash-cost threshold gate (cap >= 2t) was tried in 1.3.0
        # and REFUTED by fuzz: even with transport-beyond-cap dominated by
        # double-trashing, the two factories PARTITION peaks differently — a
        # peak bridged into a chain run by same-side gaps <= cap but with no
        # cross-side peak within cap is charged trash by dense (isolated
        # dead-end) yet absorbed free against other-side excess by the chain
        # LP (per-subgraph supply = max(E, T)).  Repro: emp {0,90,180} x1,
        # theo {270} x5, simple trash 50, cap 100 -> chain 250, dense 350.
        # No cap multiple fixes that; a sound gate needs partition-parity
        # (chain runs == dense components + identical dead-end sets), which is
        # future work.
        chain_possible = (
            self._base_distribution.dimension == 1
            and self._p == 1.0
            and chain_capable_solver
            and not self._force_dense_1d
            # Independent trash charges a per-matched-unit cost shift that
            # cannot ride per-hop chain arcs: dense only.
            and self._independent_trash_costs is None
            and self._cap_is_inf
        )
        if isinstance(self._solver, SlopeDP) and not chain_possible:
            raise ValueError(
                "SlopeDP is chain-native, but the requested configuration "
                "cannot use the 1D chain factory (it needs 1D data, p == 1, "
                "force_dense_1d=False, and no max_distance — a finite "
                "max_distance means per-pair dense semantics). "
                "Pass split_distance=... for explicit chain semantics, or "
                "drop max_distance."
            )
        return chain_possible

    def build(self) -> None:
        """Create the C++ network (factory chosen from the recorded settings),
        replay trash edges and pre-build settings onto it, and build it."""
        use_chain = self._decide_chain()
        vec_base = self._base_distribution.vecdist
        vec_targets = [t.vecdist for t in self._target_distributions]
        if use_chain:
            wnet = CWassersteinNetworkFactory.create_1d(
                vec_base, vec_targets, self._distance, self._cap, self._p
            )
        else:
            wnet = CWassersteinNetworkFactory.create(
                vec_base, vec_targets, self._distance, self._cap, self._p
            )
        # Intensities are carried as real doubles into the int64 solver, which
        # quantizes them to integer supplies as trunc(real * intensity_scale)
        # (round toward zero).
        # None => auto: 1.0 for integer-valued intensities (bit-compatible) or
        # p != 1 (the joint cost/intensity budget is future work), else a scale
        # that lifts fractional intensities onto a fine integer grid without
        # overflowing the cost accumulator — the scaler runs here, at build
        # time, so it sees the declared trash costs (a large trash cost
        # shrinks the safe intensity grid).  An explicit value is used
        # verbatim.
        intensity_scale = self._intensity_scale_arg
        if intensity_scale is None:
            intensity_scale = FineGridScaler(
                self._base_distribution,
                self._target_distributions,
                self._distance,
                self._cap,
                trash_costs=self._active_trash_costs(),
                p=self._p,
            ).sf_intensity()
        wnet.set_intensity_scale(float(intensity_scale))
        if self._cost_scaling_arg is not None:
            wnet.set_cost_scaling(self._cost_scaling_arg)
        if self._flow_budget_arg is not None:
            wnet.set_flow_budget(self._flow_budget_arg)
        if self._simple_trash_cost is not None:
            wnet.add_simple_trash(self._simple_trash_cost)
        if self._exp_trash_cost is not None:
            wnet.add_experimental_trash(self._exp_trash_cost)
        if self._theo_trash_cost is not None:
            wnet.add_theoretical_trash(self._theo_trash_cost)
        if self._independent_trash_costs is not None:
            wnet.add_independent_asymmetric_trash(*self._independent_trash_costs)
        wnet.build(self._solver)
        self._wnet_obj = wnet

    def __getattr__(self, name):
        # Lazily delegate the C++ surface (only consulted when normal
        # attribute lookup fails; bound methods are created per access, so no
        # reference cycle keeps the wrapper alive — see the Python 3.14
        # incremental-GC note in the git history).
        if name.startswith("_"):
            raise AttributeError(name)
        if name in WassersteinNetwork._CPP_DELEGATES:
            return getattr(self.wnet, name)  # raises RuntimeError pre-build
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

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
        # The C++ accumulates real-distance derivatives weighted by the INTEGER
        # solver flows, which carry the intensity scale; divide it back out so
        # the gradients match total_cost()'s real W_p**p units (the cost scale
        # never enters — distances are differentiated un-scaled).
        s = self.wnet.intensity_scale_factor()
        emp_grad = np.asarray(emp_grad_raw) / s
        theo_grads = [np.asarray(g) / s for g in theo_grads_raw]
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

    def residual_graph(self) -> "networkx.MultiDiGraph":
        """Build the residual graph of the solved min-cost flow network.

        For each edge with remaining forward capacity, a forward residual edge
        is added.  For each edge with positive flow, a reverse residual edge
        (with negated cost) is added.  The resulting graph has no negative
        cycles (a property of optimal min-cost flow solutions).

        Returns a ``networkx.MultiDiGraph``: with antiparallel original edges
        (e.g. the bidirectional arcs of the 1D chain factory) the forward
        residual of one edge and the reverse residual of its opposite share
        the same (u, v) node pair, so a plain DiGraph would silently drop one
        of them.

        Raises:
            RuntimeError: If the subgraph has not been solved yet.
        """
        if not self.is_solved():
            raise RuntimeError(
                "residual_graph() requires a solved flow. Call solve() first."
            )
        import networkx as nx

        G = self.as_networkx()
        R = nx.MultiDiGraph()
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
