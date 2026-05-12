from typing import Optional
from collections.abc import Sequence

from wnet.wnet_cpp import (
    CWassersteinNetwork,
    CWassersteinNetworkSubgraph,
    CWassersteinNetworkFactory,
    NetworkSimplex,
    CostScaling,
    CycleCanceling,
    CapacityScaling,
    NSPivotRule,
    CSMethod,
    CCMethod,
)
from wnet.distribution import Distribution
from wnet.distances import DistanceMetric, Distance
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
        solver: Solver configuration object. One of NetworkSimplex(), CostScaling(), CycleCanceling(), or CapacityScaling(). Defaults to NetworkSimplex() (warm restarts, BLOCK_SEARCH pivot).
    """

    _SOLVER_METHODS = {
        "network_simplex": NetworkSimplex,
        "cycle_canceling": CycleCanceling,
        "cost_scaling": CostScaling,
        "capacity_scaling": CapacityScaling,
    }

    def __init__(
        self,
        base_distribution: Distribution,
        target_distributions: Sequence[Distribution],
        distance: Distance,
        max_distance: Optional[float] = None,
        force_dense_1d: bool = False,
        solver=None,
        method: str = None,
    ) -> None:
        if solver is None and method is None:
            solver = NetworkSimplex()
        elif solver is None:
            if method not in self._SOLVER_METHODS:
                raise ValueError(f"Unknown method {method!r}. Choose from: {list(self._SOLVER_METHODS)}")
            solver = self._SOLVER_METHODS[method]()
        if max_distance is None or max_distance == float("inf"):
            max_distance = CWassersteinNetwork.max_value()
        vec_base = base_distribution.vecdist
        vec_targets = [t.vecdist for t in target_distributions]
        if base_distribution.dimension == 1 and not force_dense_1d:
            self.wnet = CWassersteinNetworkFactory.create_1d(
                vec_base, vec_targets, distance, max_distance)
        else:
            self.wnet = CWassersteinNetworkFactory.create(
                vec_base, vec_targets, distance, max_distance)
        self.add_simple_trash = self.wnet.add_simple_trash
        self.add_experimental_trash = self.wnet.add_experimental_trash
        self.add_theoretical_trash = self.wnet.add_theoretical_trash

        # Avoid capturing self in the lambda to prevent reference cycles that could lead to memory leaks.  The underlying C++ object should be freed when this wrapper is freed, but if we capture self in the lambda, the lambda's reference to self would keep it alive indefinitely.
        # Without this trick, the lambda would hold a reference to self, which holds a reference to the C++ object, which holds a reference back to the lambda, creating a cycle that prevents garbage collection.
        # Without this, the incremental GC introduced in Python3.14 can't collect WassersteinNetwork instances that are no longer needed, leading to memory leaks.
        _wnet = self.wnet  # avoid capturing self in the lambda (reference cycle).
        _solver = solver
        self.build = lambda: _wnet.build(_solver)

        self.solve = self.wnet.solve
        self.total_cost = self.wnet.total_cost
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
        self.no_theoretical_spectra = self.wnet.no_theoretical_spectra
        self.theoretical_spectra_sizes = self.wnet.theoretical_spectra_sizes

    def __str__(self) -> str:
        """Returns a string representation of the Wasserstein network."""
        return self.wnet.__str__()

    def signal_part_derivatives(self) -> dict[int, dict[int, int]]:
        """Compute the marginal cost of increasing each theoretical signal by 1.

        Returns a nested dict mapping spectrum_id -> {peak_index -> derivative}.
        Aggregates across all subgraphs.
        """
        ret: dict[int, dict[int, int]] = {}
        for spec_id, peak_idx, deriv in self.wnet.signal_part_derivatives():
            ret.setdefault(spec_id, {})[peak_idx] = deriv
        return ret

    def spectrum_proportion_derivatives(self) -> dict[int, int]:
        """Gradient of total cost w.r.t. scaling each spectrum's proportion.

        Aggregates across all subgraphs.  Returns spectrum_id -> derivative.
        """
        return dict(self.wnet.spectrum_proportion_derivatives())

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
        if name == '_obj':
            raise AttributeError('_obj')
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
            G.add_node(node.get_id(), layer=node.layer(), type=node.type_str())
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

    def spectrum_proportion_derivatives(self) -> dict[int, int]:
        """Gradient of total cost w.r.t. scaling each spectrum's proportion.

        Returns spectrum_id -> derivative.  The derivative approximates the
        cost change when every peak in the spectrum is scaled by (1 + eps):
        d(cost)/d(eps) at eps=0 = sum_i (peak_derivative_i * intensity_i).
        """
        return dict(self._obj.spectrum_proportion_derivatives())