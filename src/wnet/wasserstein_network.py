from typing import Optional
from collections.abc import Sequence
from collections import defaultdict
import numpy as np

from wnet.wnet_cpp import (
    CWassersteinNetwork,
    CWassersteinNetworkSubgraph,
    CWassersteinNetworkFactory,
    TheoreticalNode,
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
    """

    def __init__(
        self,
        base_distribution: Distribution,
        target_distributions: Sequence[Distribution],
        distance: Distance,
        max_distance: Optional[float] = None,
    ) -> None:
        if max_distance is None or max_distance == float("inf"):
            max_distance = CWassersteinNetwork.max_value()
        self.wnet = CWassersteinNetworkFactory.create(
            base_distribution.vecdist(),
            [t.vecdist() for t in target_distributions],
            distance,
            max_distance,
        )
        self.add_simple_trash = self.wnet.add_simple_trash
        self.build = self.wnet.build
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

    def spectrum_proportion_derivatives(self) -> dict[int, int]:
        """Gradient of total cost w.r.t. scaling each spectrum's proportion.

        Aggregates across all subgraphs.  Returns spectrum_id -> derivative.
        """
        ret: dict[int, int] = {}
        for sg in self.subgraphs():
            for spec_id, deriv in sg.spectrum_proportion_derivatives().items():
                ret[spec_id] = ret.get(spec_id, 0) + deriv
        return ret

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
        """
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

        Returns a nested dict mapping spectrum_id → {peak_index → derivative},
        where derivative is the change in total transport cost if that peak's
        intensity were increased by 1.  A negative derivative means increasing
        the signal *reduces* total cost (e.g. absorbing a trash unit).

        Builds the full residual graph (excluding the trash edge) and uses
        Bellman-Ford shortest paths to find the cheapest way to route one
        more unit to each theoretical node.  The trash edge is handled
        separately via a cost adjustment depending on the supply/demand
        balance.
        """
        import networkx as nx
        R = self.residual_graph()
        trash_cost = self.simple_trash_cost()

        sink_id = None
        for node, data in R.nodes(data=True):
            if data["type"] == "SinkNode":
                sink_id = node
                break

        # Remove trash edges (source↔sink) — trash is accounted for
        # separately based on the supply/demand balance.
        if R.has_edge(0, sink_id):
            R.remove_edge(0, sink_id)
        if R.has_edge(sink_id, 0):
            R.remove_edge(sink_id, 0)

        nxG = self.as_networkx()
        emp_cap = sum(
            data["capacity"] for u, v, data in nxG.edges(data=True)
            if nxG.nodes[u]["type"] == "SourceNode" and v != sink_id
        )
        theo_cap = sum(
            data["capacity"] for u, v, data in nxG.edges(data=True)
            if v == sink_id and nxG.nodes[u]["type"] != "SourceNode"
        )

        # Two ways to deliver 1 extra unit to a theoretical node:
        #
        # 1) From source via unused empirical capacity (BF from source).
        #    If emp > theo, this also cancels a trash unit (−trash_cost).
        #    If theo ≥ emp, source supply increases; the extra supply
        #    unit goes to trash (+trash_cost) but the match itself is
        #    just the BF distance.
        #
        # 2) By rerouting existing flow from another theo node (BF from
        #    sink, which represents the "freed" unit at sink).  The
        #    freed sink capacity is compensated by a new trash unit
        #    (+trash_cost).  If emp > theo, the reroute frees a trash
        #    unit instead (−trash_cost from absorbing the freed emp flow).
        #
        # The derivative is the minimum of both options, capped at
        # trash_cost (can always just send the extra unit to trash).

        dist_src, _ = nx.single_source_bellman_ford(
            R, source=0, weight="weight"
        )
        dist_sink, _ = nx.single_source_bellman_ford(
            R, source=sink_id, weight="weight"
        )

        if emp_cap > theo_cap:
            src_adjust = -trash_cost  # absorbs a trash unit
            sink_adjust = 0           # pure reroute, no trash change
        else:
            src_adjust = 0            # source sends 1 more, no trash change
            sink_adjust = trash_cost  # reroute + 1 new trash unit

        # Build map of theo→sink slack (unused capacity).
        theo_sink_slack = {}
        for u, v, data in nxG.edges(data=True):
            if v == sink_id and nxG.nodes[u]["type"] != "SourceNode":
                theo_sink_slack[u] = data["capacity"] - data["flow"]

        ret = defaultdict(dict)
        for node in self.get_nodes():
            nt = node.get_type()
            if isinstance(nt, TheoreticalNode):
                node_id = node.get_id()
                if theo_sink_slack.get(node_id, 0) > 0 and emp_cap > theo_cap:
                    # The solver had excess supply but chose not to
                    # fill this node — adding capacity changes nothing.
                    ret[nt.get_spectrum_id()][nt.get_peak_index()] = 0
                    continue
                candidates = [trash_cost]
                if node_id in dist_src:
                    candidates.append(dist_src[node_id] + src_adjust)
                if node_id in dist_sink:
                    candidates.append(dist_sink[node_id] + sink_adjust)
                ret[nt.get_spectrum_id()][nt.get_peak_index()] = min(candidates)
        return ret

    def spectrum_proportion_derivatives(self) -> dict[int, int]:
        """Gradient of total cost w.r.t. scaling each spectrum's proportion.

        Returns spectrum_id -> derivative.  The derivative approximates the
        cost change when every peak in the spectrum is scaled by (1 + eps):
        d(cost)/d(eps) at eps=0 = sum_i (peak_derivative_i * intensity_i).
        """
        peak_derivs = self.signal_part_derivatives()
        ret: dict[int, int] = {}
        for node in self.get_nodes():
            nt = node.get_type()
            if isinstance(nt, TheoreticalNode):
                spec_id = nt.get_spectrum_id()
                peak_idx = nt.get_peak_index()
                intensity = nt.get_intensity()
                ret[spec_id] = ret.get(spec_id, 0) + peak_derivs[spec_id][peak_idx] * intensity
        return ret