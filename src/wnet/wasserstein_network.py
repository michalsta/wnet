from typing import Optional
from collections.abc import Sequence

from wnet.wnet_cpp import (
    CWassersteinNetwork,
    CWassersteinNetworkSubgraph,
    CWassersteinNetworkFactory,
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

    def __str__(self) -> str:
        """Returns a string representation of the Wasserstein network."""
        return self.wnet.__str__()

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
                edge_data["flow"] = flows.get(edge.get_id(), 0)
            G.add_edge(start, end, **edge_data)
        return G

    def show(self) -> None:
        """Visualizes the subgraph using matplotlib and NetworkX.
        Nodes are colored based on their type ('source', 'sink', 'trash', or other).
        Edge labels display cost and capacity.
        """
        show_graph(self.as_networkx())

    def derivative_graph(self) -> "networkx.DiGraph":
        import networkx as nx
        G = self.as_networkx()
        derivative_G = nx.DiGraph()
        for node in G.nodes():
            print(G.nodes[node]["type"])
            if G.nodes[node]["type"] == "SinkNode":
                continue
            derivative_G.add_node(node, **G.nodes[node])
        for u, v, data in G.edges(data=True):
            if G.nodes[v]["type"] == "SinkNode":
                continue
            if data["capacity"] is None or data["flow"] < data["capacity"]:
                print(f"Adding edge from {u} to {v} with weight {data['weight']} and capacity {data['capacity']}")
                derivative_G.add_edge(u, v, **data)
            if data["flow"] > 0:
                print(f"Adding reverse edge from {v} to {u} with weight {-data['weight']} and capacity {data['flow']}")
                derivative_G.add_edge(v, u, capacity=data["flow"], weight=-data["weight"])
        return derivative_G

    def signal_part_derivatives(self) -> dict[int, int]:
        import networkx as nx
        G = self.derivative_graph()
        dist, _ = nx.single_source_bellman_ford(G, source=0, weight="weight")
        return dist