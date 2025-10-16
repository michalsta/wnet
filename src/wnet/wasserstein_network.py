from typing import Optional
from collections.abc import Sequence

from wnet.wnet_cpp import CWassersteinNetwork, CWassersteinNetworkSubgraph
from wnet.distribution import Distribution
from wnet.distances import Distance


class WassersteinNetwork(CWassersteinNetwork):
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
        super().__init__(
            base_distribution, target_distributions, distance, max_distance
        )

    def subgraphs(self) -> list["SubgraphWrapper"]:
        """
        Returns a list of SubgraphWrapper instances, each representing a subgraph of the network.
        Returns:
            List[SubgraphWrapper]: A list containing wrapped subgraph objects.
        """

        return [
            SubgraphWrapper(self.get_subgraph(i)) for i in range(self.no_subgraphs())
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

        as_netowkrx():
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

    def as_netowkrx(self) -> "networkx.DiGraph":
        """Converts the subgraph to a NetworkX directed graph (DiGraph).
        Returns:
            networkx.DiGraph: A directed graph representation of the subgraph with nodes and edges.
        """
        import networkx as nx

        G = nx.DiGraph()
        for node in self.get_nodes():
            G.add_node(node.get_id(), layer=node.layer(), type=node.type_str())
        for edge in self.get_edges():
            start = edge.get_start_node_id()
            end = edge.get_end_node_id()
            G.add_edge(
                start, end, capacity=edge.get_base_capacity(), weight=edge.get_cost()
            )
        return G

    def show(self) -> None:
        """Visualizes the subgraph using matplotlib and NetworkX.
        Nodes are colored based on their type ('source', 'sink', 'trash', or other).
        Edge labels display cost and capacity.
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        G = self.as_netowkrx()
        pos = nx.multipartite_layout(G, subset_key="layer")
        node_colors = []
        for _, data in G.nodes(data=True):
            if data["type"] == "source":
                node_colors.append("lightgreen")
            elif data["type"] == "sink":
                node_colors.append("lightcoral")
            elif data["type"] == "trash":
                node_colors.append("lightgray")
            else:
                node_colors.append("lightblue")
        edge_labels = {
            (u, v): f"cost: {d['weight']}\n capacity: {d['capacity']}"
            for u, v, d in G.edges(data=True)
        }
        nx.draw(G, pos, with_labels=True, node_color=node_colors, arrows=True)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.show()
