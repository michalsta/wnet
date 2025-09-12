from wnet.wnet_cpp import CWassersteinNetwork

class WassersteinNetwork(CWassersteinNetwork):
    def __init__(self, base_distribution, target_distributions, distance, max_distance=None):
        if max_distance is None:
            max_distance = CWassersteinNetwork.max_value()
        super().__init__(base_distribution, target_distributions, distance, max_distance)

    def subgraphs(self):
        return [SubgraphWrapper(self.get_subgraph(i)) for i in range(self.no_subgraphs())]


class SubgraphWrapper:
    def __init__(self, obj):
        self._obj = obj

    def __getattr__(self, name):
        return getattr(self._obj, name)

    def as_netowkrx(self):
        import networkx as nx
        G = nx.DiGraph()
        for node in self.get_nodes():
            G.add_node(node.get_id(), layer=node.layer(), type=node.type_str())
        for edge in self.get_edges():
            start = edge.get_start_node_id()
            end = edge.get_end_node_id()
            G.add_edge(start, end, capacity=edge.get_base_capacity(), weight=edge.get_cost())
        return G

    def show(self):
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
        edge_labels = {(u, v): f"cost: {d['weight']}\n capacity: {d['capacity']}" for u, v, d in G.edges(data=True)}
        nx.draw(G, pos, with_labels=True, node_color=node_colors, arrows=True)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.show()
