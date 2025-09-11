from wnet.wnet_cpp import CWassersteinNetwork

class WassersteinNetwork(CWassersteinNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            
        print("Edges:", list(self.get_edges()))


def monkeypatch_subgraph(subgraph):
    def as_netowkrx(self):
        import networkx as nx
        G = nx.DiGraph()
        for u, v, cap, cost in self.edges():
            G.add_edge(u, v, capacity=cap, weight=cost)
        return G
    subgraph.as_netowkrx = as_netowkrx
    return subgraph