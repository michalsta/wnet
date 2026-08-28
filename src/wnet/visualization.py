def show_graph(G):
    import matplotlib.pyplot as plt
    import networkx as nx

    # Node "type" attributes carry the C++ type_str() values ("SourceNode",
    # "SinkNode", ...); classify them with the same substring matching the
    # interactive views use.
    from wnet.flow_viz import _role

    _ROLE_COLORS = {
        "source": "lightgreen",
        "sink": "lightcoral",
        "trash": "lightgray",
    }

    pos = nx.multipartite_layout(G, subset_key="layer")
    node_colors = [
        _ROLE_COLORS.get(_role(data.get("type")), "lightblue")
        for _, data in G.nodes(data=True)
    ]
    edge_labels = {
        (u, v): f"cost: {d['weight']}\n capacity: {d['capacity']}"
        + (f"\n flow: {d['flow']}" if "flow" in d else "")
        for u, v, d in G.edges(data=True)
    }
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        connectionstyle="arc3,rad=0.1",
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()


def print_graph(G):
    import networkx as nx

    for node in G.nodes(data=True):
        print(f"Node {node[0]}: {node[1]}")
    for edge in G.edges(data=True):
        print(f"Edge from {edge[0]} to {edge[1]}: {edge[2]}")
