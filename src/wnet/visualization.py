

def show_graph(G):
    import matplotlib.pyplot as plt
    import networkx as nx
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
        (u, v): f"cost: {d['weight']}\n capacity: {d['capacity']}" + (f"\n flow: {d['flow']}" if "flow" in d else "")
        for u, v, d in G.edges(data=True)
    }
    nx.draw(G, pos, with_labels=True, node_color=node_colors, arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()
