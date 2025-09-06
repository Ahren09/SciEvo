import networkx as nx

def create_simple_graph_from_multigraph(G: nx.MultiGraph) -> nx.Graph:
    """
    Create a simple graph from a multigraph.

    Args:
        G: nx.MultiGraph
            The input multigraph.

    Returns:
        nx.Graph
            The simple graph, where the 'weight' attribute is the number of edges between two nodes.
    """

    simple_G = nx.Graph()
    for u, v, data in G.edges(data=True):
        w = data['weight'] if 'weight' in data else 1
        if simple_G.has_edge(u, v):
            simple_G[u][v]['weight'] += w
        else:
            simple_G.add_edge(u, v, weight=w)

    return simple_G