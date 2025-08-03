"""Graph utilities for SciEvo package.

This module provides utilities for constructing and analyzing citation graphs,
computing graph metrics, and handling network data structures.
"""

from typing import Dict, List, Tuple, Optional, Any

import networkx as nx
import numpy as np


def create_simple_graph_from_multigraph(G: nx.MultiGraph) -> nx.Graph:
    """Create a simple graph from a multigraph by combining edge weights.
    
    Converts a multigraph to a simple graph where multiple edges between
    the same nodes are combined into a single edge with aggregated weight.
    
    Args:
        G: Input multigraph to convert.
        
    Returns:
        Simple graph with combined edge weights.
        
    Example:
        >>> multi_g = nx.MultiGraph()
        >>> multi_g.add_edge(1, 2, weight=1)
        >>> multi_g.add_edge(1, 2, weight=2)  # Second edge between same nodes
        >>> simple_g = create_simple_graph_from_multigraph(multi_g)
        >>> print(simple_g[1][2]['weight'])  # 3
    """
    simple_G = nx.Graph()
    
    for u, v, data in G.edges(data=True):
        w = data.get('weight', 1)
        if simple_G.has_edge(u, v):
            simple_G[u][v]['weight'] += w
        else:
            simple_G.add_edge(u, v, weight=w)

    return simple_G


def build_citation_graph(citations_data: List[Dict[str, Any]]) -> nx.DiGraph:
    """Build a directed citation graph from citation data.
    
    Creates a directed graph where nodes represent papers and edges
    represent citation relationships.
    
    Args:
        citations_data: List of citation records containing source and target papers.
        
    Returns:
        Directed graph representing citation relationships.
        
    Example:
        >>> citations = [{"source": "paper1", "target": "paper2"}]
        >>> graph = build_citation_graph(citations)
        >>> print(graph.number_of_edges())
    """
    graph = nx.DiGraph()
    
    for citation in citations_data:
        source = citation.get('source') or citation.get('citing_paper')
        target = citation.get('target') or citation.get('cited_paper')
        
        if source and target:
            graph.add_edge(source, target)
    
    return graph


def compute_graph_metrics(graph: nx.Graph) -> Dict[str, Any]:
    """Compute basic metrics for a graph.
    
    Calculates various graph-level metrics including size, connectivity,
    and structural properties.
    
    Args:
        graph: NetworkX graph to analyze.
        
    Returns:
        Dictionary containing computed metrics.
        
    Example:
        >>> graph = nx.karate_club_graph()
        >>> metrics = compute_graph_metrics(graph)
        >>> print(metrics['num_nodes'])
    """
    metrics = {
        'num_nodes': graph.number_of_nodes(),
        'num_edges': graph.number_of_edges(),
        'density': nx.density(graph),
        'is_connected': nx.is_connected(graph) if not graph.is_directed() else nx.is_weakly_connected(graph),
    }
    
    # Add connectivity metrics
    if not graph.is_directed():
        if nx.is_connected(graph):
            metrics['diameter'] = nx.diameter(graph)
            metrics['average_shortest_path_length'] = nx.average_shortest_path_length(graph)
        else:
            metrics['diameter'] = None
            metrics['average_shortest_path_length'] = None
        
        metrics['num_connected_components'] = nx.number_connected_components(graph)
    else:
        if nx.is_weakly_connected(graph):
            metrics['diameter'] = nx.diameter(graph.to_undirected())
        else:
            metrics['diameter'] = None
        
        metrics['num_weakly_connected_components'] = nx.number_weakly_connected_components(graph)
        metrics['num_strongly_connected_components'] = nx.number_strongly_connected_components(graph)
    
    # Degree statistics
    degrees = dict(graph.degree())
    if degrees:
        degree_values = list(degrees.values())
        metrics['average_degree'] = np.mean(degree_values)
        metrics['max_degree'] = np.max(degree_values)
        metrics['min_degree'] = np.min(degree_values)
    
    return metrics


def extract_connected_components(graph: nx.Graph, min_size: int = 2) -> List[nx.Graph]:
    """Extract connected components from a graph.
    
    Returns subgraphs for each connected component that meets the minimum size requirement.
    
    Args:
        graph: Input graph to analyze.
        min_size: Minimum number of nodes for a component to be included.
        
    Returns:
        List of subgraphs representing connected components.
        
    Example:
        >>> graph = nx.Graph()
        >>> graph.add_edges_from([(1, 2), (3, 4)])
        >>> components = extract_connected_components(graph)
        >>> print(len(components))  # 2
    """
    if graph.is_directed():
        components = nx.weakly_connected_components(graph)
    else:
        components = nx.connected_components(graph)
    
    return [
        graph.subgraph(component).copy() 
        for component in components 
        if len(component) >= min_size
    ]


def compute_centrality_metrics(graph: nx.Graph, top_k: int = 10) -> Dict[str, Dict]:
    """Compute centrality metrics for nodes in the graph.
    
    Calculates various centrality measures to identify important nodes.
    
    Args:
        graph: Input graph to analyze.
        top_k: Number of top nodes to return for each centrality measure.
        
    Returns:
        Dictionary containing centrality measures and top nodes.
        
    Example:
        >>> graph = nx.karate_club_graph()
        >>> centrality = compute_centrality_metrics(graph, top_k=5)
        >>> print(list(centrality['degree'].keys())[:3])
    """
    metrics = {}
    
    # Degree centrality
    degree_centrality = nx.degree_centrality(graph)
    metrics['degree'] = dict(sorted(degree_centrality.items(), 
                                   key=lambda x: x[1], reverse=True)[:top_k])
    
    # Betweenness centrality (for smaller graphs)
    if graph.number_of_nodes() < 1000:
        betweenness_centrality = nx.betweenness_centrality(graph)
        metrics['betweenness'] = dict(sorted(betweenness_centrality.items(),
                                           key=lambda x: x[1], reverse=True)[:top_k])
    
    # Closeness centrality (for connected graphs)
    if nx.is_connected(graph) and graph.number_of_nodes() < 1000:
        closeness_centrality = nx.closeness_centrality(graph)
        metrics['closeness'] = dict(sorted(closeness_centrality.items(),
                                         key=lambda x: x[1], reverse=True)[:top_k])
    
    return metrics


def get_graph_statistics(graph: nx.Graph) -> Dict[str, Any]:
    """Get comprehensive statistics for a graph.
    
    Combines multiple analysis functions to provide a complete overview
    of graph properties.
    
    Args:
        graph: Graph to analyze.
        
    Returns:
        Dictionary containing comprehensive graph statistics.
        
    Example:
        >>> graph = nx.karate_club_graph()
        >>> stats = get_graph_statistics(graph)
        >>> print(f"Nodes: {stats['basic_metrics']['num_nodes']}")
    """
    stats = {
        'basic_metrics': compute_graph_metrics(graph),
        'components': len(extract_connected_components(graph)),
    }
    
    # Add centrality metrics for smaller graphs
    if graph.number_of_nodes() < 5000:
        stats['centrality'] = compute_centrality_metrics(graph)
    
    return stats


def filter_graph_by_degree(graph: nx.Graph, min_degree: int = 1) -> nx.Graph:
    """Filter graph by removing nodes with degree below threshold.
    
    Args:
        graph: Input graph to filter.
        min_degree: Minimum degree for nodes to keep.
        
    Returns:
        Filtered graph with low-degree nodes removed.
        
    Example:
        >>> graph = nx.path_graph(5)  # Linear chain
        >>> filtered = filter_graph_by_degree(graph, min_degree=2)
        >>> print(filtered.number_of_nodes())  # Endpoints removed
    """
    nodes_to_remove = [
        node for node, degree in graph.degree() 
        if degree < min_degree
    ]
    
    filtered_graph = graph.copy()
    filtered_graph.remove_nodes_from(nodes_to_remove)
    
    return filtered_graph


def create_keyword_cooccurrence_graph(
    keyword_lists: List[List[str]], 
    min_cooccurrence: int = 2
) -> nx.Graph:
    """Create a keyword co-occurrence graph from lists of keywords.
    
    Args:
        keyword_lists: List of keyword lists (e.g., from different papers).
        min_cooccurrence: Minimum co-occurrence count for edge creation.
        
    Returns:
        Graph where nodes are keywords and edges represent co-occurrence.
        
    Example:
        >>> keywords = [["ai", "ml"], ["ai", "nlp"], ["ml", "nlp"]]
        >>> graph = create_keyword_cooccurrence_graph(keywords)
        >>> print(list(graph.edges()))
    """
    from collections import defaultdict
    
    cooccurrence_counts = defaultdict(int)
    
    # Count co-occurrences
    for keywords in keyword_lists:
        unique_keywords = list(set(keywords))
        for i, kw1 in enumerate(unique_keywords):
            for kw2 in unique_keywords[i+1:]:
                pair = tuple(sorted([kw1, kw2]))
                cooccurrence_counts[pair] += 1
    
    # Create graph
    graph = nx.Graph()
    for (kw1, kw2), count in cooccurrence_counts.items():
        if count >= min_cooccurrence:
            graph.add_edge(kw1, kw2, weight=count)
    
    return graph