import os
import os.path as osp
import sys
from time import time

import community  # pip install python-louvain
import networkx as nx
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath('.'))

from utility.graph_utils import create_simple_graph_from_multigraph
from utility.utils_data import load_arXiv_data, load_keywords
from arguments import parse_args
from utility.utils_misc import project_setup


def build_hypergraph(keywords, backend="networkx"):
    assert backend in ["networkx", "rapids"]

    # Preparing edge data
    edge_list = []

    for index_paper, (arxiv_url, row) in enumerate(tqdm(keywords.iterrows(), total=len(keywords))):

        if isinstance(row['keywords'], str):
            keywords_one_paper = row['keywords'].split(",")
        else:
            assert isinstance(row['keywords'], list)
            keywords_one_paper = row['keywords']

        keywords_one_paper = [k.strip().lower() for k in keywords_one_paper]
        # Add edges between every pair of keywords to form a clique

        published = arxiv_data.loc[arxiv_url, "published"]
        for i in range(len(keywords_one_paper)):
            for j in range(i + 1, len(keywords_one_paper)):
                edge_list.append(
                    (keywords_one_paper[i], keywords_one_paper[j], index_paper, published.year, published.month))

    df = pd.DataFrame(edge_list, columns=['source', 'destination', 'index_paper', 'published_year', 'published_month'])
    return df


if __name__ == "__main__":
    project_setup()
    args = parse_args()

    path_graph = osp.join(args.output_dir, f'{args.feature_name}_edges.parquet')

    if not osp.exists(path_graph):
        path = os.path.join(args.data_dir, "NLP", "arXiv", f"{args.feature_name}_keywords.json")
        arxiv_data = load_arXiv_data(args.data_dir).set_index("id")

        keywords = load_keywords(args.data_dir, args.feature_name)

        edge_df = build_hypergraph(keywords, backend=args.graph_backend)
        print("Saving hypergraph edges to parquet file")

        edge_df.to_parquet(path_graph)

    else:
        print("Loading hypergraph edges from parquet file")
        edge_df = pd.read_parquet(path_graph)

    print(f"Building the graph ...", end=" ")
    t0 = time()

    if args.graph_backend == "networkx":
        stats_d = []
        # Construct snapshot for each year
        year = 1985
        while year < 2025:
            G = nx.MultiGraph()

            if year <= 1990:
                edges = edge_df.query("1985 <= published_year <= 1990")
                year = 1990

            else:
                edges = edge_df.query(f"published_year == {year}")

            if len(edges) == 0:
                continue

            # Add edges to the graph
            G.add_edges_from(edges[['source', 'destination', 'index_paper']].values)

            # Get node mappings
            nodes_mapping = list(G.nodes())
            node_index_to_id = {node_id: node_index for node_index, node_id in enumerate(nodes_mapping)}

            # Recreate the graph as a multigraph
            multi_G = nx.MultiGraph()
            multi_G.add_edges_from(G.edges(data=True))

            # Create a simple graph with 'weight' attribute as the number of edges between two nodes
            simple_G = nx.Graph()

            for u, v, data in multi_G.edges(data=True):
                if simple_G.has_edge(u, v):
                    simple_G[u][v]['weight'] += 1
                else:
                    simple_G.add_edge(u, v, weight=1)

            # Calculate average degree
            degrees = dict(multi_G.degree())
            average_degree = sum(degrees.values()) / len(degrees)

            # Degree centrality: Number of edges connected to a node
            # Useful for finding the most important nodes in a network
            degree_centrality = nx.degree_centrality(simple_G)

            # Betweenness centrality
            # Take every pair of nodes in the graph and count how many times a node can interrupt the shortest paths
            # Useful for finding the "bridging" concepts
            betweenness_centrality = nx.betweenness_centrality(simple_G)
            betweenness_centrality_df = pd.Series(betweenness_centrality).sort_values(ascending=False)

            # Closeness centrality: Average shortest distance between each person in a network.
            closeness_centrality = nx.closeness_centrality(simple_G)

            # Modularity using the Greedy Modularity Communities
            communities = nx.algorithms.community.greedy_modularity_communities(simple_G)
            modularity = nx.algorithms.community.modularity(simple_G, communities)

            # Clustering coefficient measures the extent to which the neighbors of a given node (i.e., nodes directly
            # connected to it) are also connected to each other.
            clustering_coefficient = nx.clustering(simple_G)

            # Eigenvector centrality
            eigenvector_centrality = nx.eigenvector_centrality(simple_G, max_iter=1000)

            # Assortativity
            assortativity = nx.degree_assortativity_coefficient(simple_G)

            # Find connected components
            components = list(nx.connected_components(simple_G))
            num_components = len(components)
            component_sizes = sorted([len(c) for c in components], reverse=True)

            max_component_size = component_sizes[0] if component_sizes else 0
            second_max_component_size = component_sizes[1] if len(component_sizes) > 1 else 0

            print(f"[Year={year if year >= 1991 else f'1985-1990'}]: #Nodes={simple_G.number_of_nodes()}, #Edges"
                  f"={simple_G.number_of_edges()}, "
                  f"#CC={num_components} (size of max component: {max_component_size}, size of 2nd max component:"
                  f" {second_max_component_size})")

            stats_d.append({
                "Number of Nodes": simple_G.number_of_nodes(),
                "Number of Edges": simple_G.number_of_edges(),
                "Number of Components": num_components,
                "Max Component Size": max_component_size,
                "Second Max Component Size": second_max_component_size,
                "Year": year,
            })

            """
            # k-core analysis. Note that this requires a graph without loop
            k = 3
            core_G = nx.k_core(simple_G, k)
            """

            # Calculate average degree
            degrees = dict(multi_G.degree())
            average_degree = sum(degrees.values()) / len(degrees)

            print(f"Average degree: {average_degree}")
            print("Degree of each vertex (keyword), considering multiple edges:")
            print(degrees)

            # Perform Louvain community detection
            partition = community.best_partition(G)

            year += 1


    elif args.graph_backend == "rapids":
        import cugraph

        stats_d = []

        # Construct snapshot for each year
        for year in range(1985, 2025):
            G = cugraph.MultiGraph()

            edges = edge_df.query(f"published_year == {year}")

            if len(edges) == 0:
                continue
            # G.from_cudf_edgelist(edge_df, source='source', destination='destination', edge_attr='index_paper', renumber=True)
            G.from_cudf_edgelist(edges, source='source', destination='destination', edge_attr='index_paper',
                                 renumber=True)

            # Get node mappings
            nodes_mapping = G.nodes().to_pandas().to_list()
            node_index_to_id = {node_id: node_index for node_index, node_id in enumerate(nodes_mapping)}
            custom_map = G.nodes().to_frame(name="keyword")
            # Add an 'id' column
            custom_map['node_id'] = cudf.RangeIndex(start=0, stop=len(custom_map))

            # Extract edge list from the existing graph
            edge_list = G.view_edge_list()
            # Merge custom mapping to apply it
            edge_list = edge_list.merge(custom_map, left_on='source', right_on='keyword', how='left') \
                .drop(columns=['source', 'keyword']) \
                .rename(columns={'node_id': 'source'})
            edge_list = edge_list.merge(custom_map, left_on='destination', right_on='keyword', how='left') \
                .drop(columns=['destination', 'keyword']) \
                .rename(columns={'node_id': 'destination'})

            edge_list = edge_list.astype({'source': 'int32', 'destination': 'int32', 'index_paper': 'int32'})

            # Recreate the graph as a multigraph
            multi_G = cugraph.MultiGraph()
            multi_G.from_cudf_edgelist(edge_list, source='source', destination='destination', renumber=False)

            # Create a simple graph with 'weight' attribute as the number of edges between two nodes
            simple_G = cugraph.Graph()
            edge_list['weight'] = 1

            simple_edge_list = edge_list[['source', 'destination', 'weight']].groupby(['source', 'destination'],
                                                                                      as_index=False).sum()
            simple_G.from_cudf_edgelist(simple_edge_list, source='source', destination='destination',
                                        edge_attr='weight', renumber=False)

            components_df = cugraph.connected_components(simple_G)

            num_components = components_df['labels'].nunique()

            grouped = components_df.groupby('labels')['vertex'].agg(list)

            component_sizes = grouped.to_pandas().apply(len).sort_values(ascending=False)

            max_component_size = component_sizes.max()

            print(
                f"[Year={year}]: #Nodes={simple_G.number_of_nodes()}, #Edges={simple_G.number_of_edges()}, #CC={num_components} (Max: {component_sizes.iloc[0]}, 2nd Max: {component_sizes.iloc[1] if len(component_sizes) > 1 else 0}")

            stats_d += [{
                "Number of Nodes": simple_G.number_of_nodes(),
                "Number of Edges": simple_G.number_of_edges(),
                "Number of Components": num_components,
                "Max Component Size": component_sizes.iloc[0],
                "Second Max Component Size": component_sizes.iloc[1] if len(component_sizes) > 1 else 0,
                "Year": year,

            }]

            continue

            k = 3
            core_df = cugraph.k_core(G, k)

            # Calculate average degree
            degree_df = G.degrees()
            average_degree = degree_df['degree']

            component_sizes = labels.value_counts().sort_values(ascending=False)
            largest_component_size = component_sizes.iloc[0]

            # Degree calculation considering multiple edges
            degrees = G.degrees()
            print("Degree of each vertex (keyword), considering multiple edges:")
            print(degrees)

    stats_df = pd.DataFrame(stats_d).set_index("Year")

    stats_df.to_excel(osp.join(args.output_dir, "stats_nodes_in_connected_components.xlsx"))

    # Use the Louvain algorithm to find communities
    df_louvain, modularity_score = cugraph.louvain(G)
    print("Modularity of partitioning: ", modularity_score)

    # Detect communities
    from networkx.algorithms.community import greedy_modularity_communities

    simple_G = create_simple_graph_from_multigraph(G)

    communities = list(greedy_modularity_communities(simple_G, weight='weight'))

    # Print communities
    for i, community in enumerate(communities):
        print(f"Community {i}: {sorted(community)}")

    top_nodes_per_community = {}
    for i, community in enumerate(communities):
        subgraph = simple_G.subgraph(community)
        degrees = sorted(subgraph.degree, key=lambda x: x[1], reverse=True)[:5]
        top_nodes_per_community[i] = degrees

    # Print top nodes per community
    for community, nodes in top_nodes_per_community.items():
        print(f"Top nodes in community {community}: {nodes}")

    # Gather all top nodes
    top_nodes = [node for nodes in top_nodes_per_community.values() for node, degree in nodes]

    # Create subgraph
    subG = simple_G.subgraph(top_nodes)

    # Export to GEXF for Gephi
    nx.write_gexf(subG, "top_nodes_subgraph.gexf")

    # This is still too large
    nx.write_gexf(G, "multigraph_2024.gexf")

    import graphistry

    # graphistry.register(api=3, protocol="https", server="hub.graphistry.com", personal_key_id=args.graphistry_personal_key_id, personal_key_secret=args.graphistry_personal_key_secret)

    graphistry.register(api=3, personal_key_id=args.graphistry_personal_key_id,
                        personal_key_secret=args.graphistry_personal_key_secret)

    snapshot = edge_df.query("published_year == 2024").reset_index(drop=True).to_pandas()
    plot = graphistry.bind(source='source_column', destination='destination_column').plot(snapshot)

    graphistry.register(protocol="http", server="localhost", personal_key_id=args.graphistry_personal_key_id,
                        personal_key_secret=args.graphistry_personal_key_secret)

    plotter = graphistry.bind(source='source', destination='destination', edge_title='index_paper')
    plotter.plot(edge_df)

    print("Done!")

    # G.plot()
