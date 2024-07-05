import json
import os
import os.path as osp
import sys
from time import time

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from tqdm import tqdm

from collections import defaultdict

sys.path.append(os.path.abspath('.'))

from utility.graph_utils import create_simple_graph_from_multigraph
from utility.utils_data import load_arXiv_data
from arguments import parse_args
from utility.utils_misc import project_setup


def build_hypergraph(papers, backend="networkx"):

    assert backend in ["networkx", "rapids"]

    # Preparing edge data
    edge_list = []
    
    
    for index_paper, (arxiv_url, keywords) in enumerate(tqdm(papers.items())):
        keywords = keywords.split(", ")
        # Add edges between every pair of keywords to form a clique
        
        published = arxiv_data.loc[arxiv_url, "published"]
        for i in range(len(keywords)):
            for j in range(i + 1, len(keywords)):
                edge_list.append((keywords[i], keywords[j], index_paper, published.year, published.month))

    if backend == "networkx":
        df = pd.DataFrame(edge_list, columns=['source', 'destination', 'index_paper', 'published_year', 'published_month'])

    elif backend == "rapids":
        import cudf

        # Creating a cudf DataFrame
        df = cudf.DataFrame(edge_list, columns=['source', 'destination', 'index_paper', 'published_year', 'published_month'])

    else:
        raise ValueError(f"Invalid backend: {backend}")

    return df


if __name__ == "__main__":
    project_setup()
    args = parse_args()
    
    path_graph = osp.join(args.output_dir, f'{args.feature_name}_edges.parquet')
    
    
    
    
    
    if not osp.exists(path_graph):
        path = os.path.join(args.data_dir, "NLP", "arXiv", f"{args.feature_name}_keywords.json")
        arxiv_data = load_arXiv_data(args.data_dir).set_index("id")
        

        with open(path, "r") as f:
            papers = json.load(f)

        edge_df = build_hypergraph(papers, backend=args.graph_backend)
        print("Saving hypergraph edges to parquet file")
        
        
        edge_df.to_parquet(path_graph)

    else:
        print("Loading hypergraph edges from parquet file")
        if args.graph_backend == "networkx":
            edge_df = pd.read_parquet(path_graph)
        elif args.graph_backend == "rapids":
            import cudf
            edge_df = cudf.read_parquet(path_graph).astype({
                'source': 'str',
                'destination': 'str',
                'index_paper': 'int32',
                'published_year': 'int32',
                'published_month': 'int32' 
            })
        
    
    print(f"Building the graph ...", end=" ")
    t0 = time()
    

    if args.graph_backend == "networkx":
        G = nx.MultiGraph()
        G.add_edges_from(edge_df.query("published_year == 2024")[["source", "destination"]].values)

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
            G.from_cudf_edgelist(edges, source='source', destination='destination', edge_attr='index_paper', renumber=True)
            
            # Get node mappings
            nodes_mapping = G.nodes().to_pandas().to_list()
            node_index_to_id = {node_id: node_index for node_index, node_id in enumerate(nodes_mapping)}
            custom_map = G.nodes().to_frame(name="keyword")
            # Add an 'id' column
            custom_map['node_id'] = cudf.RangeIndex(start=0, stop=len(custom_map))
            
            # Extract edge list from the existing graph
            edge_list = G.view_edge_list()
            # Merge custom mapping to apply it
            edge_list = edge_list.merge(custom_map, left_on='source', right_on='keyword', how='left')\
                                .drop(columns=['source', 'keyword'])\
                                .rename(columns={'node_id': 'source'})
            edge_list = edge_list.merge(custom_map, left_on='destination', right_on='keyword', how='left')\
                                .drop(columns=['destination', 'keyword'])\
                                .rename(columns={'node_id': 'destination'})

            edge_list = edge_list.astype({'source': 'int32', 'destination': 'int32', 'index_paper': 'int32'})

            # Recreate the graph as a multigraph
            multi_G = cugraph.MultiGraph()
            multi_G.from_cudf_edgelist(edge_list, source='source', destination='destination', renumber=False)
            
            # Create a simple graph with 'weight' attribute as the number of edges between two nodes
            simple_G = cugraph.Graph()
            edge_list['weight'] = 1
            
            simple_edge_list = edge_list[['source', 'destination', 'weight']].groupby(['source', 'destination'], as_index=False).sum()
            simple_G.from_cudf_edgelist(simple_edge_list, source='source', destination='destination', edge_attr='weight', renumber=False)

     
            components_df = cugraph.connected_components(simple_G)
            
            num_components = components_df['labels'].nunique()
            
            grouped = components_df.groupby('labels')['vertex'].agg(list)
            
            component_sizes = grouped.to_pandas().apply(len).sort_values(ascending=False)
            
            max_component_size = component_sizes.max()
            
            print(f"[Year={year}]: #Nodes={simple_G.number_of_nodes()}, #Edges={simple_G.number_of_edges()}, #CC={num_components} (Max: {component_sizes.iloc[0]}, 2nd Max: {component_sizes.iloc[1] if len(component_sizes) > 1 else 0}")
            
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
    
    
    graphistry.register(api=3, personal_key_id=args.graphistry_personal_key_id, personal_key_secret=args.graphistry_personal_key_secret)
    
    snapshot = edge_df.query("published_year == 2024").reset_index(drop=True).to_pandas()
    plot = graphistry.bind(source='source_column', destination='destination_column').plot(snapshot)
    
    graphistry.register(protocol="http", server="localhost", personal_key_id=args.graphistry_personal_key_id, personal_key_secret=args.graphistry_personal_key_secret)

    
    plotter = graphistry.bind(source='source', destination='destination', edge_title='index_paper')
    plotter.plot(edge_df)
    
    print("Done!")
    
    # G.plot()