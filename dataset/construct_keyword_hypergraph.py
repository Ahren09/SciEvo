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

from utility.graph_utils import create_simple_graph_from_multigraph

sys.path.append(os.path.abspath('.'))
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
    
    path_graph = osp.join(args.output_dir, f'{args.attribute}_edges.parquet')
    
    
    
    
    
    if not osp.exists(path_graph):
        path = os.path.join(args.data_dir, "NLP", "arXiv", f"{args.attribute}_keywords.json")
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
            edge_df = cudf.read_parquet(path_graph)
        
    
    print(f"Building the graph ...", end=" ")
    t0 = time()
    
    # Construct snapshot for each year
    
    # Creating a Graph using cuGraph

    if args.graph_backend == "networkx":
        G = nx.MultiGraph()
        G.add_edges_from(edge_df.query("published_year == 2024")[["source", "destination"]].values)

    elif args.graph_backend == "rapids":
        import cugraph
        G = cugraph.MultiGraph()

        # G.from_cudf_edgelist(edge_df, source='source', destination='destination', edge_attr='index_paper', renumber=True)
        G.from_cudf_edgelist(edge_df.query("published_year == 2024"), source='source', destination='destination', edge_attr='index_paper', renumber=True)
        print(f"Building the graph: {time() - t0:.2f} seconds")

        # Degree calculation considering multiple edges
        degrees = G.degrees()
        print("Degree of each vertex (keyword), considering multiple edges:")
        print(degrees)

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