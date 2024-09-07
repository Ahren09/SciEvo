import os
import sys
from tqdm import tqdm, trange
import pandas as pd
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import networkx as nx
import community as community_louvain  # pip install python-louvain

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import const
from arguments import parse_args
from utility.utils_data import load_arXiv_data
from utility.utils_data import load_semantic_scholar_papers, load_semantic_scholar_references_parquet
from utility.utils_data import convert_arxiv_url_to_id

args = parse_args()

# Load Semantic Scholar Papers
semantic_scholar_papers = load_semantic_scholar_papers(args.data_dir)

df = semantic_scholar_papers[['arXivId', 'paperId']].dropna()
arXivID2SemanticScholarPaperID = dict(zip(df['arXivId'], df['paperId']))
SemanticScholarPaperID2arXivID = dict(zip(df['paperId'], df['arXivId']))

# Load arXiv data
arxiv_data = load_arXiv_data(args.data_dir)

def extract_references(data):
    """Extracts references and forms edges for the citation graph."""
    edges = []
    for index, row in tqdm(data.iterrows(), total=len(data), position=0, leave=True):
        source_arXivId = row['arXivId']
        if isinstance(row['references'], (list, np.ndarray)):
            for ref in row['references']:
                if 'citedPaper' in ref and ref['citedPaper']:
                    cited_paper_arXivId = SemanticScholarPaperID2arXivID.get(ref['citedPaper'].get('paperId'), None)
                    if cited_paper_arXivId:
                        edges.append((source_arXivId, cited_paper_arXivId))
    return edges

arxiv_data['arXivId'] = arxiv_data['id'].apply(convert_arxiv_url_to_id)

# citation_graph_edge_path = os.path.join(args.output_dir, "citation_analysis", "citation_graph_edgelist.parquet")
citation_graph_path = os.path.join(args.output_dir, "citation_analysis", "citation_graph.pkl")


if os.path.exists(citation_graph_path):
    print(f"Loading graph from {citation_graph_path}")

    with open(citation_graph_path, 'rb') as f:
        G = pickle.load(f)


else:
    # Create a Graph (use nx.Graph() for undirected, nx.DiGraph() for directed)
    G = nx.DiGraph()

    all_edges = []
    for year in [1990, 2005, 2011] + list(range(2016, 2025)):
        references_one_snapshot = load_semantic_scholar_references_parquet(args.data_dir, year)
        edges = extract_references(references_one_snapshot)
        all_edges += edges

    # Convert the list of edges into a pandas DataFrame
    edges_df = pd.DataFrame(all_edges, columns=['source', 'destination'])

    # Display the DataFrame
    print(edges_df.head())

    # Add edges to the NetworkX graph
    G.add_edges_from(edges_df.itertuples(index=False, name=None))

    for _, row in arxiv_data.iterrows():
        # Add node with arXivId as the node name, and title as the attribute
        G.add_node(row['arXivId'], title=row['title'])

    os.makedirs(os.path.dirname(citation_graph_path), exist_ok=True)

    with open(citation_graph_path, 'wb') as f:
        pickle.dump(G, f)

    print(f"Saved graph to {citation_graph_path}")



path_mask = os.path.join(args.data_dir, "NLP", "arXiv", "topic_mask.parquet")

mask_df = pd.read_parquet(path_mask)

topic2num_nodes = {}
topic2num_nodes_in_largest_wcc = {}

for subject, topics in const.SUBJECT2KEYWORDS.items():
    for topic, keywords in tqdm(topics.items(), desc=subject):

        print(f"Calculating metrics for keyword={topic}")
        relevant_arxiv_ids = set(mask_df[mask_df[topic]]['arXivId'].tolist())

        nodes_of_topic = list(set(G.nodes()) & relevant_arxiv_ids)

        subgraph = G.subgraph(nodes_of_topic)
        
        print(f"Subgraph created with {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges.")

        # Sort the weakly connected components by size in descending order
        wccs = sorted(nx.weakly_connected_components(subgraph), key=lambda x: len(x), reverse=True)


        # Extract the largest WCC belonging to the topic
        largest_wcc = subgraph.subgraph(wccs[0]) 

        largest_wcc = nx.to_undirected(largest_wcc)
        partition = community_louvain.best_partition(largest_wcc)

        topic2num_nodes[topic] = subgraph.number_of_nodes()
        topic2num_nodes_in_largest_wcc[topic] = len(wccs[0])


        # Step 3: Visualize the largest connected component with colored communities
        # Define the layout for the graph visualization
        pos = nx.spring_layout(largest_wcc)

        # Get the list of unique communities
        communities = set(partition.values())

        # Color map for distinct communities (e.g., use matplotlib colormap)
        cmap = plt.get_cmap('viridis', len(communities))

        # Draw nodes with colors based on community
        for node in largest_wcc.nodes():
            community_id = partition[node]
            nx.draw_networkx_nodes(largest_wcc, pos, nodelist=[node], node_color=[cmap(community_id)])

        # Draw edges
        nx.draw_networkx_edges(largest_wcc, pos)

        # Show the plot
        plt.title('Largest Weakly Connected Component with Community Detection')
        plt.show()










topic2num_nodes = pd.Series(topic2num_nodes)
topic2num_nodes_in_largest_wcc = pd.Series(topic2num_nodes_in_largest_wcc)

df = pd.concat([topic2num_nodes, topic2num_nodes_in_largest_wcc], axis=1)


arxiv_data = load_arXiv_data(args.data_dir)


import networkx as nx

# Convert cuGraph to NetworkX graph
NxG = G.to_networkx()

nx.write_graphml(NxG, './graph.graphml')


