import os
import pickle
import sys
from collections import Counter

import community as community_louvain  # pip install python-louvain
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from utility.utils_misc import project_setup

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import const
from arguments import parse_args
from utility.utils_data import load_arXiv_data

MODE = "3D"

WRITE_GEXF = True
VISUALIZE = False
COMMUNITY_DETECTION = False

args = parse_args()

project_setup()

citation_graph_path = os.path.join(args.output_dir, "citation_analysis", "citation_graph.pkl")

path_mask = os.path.join(args.data_dir, "NLP", "arXiv", "topic_mask.parquet")

mask_df = pd.read_parquet(path_mask)


with open(citation_graph_path, 'rb') as f:
    G = pickle.load(f)

data = []

for subject, topics in const.SUBJECT2KEYWORDS.items():
    for topic, keywords in tqdm(topics.items(), desc=subject):

        path_gexf = os.path.join(args.output_dir, "visual", f"citation_graph_{topic}.gexf")
        if WRITE_GEXF and os.path.exists(path_gexf):
            continue

        print(f"Calculating metrics for keyword={topic}")
        relevant_arxiv_ids = set(mask_df[mask_df[topic]]['arXivId'].tolist())

        nodes_of_topic = list(set(G.nodes()) & relevant_arxiv_ids)

        subgraph = G.subgraph(nodes_of_topic)

        print(f"Subgraph created with {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges.")

        # Sort the weakly connected components by size in descending order
        wccs = sorted(nx.weakly_connected_components(subgraph), key=lambda x: len(x), reverse=True)

        # Extract the largest WCC belonging to the topic
        largest_wcc = subgraph.subgraph(wccs[0])

        # Calculate number of nodes in the subgraph and largest WCC
        num_nodes = subgraph.number_of_nodes()
        num_nodes_in_largest_wcc = len(largest_wcc)
        num_wccs = len(wccs)

        data.append({
            'topic': topic,
            'num_nodes': num_nodes,
            'num_nodes_in_largest_wcc': num_nodes_in_largest_wcc,
            'num_wccs': num_wccs
        })

        # Write to gexf for visualization in Gephi


        if WRITE_GEXF:
            nx.write_gexf(largest_wcc, path_gexf)
            continue

        if COMMUNITY_DETECTION:
            largest_wcc = nx.to_undirected(largest_wcc)
            partition = community_louvain.best_partition(largest_wcc, resolution=2.0, random_state=args.random_seed)
            count_communities = Counter(partition.values())
            print(f"Number of communities: {len(count_communities)}")

        if VISUALIZE:

            # Visualize the largest connected component with colored communities
            # Get the list of unique communities
            communities = set(partition.values())
            fig = plt.figure(figsize=(20, 20), dpi=300)

            # Color map for distinct communities (e.g., use matplotlib colormap)
            cmap = plt.get_cmap('Spectral', len(communities))

            if MODE == "2D":

                pos = nx.spring_layout(largest_wcc, dim=2)

                # Draw nodes with colors based on community
                for node in largest_wcc.nodes():
                    community_id = partition[node]
                    nx.draw_networkx_nodes(largest_wcc, pos, nodelist=[node], node_color=[cmap(community_id)],
                                           node_size=3)

                # Draw edges
                nx.draw_networkx_edges(largest_wcc, pos, width=0.1, arrowsize=1)

                # Show the plot
                plt.title('Largest Weakly Connected Component with Community Detection')
                plt.show()

            elif MODE == "3D":

                X = np.random.random((len(communities), 3)) * 5e-1

                X = np.concatenate((X, np.ones((X.shape[0], 1)) * 1.), axis=1)

                pos = nx.spring_layout(largest_wcc, dim=3)

                # Step 4: Create a 3D plot
                ax = fig.add_subplot(111, projection='3d')

                # Draw nodes with colors based on community
                for node in largest_wcc.nodes():
                    community_id = partition[node]
                    x, y, z = pos[node]  # Extract the 3D coordinates from pos
                    ax.scatter(x, y, z,
                               c=[X[community_id]],
                               # c=[cmap(community_id)],
                               s=5)  # s controls node size

                # Draw edges
                for edge in largest_wcc.edges():
                    x_coords = [pos[edge[0]][0], pos[edge[1]][0]]
                    y_coords = [pos[edge[0]][1], pos[edge[1]][1]]
                    z_coords = [pos[edge[0]][2], pos[edge[1]][2]]

                    ax.plot(x_coords, y_coords, z_coords, color='black', alpha=0.3, linewidth=0.05)

                # Set plot labels and title
                ax.set_title('Largest Weakly Connected Component in 3D with Community Detection')

                # Show the 3D plot
                plt.show()

                """
                # (Archived) Export graph for visualization in Blender
                graph_data = {
                    "nodes": [
                        {"id": node, "x": pos[node][0], "y": pos[node][1], "z": pos[node][2], "community": partition[node]}
                        for node in largest_wcc.nodes()],
                    "edges": [{"source": u, "target": v} for u, v in largest_wcc.edges()]
                }
    
                
                with open('graph_data.json', 'w') as f:
                    json.dump(graph_data, f)
                
                """

        df = pd.DataFrame(data)

        df.to_excel(os.path.join(args.output_dir, "stats", "citation_graph_metrics.xlsx"), index=False)

arxiv_data = load_arXiv_data(args.data_dir)

print("Done!")
