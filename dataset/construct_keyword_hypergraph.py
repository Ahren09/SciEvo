import json
import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from tqdm import tqdm
import cudf
import cugraph
from collections import defaultdict


sys.path.append(os.path.abspath('.'))
from arguments import parse_args
from utility.utils_misc import project_setup


def build_hypergraph(papers, backend="networkx"):

    assert backend in ["networkx", "rapids"]

    # Preparing edge data
    edge_list = []
    for paper_id, keywords in papers.items():
        keywords = keywords.split(", ")
        # Add edges between every pair of keywords to form a clique
        for i in range(len(keywords)):
            for j in range(i + 1, len(keywords)):
                edge_list.append((keywords[i], keywords[j], paper_id))


    if backend == "rapids":
        df = pd.DataFrame(edge_list, columns=['source', 'destination', 'paper_id'])

    elif backend == "rapids":
        # Creating a cudf DataFrame
        df = cudf.DataFrame(edge_list, columns=['source', 'destination', 'paper_id'])

    else:
        raise ValueError(f"Invalid backend: {backend}")

    return df


if __name__ == "__main__":
    project_setup()
    args = parse_args()

    path = os.path.join(args.data_dir, "NLP", "arXiv", f"{args.attribute}_keywords.json")

    with open(path, "r") as f:
        papers = json.load(f)

    edge_df = build_hypergraph(papers, backend=args.graph_backend)

    # Creating a Graph using cuGraph
    G = cugraph.MultiGraph()
    G.from_cudf_edgelist(edge_df, source='source', destination='destination', edge_attr='paper_id', renumber=True)

    # Degree calculation considering multiple edges
    degrees = G.degrees()
    print("Degree of each vertex (keyword), considering multiple edges:")
    print(degrees)

    # nx.write_gexf(G, "keyword_hypergraph.gexf")

    nx.draw(G, with_labels=True)
    plt.show()
