import collections
import os.path as osp
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

plt.ion()
from openTSNE import TSNE

from arguments import parse_args
from embed.embeddings import Embedding
from utility.utils_misc import project_setup

# Define the Embedding class if not already defined


def procrustes_align(base_embed, other_embed, common_words):
    """
    Procrustes alignment using common words
    Args:
        base_embed:
        other_embed:
        common_words:

    Returns:

    """
    # Filter matrices to only include common words
    base_indices = [base_embed.wi[word] for word in common_words]
    other_indices = [other_embed.wi[word] for word in common_words]

    basevecs = base_embed.m[base_indices] - np.mean(base_embed.m[base_indices], axis=0)
    othervecs = other_embed.m[other_indices] - np.mean(other_embed.m[other_indices], axis=0)

    m = basevecs.T.dot(othervecs)
    u, _, vh = np.linalg.svd(m)
    ortho = u.dot(vh)
    fixedvecs = othervecs.dot(ortho)


    return Embedding(fixedvecs, [other_embed.iw[i] for i in other_indices])


if __name__ == "__main__":

    BASE_YEAR = 1996

    # Load embeddings
    project_setup()
    args = parse_args()

    # Load shared vocabulary indices
    with open(f"../arXivData/checkpoints/{args.feature_name}/word2vec/shared_vocab.pkl", 'rb') as f:
        d = pickle.load(f)

    shared_wi = d["wi"]
    shared_iw = d["iw"]

    embeddings = collections.OrderedDict()
    for i, year in enumerate(range(1995, 1998)):
        shape = (len(shared_wi), 300)  # Assuming 300 dimensions
        embed = np.memmap(osp.join(f"../arXivData/checkpoints/{args.feature_name}/word2vec/data_{year - 1995}.memmap"),
                          dtype=np.float32, mode='r+', shape=shape)
        embed = Embedding(embed, shared_iw, normalize=True)
        embeddings[year] = embed

    # Use `BASE_YEAR` as the base year for alignment
    base_embedding = embeddings[BASE_YEAR]
    aligned_embeddings = {BASE_YEAR: base_embedding}
    valid_words_mask_base = base_embedding.m.sum(axis=1) != 0

    for year, embedding in embeddings.items():
        if year != BASE_YEAR:


            valid_words_mask_embed = embedding.m.sum(axis=1) != 0

            common_words = np.array(shared_iw)[valid_words_mask_base & valid_words_mask_embed]


            aligned_embeddings[year] = procrustes_align(base_embedding, embedding, common_words)

    # Assuming 'aligned_embeddings' is a dictionary of Embedding objects from 1995 to 2025, aligned to 2023
    # Assuming 'aligned_embeddings' is a dictionary of Embedding objects from 1995 to 2025, aligned to 2023

    word1_trajectory = []

    # Example of accessing trajectory of the word 'large' across years
    word = "large"
    if word in shared_wi:
        word_index = shared_wi[word]
        word1_trajectory = [embed.m[word_index] for year, embed in aligned_embeddings.items()]
        print(word1_trajectory)

    visualization_model = TSNE(initialization="pca", n_components=2, perplexity=30, metric="cosine" ,n_iter=300,
                               verbose=True)

    embedding_train = visualization_model.fit(base_embedding.m[valid_words_mask_base])

    # Convert embedding_train and words list into a DataFrame
    df = pd.DataFrame(embedding_train, columns=['x', 'y'])
    df['word'] = np.array(base_embedding.iw)[valid_words_mask_base]

    fig, ax = plt.subplots(figsize=(10, 10))

    sns.scatterplot(data=df, x='x', y='y', s=2, ax=ax)  # s is the marker size

    # Annotate each point in the scatter plot with its word
    for i, point in df.iterrows():
        ax.text(point['x'],  # This offsets the text slightly to the right of the marker
                 point['y'],
                 point['word'],
                 horizontalalignment='left',
                 fontsize=5,
                 color='black',
                 weight='semibold',
                )



    word1_trajectory = np.array(word1_trajectory)

    # Removing years with NaN values if any exist
    nan_indices = np.any(np.isnan(word1_trajectory), axis=1)
    if np.any(nan_indices):
        print("Warning: Missing 'word1' in some years, skipping these years in the trajectory.")
        word1_trajectory = word1_trajectory[~nan_indices]

    # Now plotting
    plt.figure(figsize=(10, 5))
    if word1_trajectory.size > 0:
        plt.plot(word1_trajectory[:, 0], word1_trajectory[:, 1], 'r--', label='Trajectory of word1')
        plt.scatter(word1_trajectory[:, 0], word1_trajectory[:, 1], c='red')
        plt.title("Trajectory of 'word1' Over Years")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("No valid data to plot.")

