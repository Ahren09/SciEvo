import collections
import datetime
import os
import sys
import os.path as osp
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import seaborn as sns
import torch
from dateutil.relativedelta import relativedelta
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

from utility.utils_time import TimeIterator

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import const
from analysis import alignment
from utility.utils_data import write_pickle

from openTSNE import TSNE

from arguments import parse_args
from embed.embeddings import Embedding
from utility.utils_misc import project_setup

plt.ion()
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
    # fixedvecs = othervecs.dot(ortho)

    # Apply transformation to all words in the embeddings to be aligned
    transformed_othervecs = other_embed.m - np.mean(other_embed.m[other_indices], axis=0)
    transformed_othervecs = transformed_othervecs.dot(ortho)

    # Re-introduce the mean of the base vectors (seems not needed )
    # transformed_othervecs += np.mean(basevecs, axis=0)
    return Embedding(transformed_othervecs, other_embed.iw)

if __name__ == "__main__":

    # Load embeddings
    project_setup()
    args = parse_args()

    # Load shared vocabulary indices
    # with open(f"../arXivData/checkpoints/{args.feature_name}/word2vec/shared_vocab.pkl", 'rb') as f:
    #     d = pickle.load(f)
    #
    # shared_wi = d["wi"]
    # shared_iw = d["iw"]

    embeddings = collections.OrderedDict()

    # Store the nearest words to word1 in each year
    nearest_words_set = set()

    base_embed = None

    model_path = osp.join("checkpoints", f"{args.feature_name}_{args.tokenization_mode}", args.model_name)

    iterator = TimeIterator(2021, 2025, start_month=1, end_month=1, snapshot_type='yearly')

    common_words = None

    # Align all embeddings to this timestamp
    base_embed_start_timestamp = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.utc)

    base_year = 1996
    if args.model_name == const.GCN:
        filename = f"{const.GCN}_embeds_{base_year}.pkl"
        
        with open(osp.join(model_path, filename), "rb") as f:
            d = pickle.load(f)
        
        
        base_embed = d["embed"]
        
        node_mapping = d["node_mapping"]

        base_embed = Embedding(base_embed.wv.vectors, base_embed.wv.index_to_key, normalize=False)
    
    elif args.model_name == const.WORD2VEC:
        
        base_embed_filename = f"word2vec_{base_embed_start_timestamp.strftime(const.format_string)}-" \
                  f"{(base_embed_start_timestamp + relativedelta(years=1)).strftime(const.format_string)}.model"


        base_embed = Word2Vec.load(osp.join(model_path, base_embed_filename))

        base_embed = Embedding(base_embed.wv.vectors, base_embed.wv.index_to_key, normalize=True)


    for i, (start, end) in enumerate(iterator):

        # Treat all papers before 1995 as one single snapshot

        filename = f"word2vec_{start.strftime(const.format_string)}-{end.strftime(const.format_string)}.model"

        print(f"Loading model from {filename} ...", end='\r')

        model = Word2Vec.load(osp.join(model_path, filename))

        year_embed = Embedding(model.wv.vectors, model.wv.index_to_key, normalize=True)
        # Set of words to visualize
        if common_words is None:
            common_words = set(year_embed.iw)
        else:
            common_words = common_words & set(year_embed.iw)

        embeddings[start.year] = year_embed
        # TODO
        # nearest_words_set.update([tup[1] for tup in embed.closest("large", n=10)])

        print("Aligning year:", start.year)
        if start == base_embed_start_timestamp:
            aligned_embed = year_embed


        else:
            aligned_embed = alignment.smart_procrustes_align(base_embed, year_embed)


        print("Writing year:", start.year)
        foutname = osp.join(args.output_dir, f"{start.strftime(const.format_string)}-{end.strftime(const.format_string)}")
        np.save(foutname + "-w.npy", aligned_embed.m)
        write_pickle(aligned_embed.iw, foutname + "-vocab.pkl")

    # Use `BASE_YEAR` as the base year for alignment
    base_embedding = embeddings[base_embed_start_timestamp.year]

    # This stores the aligned embeddings for each year
    # {year -> Embedding}
    aligned_embeddings = {base_embed_start_timestamp: base_embedding}
    valid_words_mask_base = base_embedding.m.sum(axis=1) != 0

    # common_words = np.array(base_embedding.iw)

    for year, embedding in embeddings.items():
        if year != base_embed_start_timestamp.year:
            valid_words_mask_embed = embedding.m.sum(axis=1) != 0
            embedding.get_subembed(common_words)
            aligned_embeddings[year] = procrustes_align(base_embedding, embedding, common_words)

    # Assuming 'aligned_embeddings' is a dictionary of Embedding objects from 1995 to 2025, aligned to 2023
    # Assuming 'aligned_embeddings' is a dictionary of Embedding objects from 1995 to 2025, aligned to 2023

    word1_trajectory = []

    # Example of accessing trajectory of the word 'large' across years
    word = "deep learning"


    years = []
    for year, aligned_embedding in aligned_embeddings.items():
        if aligned_embedding.wi.get(word) is not None:
            word_index = aligned_embedding.wi[word]
            word1_trajectory += [aligned_embedding.m[word_index]]
            years += [year]

        else:
            print(f"Word '{word}' not found in year {year}.")


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

        if point['word'] in nearest_words_set:
            ax.text(point['x'],  # This offsets the text slightly to the right of the marker
                     point['y'],
                     point['word'],
                     horizontalalignment='left',
                     fontsize=5,
                     color='black',
                     weight='semibold',
                    )

    embedding_test = embedding_train.transform(np.array(word1_trajectory))



    word1_trajectory = np.array(word1_trajectory)

    # Removing years with NaN values if any exist
    nan_indices = np.any(np.isnan(word1_trajectory), axis=1)
    if np.any(nan_indices):
        print(f"Warning: Missing '{word}' in some years, skipping these years in the trajectory.")
        word1_trajectory = word1_trajectory[~nan_indices]

    if len(word1_trajectory) > 0:
        plt.plot(embedding_test[:, 0], embedding_test[:, 1], 'r--', label='Trajectory of word1')
        plt.scatter(embedding_test[:, 0], embedding_test[:, 1], c='red', ax=ax)
        plt.title("Trajectory of 'word1' Over Years")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("No valid data to plot.")

