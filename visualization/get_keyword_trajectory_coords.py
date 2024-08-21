"""
Compute the trajectories of keywords through temporal analysis

First run `get_keyword_trajectory_coords.py` to generate the coordinates
Then run `plot_keyword_traj.py` to visualize the trajectories

"""

import datetime
import os
import os.path as osp
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
from dateutil.relativedelta import relativedelta
from gensim.models import Word2Vec
from openTSNE import TSNE

import const
from arguments import parse_args
from embed import alignment
from embed.embeddings import Embedding
from utility.utils_data import load_keywords
from utility.utils_misc import project_setup
from utility.utils_time import TimeIterator, get_file_times

plt.ion()

if __name__ == "__main__":
    BASE_YEAR = 2023

    # Load embeddings
    project_setup()
    args = parse_args()

    """
    highlighted_words_embed_dict:
    {
        word1: {
            1995: EMBED,
            1996: EMBED,
            ...
            2024: EMBED
        },
        word2: { ... }
    }
    """
    keywords = load_keywords(args.data_dir, args.feature_name)

    highlighted_words_embed_dict = defaultdict(dict)

    # We plot the trajectories of these words over time
    highlighted_words = ["artificial intelligence", "machine learning", "deep learning", "natural language "
                                                                                         "processing", "nlp",
                         "optimization",
                         "large language "
                         "models",
                         "llm", "language model", "attention", "neural networks", "transformer", "transformers", "bias",
                         "covid",
                         "covid-19", "sars-cov-2",
                         "quantum computing", "quantum", "gravitational waves", "decision making", "decision"]
    nearest_neighbors = set()

    iterator = TimeIterator(args.start_year, args.end_year, start_month=1, end_month=1, snapshot_type='yearly')

    first_iteration = True

    words_current_year = set()
    words_previous_year = set()

    embed_previous_year = None

    # Align all embeddings to this timestamp
    base_embed_start_timestamp = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.utc)

    keywords['year'] = keywords['published'].dt.year
    keyword_counts_base = keywords[keywords['year'] == BASE_YEAR].explode('keywords').groupby('keywords').size()

    if args.model_name == "word2vec":
        base_embed_filename = f"{args.model_name}_{base_embed_start_timestamp.strftime(const.format_string)}-" \
                              f"{(base_embed_start_timestamp + relativedelta(years=1)).strftime(const.format_string)}.model"

        base_embed_filename = f"{args.model_name}_{base_embed_start_timestamp.year}.model"


        base_embed = Word2Vec.load(
            osp.join(args.checkpoint_dir, f"{args.feature_name}_{args.tokenization_mode}", const.WORD2VEC, base_embed_filename))

        base_embed = Embedding(base_embed.wv.vectors, base_embed.wv.index_to_key, normalize=True)

    elif args.model_name == "gcn":

        path = osp.join(args.checkpoint_dir, f"{args.feature_name}_{args.tokenization_mode}", args.model_name,
                        f"gcn_embeds_{BASE_YEAR}.pkl")

        with open(path, "rb") as f:
            d = pickle.load(f)

        node_mapping = d['node_mapping']
        embed = d['embed']
        base_embed = Embedding(embed.numpy(), list(node_mapping.keys()), normalize=True)

    else:
        raise NotImplementedError

    # Filter out words that appear less than `min_occurrences` times
    base_embed = base_embed.get_subembed(
        keyword_counts_base[keyword_counts_base >= args.min_occurrences].index.tolist())
    print(f"#Words in base embedding: {len(base_embed.iw)}")

    valid_words_mask_base = base_embed.m.sum(axis=1) != 0

    words_base_embeds = set(np.array(base_embed.iw)[valid_words_mask_base])

    for (start, end) in iterator:

        if args.model_name == const.WORD2VEC:

            model_path = osp.join(args.checkpoint_dir, f"{args.feature_name}_{args.tokenization_mode}", args.model_name,
                                  f"{args.model_name}_{start.year}.model")

            get_file_times(model_path)

            model = Word2Vec.load(model_path)
            embed = Embedding(model.wv.vectors, model.wv.index_to_key, normalize=True)


        elif args.model_name == const.GCN:

            model_path = osp.join(args.checkpoint_dir, f"{args.feature_name}_{args.tokenization_mode}", args.model_name,
                                  f"{args.model_name}_{start.year}.pkl")

            with open(model_path, "rb") as f:
                d = pickle.load(f)

            node_mapping = d['node_mapping']
            embed = d['embed']

            embed = Embedding(embed.numpy(), list(node_mapping.keys()), normalize=True)

            get_file_times(model_path)

        else:
            raise NotImplementedError

        keyword_counts_snapshot = keywords[keywords['year'] == BASE_YEAR].explode('keywords').groupby('keywords').size()

        embed = embed.get_subembed(
            keyword_counts_snapshot[keyword_counts_snapshot >= args.min_occurrences].index.tolist())

        print(f"#Words (Year {start.year}): {len(embed.iw)}")

        print(f"Loading model from {model_path} ...", end='\r')

        valid_words_mask_cur_year = embed.m.sum(axis=1) != 0

        words_current_year = set(np.array(embed.iw)[valid_words_mask_cur_year])

        if start.year == BASE_YEAR:
            # base_embedding is the year we want to plot the scatterplot
            aligned_embed = embed


        else:
            aligned_embed = alignment.smart_procrustes_align(base_embed, embed)
            # aligned_embed = procrustes_align(base_embed, embed, words_current_year & words_base_embeds)

        highlighted_words_embed = aligned_embed.get_subembed(highlighted_words)

        for word1 in highlighted_words:
            """
            closest_words_and_similarity:
            [
                (similarity1, word1), # The 1st entry corresponds to word1 itself
                (similarity2, word2), # The 2nd entry is the most similar word to word1
                ...
            ]
            """
            if embed.wi.get(word1) is None:
                highlighted_words_embed_dict[word1][start.year] = None
                continue

            closest_words_and_similarity = embed.closest(word1, n=len(embed.iw))

            closest_words_ranking = {word2: i for i, (_, word2) in enumerate(closest_words_and_similarity)}
            # Print the ranking of the closest words to the highlighted words
            for word2 in highlighted_words:
                if word1 == word2 or word2 not in closest_words_ranking:
                    continue
                print(f"[{start.strftime(const.format_string)}] {word1} -> {word2}: {closest_words_ranking[word2]}\t"
                      f"{closest_words_ranking[word2] / len(embed.iw) * 100 :.2f}%")

            # Annotate some nodes in the background
            # We only consider the top 5 closest words in each year
            nearest_neighbors.update([word for i, (_, word) in enumerate(closest_words_and_similarity) if i < 5])

            if word1 in highlighted_words_embed.wi:
                highlighted_words_embed_dict[word1][start.year] = highlighted_words_embed.m[
                    highlighted_words_embed.wi[word1]].astype(np.float32)

            else:
                highlighted_words_embed_dict[word1][start.year] = None

        embed_previous_year = embed
        words_previous_year = words_current_year




    visualization_model = TSNE(initialization="pca", n_components=2, perplexity=30, metric="cosine", n_iter=300,
                               verbose=True)
    # valid_words_mask_base should be all `True`
    embedding_train = visualization_model.fit(base_embed.m[valid_words_mask_base])

    # Convert embedding_train and words list into a DataFrame
    background_df = pd.DataFrame(embedding_train, columns=['x', 'y'])
    background_df['word'] = np.array(base_embed.iw)[valid_words_mask_base]



    trajectories = []
    for word in highlighted_words_embed_dict:
        years = list(highlighted_words_embed_dict[word].keys())

        embeds, valid_embed_years = [], []
        for year in years:
            # Skip the years in which the keyword does not appear
            if highlighted_words_embed_dict[word][year] is None:
                continue
            embeds.append(highlighted_words_embed_dict[word][year])
            valid_embed_years.append(year)

        if len(embeds) == 0:
            continue

        # Transform embeddings into 2D coordinates
        embeds = np.array(embeds)
        embeds = embedding_train.transform(embeds)

        prev_coords = None
        for idx_year, year in enumerate(valid_embed_years):
            trajectories.append({
                "word": word,
                "year": year,
                "x": embeds[idx_year, 0],
                "y": embeds[idx_year, 1]
            })

    trajectories_df = pd.DataFrame(trajectories)

    # Create a Plotly scatter plot for better interactivity
    if args.do_plotly:
        import plotly.graph_objs as go
        import plotly.express as px

        # Only display nearest neighbors in the background
        background_mask = background_df.word.isin(nearest_neighbors)

        scatter = go.Scatter(
            x=background_df[background_mask]['x'].values,
            y=background_df[background_mask]['y'].values,
            mode='markers',
            marker=dict(size=10, color='blue'),
            name='Scatter',
            text=background_df['word'],  # Add the words as hover text
            hoverinfo='text'  # Display only the hover text
        )

        plotly_trajectories = []

        for word in highlighted_words:

            traj = trajectories_df[trajectories_df['word'] == "decision making"].reset_index(drop=True)

            line = go.Scatter(
                x=traj['x'],
                y=traj['y'],
                mode='lines',
                line=dict(color='green'),
                name=word,
                text=[f'{word} ({year})' for year in traj['year']],
                hoverinfo="text"
            )
            plotly_trajectories += [line]

        # Combine the scatter plot and line plots into a single figure
        fig = go.Figure(data=[scatter] + plotly_trajectories)

        # Update layout for better visibility
        fig.update_layout(
            title="Scatter Plot with Line Plots",
            xaxis_title="X Axis",
            yaxis_title="Y Axis"
        )

        fig.write_html("scatter_with_lines.html")



    with pd.ExcelWriter(osp.join(args.output_dir, f"keyword_trajectories.xlsx")) as writer:
        trajectories_df.to_excel(writer, sheet_name="trajectories", index=False)
        background_df.to_excel(writer, sheet_name="background", index=False)
        pd.Series(list(nearest_neighbors)).to_frame("word").to_excel(writer, sheet_name="nearest_neighbors",
                                                                     index=False)

    # plt.savefig("keyword_trajectories.pdf", dpi=600, bbox_inches='tight')
    print("Done!")
