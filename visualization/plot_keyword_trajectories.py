import datetime
import os.path as osp
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import seaborn as sns
from adjustText import adjust_text
from dateutil.relativedelta import relativedelta
from gensim.models import Word2Vec
from openTSNE import TSNE

import const

from arguments import parse_args
from embed import alignment

from embed.embeddings import Embedding
from model.procrustes import procrustes_align
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

    highlighted_words_embed_dict = defaultdict(dict)

    # We plot the trajectories of these words over time
    highlighted_words = ["machine learning", "optimization", "large language models", "llm"]
    nearest_neighbors = set()

    iterator = TimeIterator(1985, 2025, start_month=1, end_month=1, snapshot_type='yearly')

    first_iteration = True

    words_current_year = set()
    words_previous_year = set()

    embed_previous_year = None

    # Align all embeddings to this timestamp
    base_embed_start_timestamp = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.utc)

    base_embed_filename = f"word2vec_{base_embed_start_timestamp.strftime(const.format_string)}-" \
                          f"{(base_embed_start_timestamp + relativedelta(years=1)).strftime(const.format_string)}.model"

    base_embed = Word2Vec.load(osp.join(args.checkpoint_dir, args.feature_name, const.WORD2VEC, base_embed_filename))

    base_embed = Embedding(base_embed.wv.vectors, base_embed.wv.index_to_key, normalize=True)

    valid_words_mask_base = base_embed.m.sum(axis=1) != 0

    words_base_embeds = set(np.array(base_embed.iw)[valid_words_mask_base])

    for (start, end) in iterator:

        model_path = osp.join(args.checkpoint_dir, args.feature_name, const.WORD2VEC, f"word2vec"
                                                                                      f"_{start.strftime(const.format_string)}-{end.strftime(const.format_string)}.model")

        get_file_times(model_path)


        print(f"Loading model from {model_path} ...", end='\r')

        model = Word2Vec.load(model_path)
        embed = Embedding(model.wv.vectors, model.wv.index_to_key, normalize=True)

        valid_words_mask_cur_year = embed.m.sum(axis=1) != 0

        words_current_year = set(np.array(embed.iw)[valid_words_mask_cur_year])

        if start.year == BASE_YEAR:
            # base_embedding is the year we want to plot the scatterplot
            aligned_embed = embed


        else:
            # aligned_embed = alignment.smart_procrustes_align(aligned_embed, embed)

            aligned_embed = procrustes_align(base_embed, embed, words_current_year & words_base_embeds)

        highlighted_words_embed = aligned_embed.get_subembed(highlighted_words)



        for word1 in highlighted_words:
            """
            closest_words_and_similarity:
            [
                (similarity1, word1), # The 1st entry corresponds to word1 itself
                (similarity2, word2), 
                ...
            ]
            """

            closest_words_and_similarity = embed.closest(word1, n=len(embed.iw))

            closest_words_ranking = {word2: i for i, (_, word2) in enumerate(closest_words_and_similarity)}

            for word2 in highlighted_words:
                if word2 not in closest_words_ranking:
                    continue
                print(f"[{start.strftime(const.format_string)}] {word1} -> {word2}: {closest_words_ranking[word2]}\t"
                      f"{closest_words_ranking[word2] / len(embed.iw) * 100 :.2f}%")


            # Only consider the top 5 closest words
            nearest_neighbors.update([word for i, (_, word) in enumerate(closest_words_and_similarity) if i < 5])

            if word1 in highlighted_words_embed.wi:
                highlighted_words_embed_dict[word1][start.year] = highlighted_words_embed.m[
                    highlighted_words_embed.wi[word1]].astype(np.float32)

            else:
                highlighted_words_embed_dict[word1][start.year] = np.zeros(args.embed_dim, dtype=np.float32)

        embed_previous_year = embed
        words_previous_year = words_current_year


    visualization_model = TSNE(initialization="pca", n_components=2, perplexity=30, metric="cosine", n_iter=300,
                               verbose=True)

    embedding_train = visualization_model.fit(base_embed.m[valid_words_mask_base])

    # Convert embedding_train and words list into a DataFrame
    df = pd.DataFrame(embedding_train, columns=['x', 'y'])
    df['word'] = np.array(base_embed.iw)[valid_words_mask_base]

    fig, ax = plt.subplots(figsize=(10, 10))

    sns.scatterplot(data=df, x='x', y='y', s=1, ax=ax, edgecolor="none", alpha=0.25, color="lightgray")  # s is the
    # marker size

    texts = []

    for idx_row, row in df.iterrows():
        if row.word in nearest_neighbors and not row.word in highlighted_words_embed_dict and not any([punc in
                                                                                                       row.word for
                                                                                                       punc in ",:_"]):
            texts += [ax.text(row.x, row.y, row.word, fontsize=5)]

    for word in highlighted_words_embed_dict:
        years = list(highlighted_words_embed_dict[word].keys())
        embeds = [highlighted_words_embed_dict[word][year] for year in years]
        embeds = np.array(embeds)
        embeds = embedding_train.transform(embeds)

        for idx_year, year in enumerate(years):
            if year in [1995, 2000, 2010, 2022, 2023]:
                texts += [ax.text(embeds[idx_year, 0], embeds[idx_year, 1], f"{word}_{year}", fontsize=7, color='blue')]

    print("Adjusting text...", end="")

    adjust_text(texts, autoalign='xy', expand_points=(1.2, 1.2), expand_text=(1.2, 1.2))

    ax.set_axis_off()
    plt.show()

    # plt.savefig("keyword_trajectories.pdf", dpi=600, bbox_inches='tight')
    print("Done!")
