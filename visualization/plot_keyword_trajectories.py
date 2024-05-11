import datetime
import os.path as osp
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
from adjustText import adjust_text
from openTSNE import TSNE
import seaborn as sns



from arguments import parse_args
from embed.embeddings import Embedding
from utility.utils_misc import project_setup

plt.ion()

if __name__ == "__main__":
    BASE_YEAR = 2023
    format_string = "%Y-%m-%d"

    # Load embeddings
    project_setup()
    args = parse_args()

    base_embedding, valid_words_mask_base = None, None

    highlighted_words_embed_dict = defaultdict(dict)

    highlighted_words = []
    nearest_neighbors = set()

    for i, start_year in enumerate(range(1995, 2025)):

        start_month = 1

        # Treat all papers before 1990 as one single snapshot
        if start_year == 1994:
            start = datetime.datetime(1970, 1, 1, 0, 0, 0, tzinfo=pytz.utc)

            end = datetime.datetime(1995, 1, 1, 0, 0, 0, tzinfo=pytz.utc)


        else:

            end = datetime.datetime(start_year + 1, start_month, 1, 0, 0, 0, tzinfo=pytz.utc)

            start = datetime.datetime(start_year, start_month, 1, 0, 0, 0, tzinfo=pytz.utc)

        model_path = osp.join(args.output_dir, f"{start.strftime(format_string)}-{end.strftime(format_string)}")
        embed = Embedding.load(model_path)

        if start_year == 2023:
            base_embedding = embed
            valid_words_mask_base = embed.m.sum(axis=1) != 0

        highlighted_words_embed = embed.get_subembed(["large", "language", "llm"])

        for word in highlighted_words_embed.iw:
            closest_words_and_similarity = embed.closest(word, n=5)
            nearest_neighbors.update([word for _, word in closest_words_and_similarity])

            highlighted_words_embed_dict[word][start_year] = highlighted_words_embed.m[highlighted_words_embed.wi[word]]



    visualization_model = TSNE(initialization="pca", n_components=2, perplexity=30, metric="cosine", n_iter=300,
                               verbose=True)

    embedding_train = visualization_model.fit(base_embedding.m[valid_words_mask_base])

    # Convert embedding_train and words list into a DataFrame
    df = pd.DataFrame(embedding_train, columns=['x', 'y'])
    df['word'] = np.array(base_embedding.iw)[valid_words_mask_base]

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

    print("Done!")

    plt.savefig("keyword_trajectories.pdf", dpi=600, bbox_inches='tight')
    print("Done!")


