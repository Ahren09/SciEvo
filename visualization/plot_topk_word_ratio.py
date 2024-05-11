"""
Plot the change in percentile of word1 w.r.t. word2 over time.

"""

import datetime
import itertools
import os.path as osp
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import seaborn as sns
from adjustText import adjust_text
from matplotlib.ticker import FuncFormatter
from openTSNE import TSNE

import const
from arguments import parse_args
from embed.embeddings import Embedding
from utility.utils_misc import project_setup

plt.ion()

highlighted_word_tuples = [("large", "language"), ("large", "llm"), ("deep", "learning"), ("language", "bert"),
                           ("large", "bert"), ("bert", "large"), ("generative", "adversarial"), ("recommendation",
                                                                                                 "llm"), ("recommendation",
                                                                                                 "mf"), ("recommendation",
                                                                                                 "collaborative")]
word_rank = defaultdict(dict)
years = defaultdict(dict)
year2vocab_size = {}

for i, start_year in enumerate(range(1995, 2025)):
    start_month = 1

    project_setup()
    args = parse_args()

    # Treat all papers before 1990 as one single snapshot
    if start_year == 1994:
        start = datetime.datetime(1970, 1, 1, 0, 0, 0, tzinfo=pytz.utc)

        end = datetime.datetime(1995, 1, 1, 0, 0, 0, tzinfo=pytz.utc)


    else:

        end = datetime.datetime(start_year + 1, start_month, 1, 0, 0, 0, tzinfo=pytz.utc)

        start = datetime.datetime(start_year, start_month, 1, 0, 0, 0, tzinfo=pytz.utc)

    model_path = osp.join(args.output_dir, f"{start.strftime(const.format_string)}-{end.strftime(const.format_string)}")
    embed = Embedding.load(model_path)

    valid_words_in_embedding = set(np.array(embed.iw)[np.sum(embed.m, axis=1) != 0])

    year2vocab_size[start_year] = len(valid_words_in_embedding)
    print(f"Year: {start_year}, Vocab size: {len(valid_words_in_embedding)}")

    for word1, word2 in highlighted_word_tuples:
        if word1 not in valid_words_in_embedding or word2 not in valid_words_in_embedding:
            continue
        closest_words_and_similarity = embed.closest(word1, n=len(embed.iw))

        closest_words_ranking = {word2: i for i, (_, word2) in enumerate(closest_words_and_similarity)}

        if word1 != word2:
            print(f"[{start.strftime(const.format_string)}] {word1} -> {word2}: {closest_words_ranking[word2]}\t"
                  f"{closest_words_ranking[word2] / len(embed.iw) * 100 :.2f}%")
            word_rank[word1][word2] = word_rank[word1][word2] + [closest_words_ranking[word2]] if word2 in word_rank[word1] else [closest_words_ranking[word2]]
            years[word1][word2] = years[word1][word2] + [start_year] if word2 in years[word1] else [start_year]




data = []
for (word1, word2) in highlighted_word_tuples:
    if word1 not in word_rank or word2 not in word_rank[word1]:
        continue

    percentiles = [rank / year2vocab_size[year] * 100 for year, rank in zip(years[word1][word2], word_rank[word1][word2])]
    for year, percentile in zip(years[word1][word2], percentiles):
        data.append({'Year': year, 'Percentile': percentile, 'Word Pair': f'{word1}->{word2}'})

df = pd.DataFrame(data).astype({
    'Year': int,
    'Word Pair': str
})



# Plot using seaborn
sns.set_theme(style="white")
plt.figure(figsize=(10, 6))
ax = sns.lineplot(data=df, x='Year', y='Percentile', hue='Word Pair', palette='viridis', marker='o')
ax.set_ylim(100, 0)  # Invert y-axis

ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))

# ax.set_yticks(np.linspace(0, 100, 11))  # Set y-ticks to show 0 at top and 100 at bottom
# plt.gca().invert_yaxis()
plt.show()

print("Done")













