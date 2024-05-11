# Load embeddings
import datetime
import os.path as osp
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytz
import seaborn as sns
from gensim.models import Word2Vec

import matplotlib.ticker as ticker

import const
from arguments import parse_args
from utility.utils_misc import project_setup
from visualization import plot_config

project_setup()
args = parse_args()

word_occurrences = defaultdict(dict)

years = list(range(1995, 2024))

for i, start_year in enumerate(years):

    start_month = 1

    # Treat all papers before 1990 as one single snapshot
    if start_year == 1994:
        start = datetime.datetime(1970, 1, 1, 0, 0, 0, tzinfo=pytz.utc)

        end = datetime.datetime(1995, 1, 1, 0, 0, 0, tzinfo=pytz.utc)


    else:

        end = datetime.datetime(start_year + 1, start_month, 1, 0, 0, 0, tzinfo=pytz.utc)

        start = datetime.datetime(start_year, start_month, 1, 0, 0, 0, tzinfo=pytz.utc)

    embed_path = osp.join(args.output_dir, f"{start.strftime(const.format_string)}"
                                           f"-{end.strftime(const.format_string)}")

    model_path = osp.join(args.checkpoint_dir, args.feature_name, const.WORD2VEC, f"word2vec"
                                                                                  f"_{start.strftime(const.format_string)}-{end.strftime(const.format_string)}.model")

    print(f"Loading model from {model_path} ...", end='\r')

    model = Word2Vec.load(model_path)

    for word in plot_config.highlighted_words:
        if word not in model.wv.key_to_index:
            count = None

        else:
            count = model.wv.get_vecattr(word, 'count')

        word_occurrences[word][start_year] = count

df = pd.DataFrame(word_occurrences)

df['year'] = years

df['year'] = pd.to_datetime(df['year'], format='%Y')

# Convert the DataFrame to a 'long-form' or 'tidy' format.
df_long = df.melt('year', var_name='category', value_name='occurrences')

# Use Seaborn to create a line plot
fig, ax = plt.subplots(figsize=(18, 8))


sns.lineplot(data=df_long, x='year', y='occurrences', hue='category', marker='o', ax=ax)

ax.set_yscale('log')
ax.set_ylim(1, 1e4)

ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(base=1))
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))


# ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0,), numticks=4))

# For the y-axis: Set major ticks at each power of 10 within the range
ax.yaxis.set_major_locator(ticker.LogLocator(base=10))


# 尝试直接使用plt的接口设置年份间隔
# plt.xticks(list(range(1990, 2024, 4)))

# ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))


plt.title('#Papers under Each Topic Over Years', fontsize=plot_config.TITLE_FONT_SIZE)
plt.xlabel('Published Year', fontsize=plot_config.FONT_SIZE)
plt.ylabel('Number of Papers', fontsize=plot_config.FONT_SIZE)
legend = plt.legend(title='',
           # loc='upper center',
           # bbox_to_anchor=(0.5, 1.15),
           # nrow=2,
           fontsize=plot_config.FONT_SIZE,
           frameon=False,)

legend.get_frame().set_edgecolor('none')

# Set font sizes for the x-axis and y-axis labels
plt.xticks(fontsize=plot_config.FONT_SIZE)  # Set font size for x-axis scale
plt.yticks(fontsize=plot_config.FONT_SIZE)  # Set font size for y-axis scale

# For the x-axis: Set ticks every 4 years starting from 1990
ax.set_xticks([str(year) for year in range(1995, 2025, 5)])
ax.set_yticks([10, 100, 1000, 10000])


plt.show()

print("Done")
