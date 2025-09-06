"""
Count the number of papers and keywords each year in the dataset
"""

import os
import os.path as osp
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker
# import scienceplots

sys.path.append(os.path.abspath('.'))

from visualization import plot_config
from utility.utils_data import load_keywords, load_arXiv_data

from arguments import parse_args
from utility.utils_misc import project_setup

# plt.rcParams['text.usetex'] = False
# plt.style.use('science')

if __name__ == "__main__":
    project_setup()
    args = parse_args()

    path = os.path.join(args.output_dir, "stats", "num_papers_and_keywords.xlsx")

    YMAX = 100
    INTERVAL = 20

    if os.path.exists(path):
        num_papers_keywords_per_year = pd.read_excel(path, index_col=0)

    else:

        arxiv_data = load_arXiv_data(args.data_dir).set_index("id")
        keywords_df = load_keywords(args.data_dir, args.feature_name)
        path_graph = osp.join(args.output_dir, f'{args.feature_name}_edges.parquet')

        edge_df = pd.read_parquet(path_graph)

        year = 1985

        num_papers_per_year = {}
        num_keywords_per_year = {}

        keywords_df['published'] = keywords_df['published'].dt.tz_convert(None)
        keywords_df['year'] = keywords_df['published'].dt.year

        while year < 2025:
            G = nx.MultiGraph()

            if year <= 1990:
                edges = edge_df.query("1985 <= published_year <= 1990")
                year = 1990

                keywords = keywords_df.query("1985 <= year <= 1990")["keywords"]

            else:
                edges = edge_df.query(f"published_year == {year}")
                keywords = keywords_df.query(f"year == {year}")["keywords"]

            if len(edges) == 0:
                continue

            print(f"[Year={year}] calculating #papers and keywords")

            num_papers = len(edges)
            num_papers_per_year[year] = num_papers

            keywords_li = sum(keywords.tolist(), [])
            num_keywords_per_year[year] = len(set(keywords_li))
            year += 1

        num_keywords_per_year = pd.Series(num_keywords_per_year)
        num_papers_per_year = pd.Series(num_papers_per_year)

        num_papers_keywords_per_year = pd.concat([num_keywords_per_year, num_papers_per_year], axis=1)
        num_papers_keywords_per_year.columns = ["Number of Keywords", "Number of Papers"]

        num_papers_keywords_per_year.to_excel(path, index=True)

    # Your existing data and setup
    fig, ax1 = plt.subplots(figsize=(7, 6), dpi=300)

    # Plot the first column with respect to the primary y-axis (left side)
    line1, = ax1.plot(num_papers_keywords_per_year.iloc[:, 0].values, marker='o', color='#8c51ff',
                      markersize=5, linewidth=2, label='Number of Keywords')

    # Create a secondary y-axis (right side) for the second column
    ax2 = ax1.twinx()
    line2, = ax2.plot(num_papers_keywords_per_year.iloc[:, 1].values, marker='s', color='#11b5f5',
                      markersize=5, linewidth=2, label='Number of Papers')

    # Set major and minor ticks for the x-axis
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))  # Big ticks
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))  # Small ticks

    # Define the range and interval for the x-axis ticks
    x_ticks = [1990] + np.arange(1995, 2030, 5).tolist()
    x_ticks_labels = [str(year) for year in x_ticks]
    x_ticks = np.array(x_ticks) - 1990
    x_ticks_labels[0] = 'before\n1990'  # Rename the first tick label

    plt.xticks(x_ticks, x_ticks_labels, fontsize=plot_config.FONT_SIZE)  # Set custom ticks and labels

    # Set y-axis intervals for the first column (major axis, left side)
    ax1.set_ylim(0, num_papers_keywords_per_year.iloc[:, 0].max() + 100000)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(100000))
    ax1.set_ylabel('Number of Keywords', fontsize=plot_config.FONT_SIZE)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.0e}'))  # Scientific notation

    # Set y-axis intervals for the second column (minor axis, right side)
    ax2.set_ylim(0, num_papers_keywords_per_year.iloc[:, 1].max() + 200000)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(200000))
    ax2.set_ylabel('Number of Papers', fontsize=plot_config.FONT_SIZE)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.0e}'))  # Scientific notation

    # Combine legends from both axes and display it only once
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=plot_config.FONT_SIZE)


    # Add title and labels
    plt.title(f'Number of Keywords and Papers (1985-2024)', fontsize=plot_config.FONT_SIZE)
    plt.xlabel('Year', fontsize=plot_config.FONT_SIZE)
    plt.tight_layout()

    # Save and display the plot
    os.makedirs(os.path.join(args.output_dir, "visual"), exist_ok=True)
    plt.savefig(os.path.join(args.output_dir, "visual", 'num_papers_keywords_per_year.pdf'), dpi=300)

    plt.show()

