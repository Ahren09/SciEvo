"""
Run `get_keyword_trajectory_coords.py` first, then run this script to plot the trajectories

For the trajectories in the paper, see `notebooks/Plot_selected_trajectories.ipynb`
"""
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import os.path as osp


import json
import seaborn as sns
from adjustText import adjust_text
from matplotlib.patches import FancyArrowPatch
from sklearn.cluster import KMeans
from tqdm import tqdm

from arguments import parse_args
from utility.utils_data import load_keywords
from utility.utils_misc import project_setup

if __name__ == "__main__":

    NUM_CLUSTERS = 16
    project_setup()
    args = parse_args()


    path = os.path.join(args.output_dir, "keyword_trajectories.xlsx")
    trajectories_df = pd.read_excel(path, sheet_name="trajectories")

    FEATURE_TO_VISUALIZE = "trajectories"  # "background" or "trajectories"

    # This is taking all the words in the trajectories
    # This can make the plot very crowded
    trajectory_words = set(trajectories_df["word"].values)

    trajectory_words = ['machine learning', 'deep learning']


    background_df = pd.read_excel(path, sheet_name="background")
    nearest_neighbors = pd.read_excel(path, sheet_name="nearest_neighbors")

    path_graph = osp.join(args.output_dir, f'{args.feature_name}_edges.parquet')

    edges = pd.read_parquet(path_graph)


    # Fit the model
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=args.seed).fit(background_df[["x", "y"]].values)

    # Get the cluster labels
    labels = kmeans.labels_

    background_df['center'] = labels



    nearest_neighbors = set(nearest_neighbors["word"].values)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    # `s`: marker size
    sns.scatterplot(data=background_df, x='x', y='y', hue="center", s=20, ax=ax, edgecolor="none", alpha=.1,
                    palette="Paired"
     # color="lightgray"
    )

    # Plot the cluster centers
    centers = kmeans.cluster_centers_
    centers = pd.DataFrame(centers, columns=["x", "y"])
    sns.scatterplot(data=centers, x='x', y='y', c='red', marker='x', ax=ax)

    texts = []

    # Annotate the words in the background

    sampled_words = []

    for label in range(NUM_CLUSTERS):
        # Get the words in the cluster
        words = background_df[background_df["center"] == label]["word"].values
        sampled_words += random.sample(list(words), 2)

    sampled_words = set(sampled_words)

    if FEATURE_TO_VISUALIZE == "background":

        for idx_row, row in background_df.iterrows():
            if row.word in sampled_words and not row.word in trajectory_words and not any([punc in
                                                                                                           row.word for
                                                                                                           punc in ",:_"]):
                texts += [ax.text(row.x, row.y, row.word, fontsize=10)]

        adjust_text(texts, autoalign='xy', expand_points=(1.2, 1.2), expand_text=(1.2, 1.2))

        ax.get_legend().remove()
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()

    else:

        prev_coords = None

        colors = ["#cdb4db", "#ffc8dd", "#ffafcc", "#bde0fe", "#a2d2ff"]

        # Add the word trajectories
        for index_word, word in enumerate(trajectory_words):

            # Get the trajectory of one word
            trajectory = trajectories_df[trajectories_df["word"] == word].sort_values("year").reset_index()
            trajectory = trajectory[trajectory["visible"]].reset_index(drop=True)

            alpha = np.linspace(0, 1, len(trajectory))

            prev_x, prev_y = None, None
            for index_year, row in trajectory.iterrows():
                x, y = row['x'], row['y']

                # Do not add arrow for the first year
                if not index_year == 0:
                    # This arrow is OK
                    arrow = FancyArrowPatch(
                        (x, y), (prev_x, prev_y),
                        connectionstyle="arc3,rad=0.05",  # 曲线样式，rad 是曲率半径
                        arrowstyle='-|>',
                        color='gray',

                        alpha=alpha[index_year],
                        linewidth=3,
                        mutation_scale=20  # control the size of the arrow
                    )


                    ax.add_patch(arrow)

                    """
                    # This arrow looks dumb
                    ax.annotate(
                        '',
                        xy=(x, y),  # end
                        xytext=(prev_x, prev_y),  # start
                        arrowprops=dict(
                            arrowstyle='->',
                            color=colors[index_word % len(colors)],
                            alpha=0.3,
                            linewidth=5
                        )
                    )
                    """


                # texts += [ax.text(embeds[idx_year, 0], embeds[idx_year, 1], f"{word}_{year}", fontsize=7, color='blue')]

                prev_x, prev_y = x, y

            ax.text(x, y, f"{word}", fontsize=10, color='blue')

        print("Adjusting text...", end="")

        ax.get_legend().remove()
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()
        print("Done!")

