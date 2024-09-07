"""
Visualize changes of the rank of popular keywords over time.

Run `analysis/rank_keywords_by_number_of_occurrences.py` first to generate the data.



"""
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
})

from arguments import parse_args
from utility.utils_misc import project_setup
from visualization import plot_config

sns.set(style="whitegrid")


def main():
    project_setup()
    args = parse_args()
    data = pd.read_excel(osp.join(args.output_dir, "stats", f"{args.feature_name}_keyword_ranks.xlsx"), index_col=0)

    YMAX = 200
    INTERVAL = 50

    for column in data.columns:
        data[column] = data[column].apply(lambda x: YMAX if x > YMAX else x)

    d = {
        "Machine Learning": ['Large Language Models',
                             'Reinforcement Learning',
                             'Transfer Learning',
                             'Zero-Shot Learning',
                             'Machine Learning',
                             'Deep Learning'],
        "Mathematics": ['Algebraic Geometry',
                        'Differential Geometry',
                        'Graph Theory',
                        'Group Theory',
                        'Knot Theory'],
        "Epidemiology": ["COVID-19"],
        "Cosmology": ["Dark Matter", "Gravitational Waves", "Star Formation", "Black Holes"],
        "Economics": ["Inflation", "Optimal Control", "Game Theory"],
        "Physics": ["Particle Physics", "Quantum Field Theory", "Thermodynamics", "Superconductivity", "General Relativity"],
    }

    for subject, keywords_of_interest in d.items():
        # if not subject in ["Machine Learning"]:
        #     continue

        # Create the line plot
        fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
        sns.lineplot(data=data.T[keywords_of_interest], marker='o', palette='deep', ax=ax)
        ax.grid(False)

        # Styling the plot
        ax.set_title(f'Ranks of Popular Keywords in {subject}', fontsize=plot_config.FONT_SIZE)
        ax.set_xlabel('Year', fontsize=plot_config.FONT_SIZE)
        ax.set_ylabel('Rank', fontsize=plot_config.FONT_SIZE)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))  # set BIG ticks
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))  # set small ticks

        # Define the range and interval for the x-axis ticks
        x_ticks = np.arange(1995, 2025, 5).tolist() + [2023]
        x_ticks_labels = [str(year) for year in x_ticks]
        x_ticks = np.array(x_ticks) - 1994
        x_ticks_labels[0] = '$\sim$1995'  # Rename the first tick label

        ax.set_xticks(x_ticks, x_ticks_labels, fontsize=plot_config.FONT_SIZE)  # Set custom ticks and labels

        y_ticks = np.arange(0, YMAX + 1, INTERVAL)  # Create ticks up to the nearest 5

        # Replace the highest tick value with 'max'
        y_ticks_labels = [str(int) for int in y_ticks]
        if y_ticks_labels:
            y_ticks_labels[-1] = f'$\ge${YMAX}'  # Annotate the maximum value as 'max'

        ax.set_yticks(y_ticks, y_ticks_labels, fontsize=plot_config.FONT_SIZE)  # Set custom ticks and labels

        ax.set_ylim(1, YMAX)

        # Invert y-axis to show higher ranks at the top
        # plt.gca().invert_yaxis()
        ax.invert_yaxis()





        loc = 'lower right' if subject in ["Physics", "Mathematics"] else 'best'
        ax.legend(title='',
                   fontsize=plot_config.FONT_SIZE,
                   framealpha=0.5,

                   loc=loc,
                   # title_fontsize=plot_config.FONT_SIZE
                   )

        for side in ['top', 'right', 'bottom', 'left']:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color('grey')

        plt.tight_layout()

        path = osp.join(args.output_dir, "visual", "keyword_ranks",
                        f"lineplot_{subject.replace(' ', '_')}_keywords.pdf")

        os.makedirs(osp.dirname(path), exist_ok=True)

        plt.savefig(path, dpi=300)

        plt.show()

    print("Done")


if __name__ == "__main__":
    main()
