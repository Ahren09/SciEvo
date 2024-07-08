"""
Visualize changes of the rank of popular keywords over time.

Run `analysis/analyze_keyword_ranks.py` first to generate the data.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os.path as osp
import matplotlib.ticker as ticker

from arguments import parse_args
from utility.utils_misc import project_setup

# Create a DataFrame for the data


if __name__ == "__main__":
    project_setup()
    args = parse_args()
    data = pd.read_excel(osp.join(args.output_dir, "all_ranks_of_interest.xlsx"), index_col=0)

    YMAX = 100
    INTERVAL = 20



    for column in data.columns:
        data[column] = data[column].apply(lambda x: YMAX if x > YMAX else x)


    d = {
        "Machine Learning": ["machine learning", "deep learning", "large language models", "transformer", "llms"],
        "Mathematics": ["algebraic geometry", "differential geometry", "graph theory"],
        "Epidemiology": ["covid-19"],
        "Cosmology": ["dark matter", "gravitational waves", "star formation", "black holes"],
        "Economics": ["inflation", "optimal control ", "game theory"],
        "Physics": ["quantum computing", "quantum field theory", "quantum gravity", "superconductivity"],
    }


    for subject, keywords_of_interest in d.items():

        # Create the line plot
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        sns.lineplot(data=data.T[keywords_of_interest], marker='o')



        # Styling the plot
        plt.title(f'Rank Changes of Popular Keywords in {subject} (1985-2024)')
        plt.xlabel('Year')
        plt.ylabel('Rank')

        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))  # set BIG ticks
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))  # set small ticks

        # plt.yticks(range(1, 10), labels=[f'Rank {i}' for i in range(1, 10)])
        # Invert y-axis to show better ranks at the top and adjust the y-axis ticks and labels
        # plt.yticks(range(1, YMAX, 10), np.arange(1, YMAX, 10).tolist() + [f'{YMAX} (or beyond)'])

        # x_ticks = np.arange(1985, 2030, 5)
        # x_ticks_labels = [str(int) for int in x_ticks]
        # if x_ticks_labels:
        #     x_ticks_labels[0] = f'Before 1995'  # Annotate the minimum year
        #
        # plt.xticks(x_ticks, x_ticks_labels)  # Set custom ticks and labels

        # plt.xlim(1994, 2024)

        # Define the range and interval for the x-axis ticks
        x_ticks = [1994] + np.arange(1995, 2025, 5).tolist()
        x_ticks_labels = [str(year) for year in x_ticks]
        x_ticks = np.array(x_ticks) - 1994
        x_ticks_labels[0] = 'before\n1995'  # Rename the first tick label

        plt.xticks(x_ticks, x_ticks_labels)  # Set custom ticks and labels

        y_ticks = np.arange(0, YMAX + 1, INTERVAL)  # Create ticks up to the nearest 5

        # Replace the highest tick value with 'max'
        y_ticks_labels = [str(int) for int in y_ticks]
        if y_ticks_labels:
            y_ticks_labels[-1] = f'{YMAX}\nor beyond'  # Annotate the maximum value as 'max'

        plt.yticks(y_ticks, y_ticks_labels)  # Set custom ticks and labels

        plt.ylim(1, YMAX)

        # Invert y-axis to show higher ranks at the top
        plt.gca().invert_yaxis()

        plt.grid(False)
        sns.set(style="whitegrid")

        plt.show()
        print("Done")



