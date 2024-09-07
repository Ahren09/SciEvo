import os
import os.path as osp
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath('.'))

from utility.utils_data import load_keywords
from arguments import parse_args
from utility.utils_misc import project_setup


def main():
    project_setup()

    args = parse_args()

    keywords = load_keywords(args.data_dir, args.feature_name)
    keywords['year'] = keywords['published'].dt.year

    # Dictionary to store keyword counts for each time range
    keyword_counts_dict: dict = {}

    # List for storing most common keywords
    most_common_keywords: list = []

    time_ranges = [1985] + np.arange(1995, 2026, 1).tolist()

    """
    keywords_of_interest = ['algebraic geometry',
                            'cosmology',
                            'covid-19',
                            'dark matter',
                            'deep learning',
                            'differential geometry',
                            'geometry',
                            'graph theory',
                            'graphene',
                            'lattice qcd',
                            'language models',
                            'large language models',
                            'machine learning',
                            'mgb2',
                            'natural language processing',
                            'neural networks',
                            'nuclear theory',
                            'optimization',
                            'particle physics',
                            'quantum algebra',
                            'quantum computing',
                            'quantum field theory',
                            'quantum gravity',
                            'quantum groups',
                            'quantum mechanics',
                            'reinforcement learning',
                            'superconductivity']
                            
    """
    keywords_of_interest: set = set()

    keyword_ranks = {}

    for i in range(len(time_ranges) - 1):
        start, end = time_ranges[i], time_ranges[i + 1]

        # Filter the keywords by the given time range (a year)
        snapshot = keywords[(keywords['year'] >= start) & (keywords['year'] < end)]

        # Group by year and keyword, then count the occurrences
        keyword_counts = snapshot[['keywords']].explode('keywords').groupby(['keywords']).size().sort_values(
            ascending=False)

        print("=" * 30)
        print(f"Most common keywords ({start}-{end})")
        print(keyword_counts.head(20))

        keyword_counts_dict[f"{start}"] = keyword_counts

        most_common_keywords += [keyword_counts.head(100).rename(f"{start}").reset_index()]

        keywords_of_interest.update(keyword_counts.head(100).index.tolist())

        # Get the rank of all keywords
        keyword_ranks[f"{start}"] = keyword_counts.rank(ascending=False, method='min').astype(int)

    ranks_of_interest = {}
    for i in range(len(time_ranges) - 1):
        start = time_ranges[i]

        # Create a Series with NaN values for the keywords of interest
        ranks_of_interest[f"{start}"] = pd.Series(index=keywords_of_interest, data=np.nan, dtype=int)

        # Update the interest_series with the ranks from keyword_ranks

        # Absolute ranks
        ranks_of_interest[f"{start}"].update(keyword_ranks[f"{start}"])

        # Percentile
        # ranks_of_interest[f"{start}"].update(all_ranks[f"{start}"] / len(all_ranks[f"{start}"]) * 100)

    ranks_of_interest_df = pd.DataFrame(ranks_of_interest)

    # Sort the keywords by the sum of their ranks so that the most popular keywords across all years are at the top
    ranks_of_interest_df = ranks_of_interest_df.sort_index(key=ranks_of_interest_df.sum(1).get)
    # ranks_of_interest_df = ranks_of_interest_df.sort_index(by=ranks_of_interest_df.sum(axis=1), ascending=True)

    path = osp.join(args.output_dir, "stats", f"{args.feature_name}_keyword_ranks.xlsx")
    ranks_of_interest_df.to_excel(path)

    print("Done!")

if __name__ == "__main__":
    main()
