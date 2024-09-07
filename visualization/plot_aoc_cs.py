"""

Date: 2024-09-13

Run analysis/analyze_citation.py first to generate the statistics.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from arguments import parse_args
from utility.utils_misc import project_setup


FONTSIZE = 16

project_setup()
args = parse_args()

df = pd.read_csv(os.path.join(args.output_dir, "stats", "aoc_cs.csv"))

df = df[df['AoC'] < 1.2e9]

# Compute median AoC per subject and sort by it
median_order = df.groupby('subject')['AoC'].median().sort_values().index

plt.figure(figsize=(15, 10), dpi=300)

# Set up the plot with sorted subjects
sns.stripplot(
    data=df,
    x="AoC",
    y="subject",
    hue="subject",  # Map subjects to the y-axis
    # order=median_order,  # Sort subjects by median AoC
    dodge=True,
    alpha=.005,
    zorder=1,
    legend=False
)

plt.xlabel("Age of Citation (AoC) in seconds", fontsize=FONTSIZE)
plt.ylabel("Subcategories of Computer Science", fontsize=FONTSIZE)
plt.savefig(os.path.join(args.output_dir, "visual", "AoC_CS.png"), dpi=600)
plt.show()