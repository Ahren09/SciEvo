import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from arguments import parse_args
from utility.utils_misc import project_setup


sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

project_setup()
args = parse_args()

# Create the data
rs = np.random.RandomState(42)
# x = rs.randn(500)
# g = np.tile(list("ABCDEFGHIJ"), 50)
# df = pd.DataFrame(dict(x=x, g=g))
df = pd.read_csv(os.path.join(args.output_dir, "stats", "aoc_all_subjects.csv"))
df = df.dropna(axis=0).reset_index(drop=True)
df = df[df['AoC'] < 1.2e9]

df['AoC'] = df['AoC'] / (60 * 60 * 24 * 365)  # Convert seconds to years

# Initialize the FacetGrid object
pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
g = sns.FacetGrid(df, row="subject", hue="subject", aspect=15, height=.5, palette=pal)

# Draw the densities in a few steps
g.map(sns.kdeplot, "AoC",
      bw_adjust=.5, clip_on=False,
      fill=True, alpha=1, linewidth=1.5)
g.map(sns.kdeplot, "AoC", clip_on=False, color="w", lw=2, bw_adjust=.5)

# passing color=None to refline() uses the hue mapping
g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)


# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .4, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)


g.map(label, "AoC")

# Set the subplots to overlap
g.figure.subplots_adjust(hspace=-.25)

# Adjust the bottom margin to make space for the labels
plt.subplots_adjust(bottom=0.2)


# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)
g.set_axis_labels("Age of Citation (AoC) in years")  # Change the x-label
# plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "visual", "aoc_all_subjects.pdf"), dpi=300)
plt.show()