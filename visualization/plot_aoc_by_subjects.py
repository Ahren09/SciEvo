import os

import matplotlib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from arguments import parse_args
from visualization import plot_config

matplotlib.rcParams['figure.dpi'] = 300

SECONDS_IN_A_YEAR = 86400 * 365

args = parse_args()

path = os.path.join(args.output_dir, "stats", "citation_diversity_and_aoc_by_topic.xlsx")

aoc_df = pd.read_excel(path, sheet_name='AoC', index_col=0)
citation_diversity_df = pd.read_excel(path, sheet_name='Diversity', index_col=0)

aoc_df = aoc_df.loc[['Large Language Models', 'Question Answering', '5G', 'Multi-lingual Applications',
                     'Computer Vision',
                     'Natural Language Processing', 'COVID-19', 'Fintech', 'Internet Of Things', 'Digital Health',
                     'Edge Computing', 'CNN', 'Recommender Systems', 'Mental Health',
                     'Social Computing', 'Explainable AI', 'Reinforcement Learning', 'Cybersecurity', 'Sustainability',
                     'Additive Manufacturing', 'Graph Neural Networks', 'Stem Cell Therapy', 'Genome Engineering',
                     'Precision Medicine', 'Radiology', 'Bioinformatics', 'Microbiology',
                     'Multiculture', 'Solar Energy', 'Epidemiology', 'Corporate Governance',
                     'Particle Physics', 'Cognitive Strategies', 'Plant Biology', 'Pandemics',
                     'Neuroscience', 'Dark Energy', 'Microbiome Therapeutics',
                     'Optimization', 'Oral History', 'Synthetic Biology', 'Mathematical Biology', 'Cosmology',
                     'Topological Insulators', 'Quantum Computing', 'Immunotherapy', 'Behavioral Economics',
                     'Marine Biology',
                     'Graph Theory', 'Developmental Biology', 'Narrative History',
                     'Evolutionary Biology', 'PDE', 'Quantum Mechanics', 'Algebraic Geometry']]

# First, reset the index to turn the index into a column
aoc_df = aoc_df.reset_index().rename(columns={'index': 'Topic'})

# # Now, melt the DataFrame for easy plotting
# aoc_df_melted = pd.melt(aoc_df, id_vars=['Topic'], value_vars=['Mean AoC', 'Std AoC', 'Median AoC'],
#                         var_name='Metric', value_name='Value')

# Convert the AoC values from seconds to days
aoc_df['Mean AoC'] = aoc_df['Mean AoC'] / SECONDS_IN_A_YEAR
aoc_df['Std AoC'] = aoc_df['Std AoC'] / SECONDS_IN_A_YEAR
aoc_df['Median AoC'] = aoc_df['Median AoC'] / SECONDS_IN_A_YEAR

# Now, melt the DataFrame for easy plotting
aoc_df_melted = pd.melt(aoc_df, id_vars=['Topic'], value_vars=['Mean AoC', 'Std AoC', 'Median AoC'],
                        var_name='Metric', value_name='Value')

# Set plot style
sns.set(style="whitegrid")

# Create the lineplot
plt.figure(figsize=(12, 6))
# sns.lineplot(data=aoc_df_melted, x='Topic', y='Value', hue='Metric', marker='o')

LINEWIDTH = 3
MARKERSIZE = 7
sns.lineplot(data=aoc_df, x='Topic', y='Mean AoC', marker='o', label='Mean ± Std AoC', color='#219ebc',
             linewidth=LINEWIDTH, markersize=MARKERSIZE)

sns.lineplot(data=aoc_df, x='Topic', y='Median AoC', marker='o', label='Median AoC', color='#fb8500',
             linewidth=LINEWIDTH, markersize=MARKERSIZE)



# Add the shaded area using matplotlib fill_between for Mean ± Std
plt.fill_between(aoc_df['Topic'],
                 aoc_df['Mean AoC'] - aoc_df['Std AoC'],
                 aoc_df['Mean AoC'] + aoc_df['Std AoC'],
                 color='#8ecae6', alpha=0.5, label='')


# Rotate x-axis labels for better readability
plt.xticks(rotation=90)

# Add title and labels
plt.title('Age of Citations (AoC) Across Topics', fontsize=plot_config.FONT_SIZE)
plt.ylabel('Age of Citations (AoC) In Years', fontsize=plot_config.FONT_SIZE)
plt.xlabel('Academic Topics', fontsize=plot_config.FONT_SIZE)
plt.legend(title="Metrics", title_fontsize=plot_config.FONT_SIZE + 2, prop={'size': plot_config.FONT_SIZE})

# Show the plot
plt.tight_layout()

path = os.path.join(args.output_dir, "visual", "AoC", 'AoC_by_topics.pdf')

os.makedirs(os.path.dirname(path), exist_ok=True)

plt.savefig(path, dpi=300)
plt.show()

print("Done!")
