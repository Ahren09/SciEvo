import os

import matplotlib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from arguments import parse_args
from visualization import plot_config

matplotlib.rcParams['figure.dpi'] = 300

SECONDS_IN_A_DAY = 60 * 60 * 24

args = parse_args()

path = os.path.join(args.output_dir, "stats", "citation_diversity_and_aoc_by_topic.xlsx")

citation_diversity_df = pd.read_excel(path, sheet_name='Diversity', index_col=0)

# Define the topics where the background color should be changed

# Computer Science topics
CS_Topics = [
    'Question Answering',
    'Recommender Systems',
    'Computer Vision',
    'CNN',
    'Natural Language Processing',
    'Large Language Models',
    'Explainable AI',
    'Multi-lingual Applications',
    'Quantum Computing',
    'Reinforcement Learning',
    'Cybersecurity',
    'Graph Neural Networks',
    'Edge Computing',
    'Fintech',
    '5G',
    'Internet Of Things',
    'Social Computing'
]

# Mathematics topics
Math_Topics = [
    'Algebraic Geometry',
    'Graph Theory',
    'Optimization',
    'PDE'
]

# Physics topics
Physics_Topics = [
    'Dark Energy',
    'Cosmology',
    'Particle Physics',
    'Quantum Mechanics',
    'Topological Insulators'
]

# Biology topics
Biology_Topics = [
    'Mathematical Biology',
    'Marine Biology',
    'Plant Biology',
    'Evolutionary Biology',
    'Developmental Biology',
    'Genome Engineering',
    'Neuroscience',
    'Stem Cell Therapy',
    'Microbiome Therapeutics',
    'Epidemiology',
    'Microbiology',
    'Pandemics',
    'Synthetic Biology'
]

# Medicine topics
Medicine_Topics = [
    'Precision Medicine',
    'Radiology',
    'Bioinformatics',
    'Immunotherapy',
    'Cognitive Strategies',
    'Digital Health',
    'Mental Health',
    'COVID-19'
]

# Humanities and Social Sciences topics
Humanities_SocialSciences_Topics = [
    'Oral History',
    'Behavioral Economics',
    'Multiculture',
    'Corporate Governance',
    'Narrative History'
]

# Engineering topics
Engineering_Topics = [
    'Additive Manufacturing',
    'Solar Energy',
    'Sustainability'
]

citation_diversity_df = citation_diversity_df.loc[['Algebraic Geometry',
                                                   'Dark Energy',
                                                   'Cosmology',
                                                   'Particle Physics',
                                                   'Topological Insulators',
                                                   'Question Answering',
                                                   'Quantum Mechanics',
                                                   'Recommender Systems',
                                                   'Computer Vision',
                                                   'CNN',

                                                   'Graph Theory',
                                                   'Natural Language Processing',
                                                   'Large Language Models',

                                                   'Precision Medicine',
                                                   'Explainable AI',
                                                   'PDE',
                                                   'Multi-lingual Applications',
                                                   'Quantum Computing',
                                                   'Reinforcement Learning',
                                                   'Cybersecurity',
                                                   'Behavioral Economics',
                                                   'Oral History',
                                                   'Mathematical Biology',

                                                   'Multiculture',
                                                   'Graph Neural Networks',
                                                   'Fintech',

                                                   'Optimization',
                                                   'Radiology',
                                                   'Edge Computing',

                                                   'Bioinformatics',
                                                   '5G',
                                                   'Immunotherapy',
                                                   'Cognitive Strategies',
                                                   'Corporate Governance',
                                                   'Internet Of Things',
                                                   'Evolutionary Biology',

                                                   'Additive Manufacturing',
                                                   'Marine Biology',
                                                   'Plant Biology',
                                                   'Digital Health',
                                                   'Solar Energy',
                                                   'Social Computing',
                                                   'Developmental Biology',
                                                   'Genome Engineering',
                                                   'Neuroscience',
                                                   'Narrative History',
                                                   'Mental Health',
                                                   'Sustainability',
                                                   'COVID-19',
                                                   'Stem Cell Therapy',
                                                   'Microbiome Therapeutics',
                                                   'Epidemiology',
                                                   'Microbiology',
                                                   'Pandemics',
                                                   'Synthetic Biology']]

citation_diversity_df = citation_diversity_df.sort_values(by='Simpson', ascending=True).reset_index().rename(
    columns={'index': 'Topic'})

citation_diversity_df_melted = pd.melt(citation_diversity_df, id_vars=['Topic'], value_vars=['Simpson', 'Shannon',
                                                                                             'Gini'], var_name='Metric',
                                       value_name='Value')

# Set plot style
sns.set(style="whitegrid")

# Create the lineplot
plt.figure(figsize=(12, 6))

LINEWIDTH = 3
MARKERSIZE = 7

ax = sns.lineplot(data=citation_diversity_df_melted, x='Topic', y='Value', hue='Metric', marker='o',
                  linewidth=LINEWIDTH, markersize=MARKERSIZE)

# Rotate x-axis labels for better readability
plt.xticks(rotation=90)

# -----------------
# Background Color
# -----------------

COLORS = ["#84e3c8", "#a8e6cf", "#dcedc1", "#ffd3b6", "#ffaaa5", "#ff8b94", "#ff7480"]
COLORS = ["#ffadad", "#ffd6a5", "#fdffb6", "#caffbf", "#9bf6ff", "#a0c4ff", "#bdb2ff"]

x_tick_labels = ax.get_xticks()

ALL_TOPICS = {
    "CS": CS_Topics,
    "Physics": Physics_Topics,
    "Math": Math_Topics,
    "Medicine": Medicine_Topics,
    "Biology": Biology_Topics,
    "Humanities / Social Science": Humanities_SocialSciences_Topics,
    "Engineering": Engineering_Topics
}
ALL_TOPICS_LIST = [topic for topic_list in ALL_TOPICS for topic in topic_list]

# Loop through the x_tick_labels and color the background accordingly
for index_topic, label in enumerate(citation_diversity_df['Topic']):
    for index_topic_category, (category_name, topics) in enumerate(ALL_TOPICS.items()):
        if label in topics:
            ax.axvspan(x_tick_labels[index_topic] - 0.5, x_tick_labels[index_topic] + 0.5,
                       color=COLORS[index_topic_category], alpha=0.3)

ax.set_title('Citation Diversity Across Topics', fontsize=plot_config.FONT_SIZE)

ax.set_ylabel('Metrics', fontsize=plot_config.FONT_SIZE)
ax.set_xlabel('Academic Topics', fontsize=plot_config.FONT_SIZE)

# Add legend for the Metrics
# ax.legend(title="Metrics", title_fontsize=plot_config.FONT_SIZE + 2, prop={'size': plot_config.FONT_SIZE})


# Get the line handles for the "Metrics" legend from the seaborn plot
handles, labels = ax.get_legend_handles_labels()

# Add legend for the background colors
# Create custom legend elements for topic categories
legend_elements = []
for index_topic, category in enumerate(ALL_TOPICS.keys()):
    legend_elements.append(Patch(facecolor=COLORS[index_topic], label=category))

# Combine the metric handles with category patches
combined_legend_elements = handles + legend_elements
combined_legend_labels = labels + list(ALL_TOPICS.keys())

# Add the merged legend to the plot
plt.legend(handles=combined_legend_elements, labels=combined_legend_labels, title="Metrics and Categories",
           title_fontsize=12, prop={'size': 10}, loc='upper left',
           # bbox_to_anchor=(0.5, 1.15),
           facecolor='white', framealpha=0.5,
           frameon=True, edgecolor='white',
           ncol=2)

# Show the plot
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "visual", 'Citation_Diversity_by_topics.pdf'), dpi=300)
plt.show()

print(citation_diversity_df)
