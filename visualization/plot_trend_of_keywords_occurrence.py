import json
import os
import os.path as osp
import sys
import traceback
import warnings
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from utility.utils_data import load_arXiv_data
from utility.utils_text import load_semantic_scholar_data

sys.path.insert(0, os.path.abspath('..'))

import const
from arguments import parse_args

# Suppress FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

START_YEAR, START_MONTH = 1990, 1
END_YEAR, END_MONTH = 2024, 5

args = parse_args()

semantic_scholar_data = load_semantic_scholar_data(args.data_dir, START_YEAR, START_MONTH, END_YEAR, END_MONTH)

data = load_arXiv_data(args.data_dir, start_year=START_YEAR, start_month=START_MONTH, end_year=END_YEAR,
                               end_month=END_MONTH)

ngrams = json.load(open(osp.join(args.output_dir, "ngrams.json"), "r"))
print(f"Number of ngrams: {len(ngrams)}")

keyword2SSId_list = {subject: defaultdict(set) for subject in const.SUBJECT2KEYWORDS}

G = {}

# Temporal Graph
for subject in const.SUBJECT2KEYWORDS:
    for keywords_tuple in const.SUBJECT2KEYWORDS[subject]:
        assert keywords_tuple[0] not in G
        G[keywords_tuple[0]] = defaultdict(set)

"""
Structure of keyword2SSId_list: 
{
    subject -> {
        keyword -> [list of paper],
        keyword -> [list of paper],
        ...
    },
    subject -> {
        keyword -> [list of paper],
        keyword -> [list of paper],
        ...
    },
}

"""

# Pre-extract necessary data from DataFrame for faster speed
fos_data = semantic_scholar_data[const.FOS].tolist()
semantic_scholar_id_data = semantic_scholar_data[const.PAPERID].tolist()
published_data = data[const.PUBLISHED].dt.strftime("%Y-%m").tolist()

# Construct temporal graph

for idx_example, ngrams_one_example in enumerate(tqdm(ngrams)):

    try:

        subject_one_paper = fos_data[idx_example]

        semantic_scholar_id = semantic_scholar_id_data[const.PAPERID]

        published_year_and_month = published_data[idx_example].strftime(
            "%Y-%m")

        for subject in subject_one_paper:

            if not subject in const.SUBJECT2KEYWORDS:
                continue

            for keywords_tuple in const.SUBJECT2KEYWORDS[subject]:

                common_keywords = set(ngrams_one_example) & set(keywords_tuple)

                if common_keywords:
                    print("=====================================")
                    print(f"ngrams: {ngrams_one_example}")
                    print(f"Subject: {subject}")
                    print(f"Common keywords: {common_keywords}")

                    # Each tuple in `keywords_tuple` contains a list of related keywords
                    # We use the first keyword in each tuple to represent the group

                    representative_keyword = keywords_tuple[0]

                    keyword2SSId_list[subject][representative_keyword].add(semantic_scholar_id)

                    G[representative_keyword][published_year_and_month].add(semantic_scholar_id)



    except:
        traceback.print_exc()

stack_data = []
snapshot_names = []
for keyword in const.KEYWORD2ID:
    num_papers = []

    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            if (month < START_MONTH and year <= START_YEAR) or (month >= END_MONTH and year >= END_YEAR):
                continue

            snapshot_name = f"{year}-{month:02d}"
            num_papers += [len(G[keyword][snapshot_name])]

            if snapshot_name not in snapshot_names:
                snapshot_names += [snapshot_name]

    stack_data += [num_papers]

stack_data = np.array(stack_data)

# Mask out keywords that have no papers
mask = np.array(stack_data).sum(axis=1) > 20

labels = np.array(list(const.KEYWORD2ID.keys()))[mask].tolist()
plt.stackplot(np.arange(len(snapshot_names)), *stack_data[mask], labels=labels)
plt.legend(loc='upper left')
plt.show()
