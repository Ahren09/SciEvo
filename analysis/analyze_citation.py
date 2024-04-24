import json
import os.path as osp
from collections import defaultdict
from datetime import datetime

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from arguments import parse_args
from utility.metrics import calculate_citation_diversity
from utility.utils_misc import project_setup
from utility.utils_time import time_difference_in_days


def load_references(year, month, data_dir):
    path = osp.join(data_dir, "NLP", "semantic_scholar", f"references_{year}_{month}.json")
    with open(path) as f:
        paper2references = json.load(f)

    # Filter out references that do not have the necessary information (likely null)
    for arXivID in paper2references:
        paper2references[arXivID] = [ref for ref in paper2references[arXivID] if (ref.get('citedPaper') is not None and
                                                                                  all(ref[
                                                                                          'citedPaper'].get(
                                                                                      key) is not None for key in
                                                                                      ['paperId', 'fieldsOfStudy',
                                                                                       'publicationDate']))]
    return paper2references


def load_semantic_scholar_papers(year, month, data_dir):
    path = osp.join(data_dir, "NLP", "semantic_scholar", f"semantic_scholar_{year}_{month}.json")
    with open(path) as f:
        semantic_scholar_papers = json.load(f)

    return semantic_scholar_papers


if __name__ == "__main__":
    project_setup()
    args = parse_args()

    arXivID2References, arXivID2SemanticScholarPaper = {}, {}

    for year in range(1990, 1995):
        for month in range(1, 2):
            references_one_month = load_references(year, month, args.data_dir)
            semantic_scholar_papers_one_month = load_semantic_scholar_papers(year, month, args.data_dir)
            arXivID2References.update(references_one_month)
            arXivID2SemanticScholarPaper.update(semantic_scholar_papers_one_month)

    # TODO: Get citation age

    # Dictionary to store aggregated results
    aggregated_results = {}

    diversity_stats_df = pd.DataFrame()

    # Iterate over each paper
    for source_paper_arXivID, source_PaperInfo in tqdm(arXivID2SemanticScholarPaper.items()):
        fieldsOfStudy_Source = source_PaperInfo.get('fieldsOfStudy', [])
        source_paper_SemanticScholarID = source_PaperInfo.get('paperId')

        if fieldsOfStudy_Source in [None, 'n/a'] or source_paper_SemanticScholarID is None:
            continue

        fieldsOfStudy2NumReferences = defaultdict(int)
        fieldsOfStudy2TimeDiffs = defaultdict(list)

        # Check if the paper has references
        if source_paper_arXivID in arXivID2References:
            references = arXivID2References[source_paper_arXivID]
            publish_date_paper = source_PaperInfo['publicationDate']

            if publish_date_paper is None:
                continue

            # Papers cited by the source paper
            for reference in references:
                reference_info = reference['citedPaper']
                if reference_info is None or reference_info['fieldsOfStudy'] is None or reference_info[
                    'publicationDate'] is None:
                    continue

                publish_date_reference = reference_info['publicationDate']

                time_diff = time_difference_in_days(publish_date_paper, publish_date_reference)

                for fieldsOfStudy_reference in reference_info['fieldsOfStudy']:
                    if fieldsOfStudy_reference not in [None, 'n/a']:
                        fieldsOfStudy2NumReferences[fieldsOfStudy_reference] += 1

                        fieldsOfStudy2TimeDiffs[fieldsOfStudy_reference].append(time_diff)

        fieldsOfStudy2NumReferences = dict(fieldsOfStudy2NumReferences)

        fieldsOfStudy2TimeDiffs = dict(fieldsOfStudy2TimeDiffs)

        # Fill in missing fields of study

        for field in fieldsOfStudy2NumReferences:
            if field not in fieldsOfStudy2TimeDiffs and field not in fieldsOfStudy2NumReferences:
                fieldsOfStudy2TimeDiffs[field] = []
                fieldsOfStudy2NumReferences[field] = 0

        results = calculate_citation_diversity(fieldsOfStudy2NumReferences)

        if results is None:
            continue
        results['arXivID'] = source_paper_arXivID
        results["SemanticScholarID"] = source_paper_SemanticScholarID
        results['fields_of_study'] = fieldsOfStudy_Source
        results['publicationDate'] = datetime.strptime(source_PaperInfo['publicationDate'], '%Y-%m-%d')
        results = pd.Series(results)

        diversity_stats_df = diversity_stats_df.append(results, ignore_index=True)

    diversity_stats_df['year'] = diversity_stats_df['publicationDate'].dt.year

    print("Aggregating results by fields of study (FoS) ...")

    diversity_wrt_FoS = diversity_stats_df.explode('fields_of_study').agg({
        'simpsons_diversity_index': 'mean',
        'shannons_diversity_index': 'mean',
        'normalized_entropy': 'mean',
        'berger_parker_index': 'mean',
        'gini': 'mean'
    })

    print(diversity_wrt_FoS)

    diversity_wrt_FoS_and_year = diversity_stats_df.explode(['fields_of_study']).groupby(
        ['fields_of_study', 'year']).agg({
        'simpsons_diversity_index': 'mean',
        'shannons_diversity_index': 'mean',
        'normalized_entropy': 'mean',
        'berger_parker_index': 'mean',
        'gini': 'mean'
    })

    plt.figure(figsize=(12, 8))  # Adjust the size as per your visualization needs
    sns.lineplot(data=diversity_wrt_FoS_and_year, x='year', y='gini', hue='fields_of_study', marker='o')
    plt.title('Changes in Gini Index Over Years by Fields of Study')
    plt.xlabel('Year')
    plt.ylabel('Gini Index')
    plt.legend(title='Fields of Study')
    plt.grid(True)

    print("Aggregating results by fields of study (FoS) and year ...")
