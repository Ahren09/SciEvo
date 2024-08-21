import os.path
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import pytz
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

import const
from arguments import parse_args
from utility.metrics import calculate_citation_diversity
from utility.utils_data import load_semantic_scholar_papers, \
    load_semantic_scholar_references_parquet, load_keywords, load_arXiv_data
from utility.utils_misc import project_setup
from utility.utils_time import time_difference_in_days


def convert_arxiv_url_to_id(url: str):
    id = url.split("arxiv.org/abs/")[-1]
    if id[-2] == 'v':
        id = id[:-2]

    elif id[-3] == 'v':
        id = id[:-3]
    return id

if __name__ == "__main__":
    project_setup()
    args = parse_args()

    arXivID2References, arXivID2SemanticScholarPaper = {}, {}

    # references_one_year = load_semantic_scholar_references_parquet(start_year=1990, end_year=2024,
    #                                                                data_dir=args.data_dir)

    semantic_scholar_papers = load_semantic_scholar_papers(args.data_dir)


        # for month in range(1, 13):
        #     references_one_month = load_semantic_scholar_references(year, month, args.data_dir)
        #     semantic_scholar_papers_one_month = load_semantic_scholar_papers(year, month, args.data_dir)
        #     arXivID2References.update(references_one_month)
        #     arXivID2SemanticScholarPaper.update(semantic_scholar_papers_one_month)

    # TODO: Get citation age

    arxiv_data = load_arXiv_data(args.data_dir)

    arxiv_data[f'{args.feature_name}_keywords'] = arxiv_data[f'{args.feature_name}_keywords'].apply(
                    lambda x: set([paper_kwd.strip() for paper_kwd in x.lower().split(',')]))

    path_mask = os.path.join(args.data_dir, "NLP", "arXiv", "topic_mask.parquet")

    if os.path.exists(path_mask):
        mask_df = pd.read_parquet(path_mask)

    else:

        mask_df = pd.DataFrame(index=arxiv_data.index)
        # Iterate over each subject and topic in SUBJECT2KEYWORDS
        for subject, topics in const.SUBJECT2KEYWORDS.items():
            for topic, keywords in tqdm(topics.items(), desc=subject):
                # Initialize a column for each topic
                mask_df[topic] = arxiv_data[f'{args.feature_name}_keywords'].apply(
                    lambda x: any(kw in x for kw in (
                        keywords))
                )

        """
        # Get mask for papers that contain keywords
        mask = {}
        for index_row, row in tqdm(arxiv_data.iterrows(), total=len(arxiv_data)):
            # print(row)
            contains_keywords = False
            # for keyword in const.SUBJECT2KEYWORDS['Computer Science'][8]:
            keywords_one_paper = set([kwd.strip().lower() for kwd in row[f"{feature_name}_keywords"].split(',')])

            for keyword in const.SUBJECT2KEYWORDS['Computer Science'][8]:
                # print(keywords_one_paper)

                if keyword.lower() in keywords_one_paper:
                    contains_keywords = True
                    # print(keyword)

            id = row['id']
            # id = row['id'].split("arxiv.org/abs/")[-1]
            # if id[-2] == 'v':
            #     id = id[:-2]

            # elif id[-3] == 'v':
            #     id = id[:-3]

            mask[id] = contains_keywords

        mask = pd.Series(mask).reset_index(name='contains_keyword').rename({'index': 'arXivId'}, axis=1)
        mask['arXivId'] = mask['arXivId'].apply(convert_arxiv_url_to_id)
        mask = mask.drop_duplicates("arXivId").reset_index(drop=True)
        mask.to_csv("mask.csv", index=False)
        """
        mask_df.to_parquet(path_mask)

    arxiv_data['arXivId'] = arxiv_data['id'].apply(convert_arxiv_url_to_id)
    arxiv_data = arxiv_data.drop_duplicates('arXivId')



    # Set the timezone once
    timezone = pytz.timezone('UTC')

    years = [1990, 2005, 2010] + list(np.arange(2011, 2025))
    print("TODO: years")
    # years = list(np.arange(2023, 2025))

    def calculate_topical_diversity(references):
        topic_counts = defaultdict(float)
        total_topics = 0

        for reference in references:
            cited_paper = reference.get('citedPaper', {})
            fields_of_study = cited_paper.get('s2FieldsOfStudy', [])

            if fields_of_study is None:
                continue
            # Get all unique topics from the reference
            topics = [field['category'] for field in fields_of_study if field['source'] == 's2-fos-model']
            unique_topics = set(topics)

            # Update counts
            if unique_topics:
                weight = 1.0 / len(unique_topics)
                for topic in unique_topics:
                    topic_counts[topic] += weight
                total_topics += 1

        if len(topic_counts) > 0:

            # Calculate percentages
            topic_percentages = {topic: count / total_topics for topic, count in topic_counts.items()}

        else:
            topic_percentages = None

        return topic_percentages


    def calculate_age_of_citations(reference_one_paper):

        age_of_citations_one_paper = []
        for paper in reference_one_paper['references']:
            # Skip if publication date is unknown
            if paper['citedPaper']['publicationDate'] is None:
                continue

            else:
                # Parse the publication date of the cited paper
                timestamp_reference = timezone.localize(
                    datetime.strptime(paper['citedPaper']['publicationDate'], "%Y-%m-%d"))

                # Ensure the arXivPublicationDate is a datetime object
                if not isinstance(reference_one_paper['arXivPublicationDate'], datetime):
                    raise NotImplementedError()

                # Calculate the age of the citation in seconds
                age_of_citation = max(0, (
                            reference_one_paper['arXivPublicationDate'] - timestamp_reference).total_seconds())
                age_of_citations_one_paper.append(age_of_citation)

        return age_of_citations_one_paper


    age_of_citations, citation_diversity = {}, {}

    # Count the age of citation for each of the 2 million papers
    # Count the ratio of citations
    for year in years:
        references_one_snapshot = load_semantic_scholar_references_parquet(args.data_dir, year)



        total_age = 0
        count = 0
        # Convert arXivId to Categorical if it has limited unique values
        references_one_snapshot['arXivId'] = references_one_snapshot['arXivId'].astype('category')

        # Iterate through each paper
        for index_row, reference_one_paper in tqdm(references_one_snapshot.iterrows(), desc=f"References "
                                                                                                     f"Year={year}",
                                                   total=len(references_one_snapshot)):

            references = reference_one_paper['references']
            citation_diversity[reference_one_paper['arXivId']] = calculate_topical_diversity(references)


            age_of_citations_one_paper = calculate_age_of_citations(reference_one_paper)

            if len(age_of_citations_one_paper) > 0:
                age_of_citations[reference_one_paper['arXivId']] = age_of_citations_one_paper

    # Convert Series to DataFrames
    citation_diversity = pd.Series(citation_diversity).to_frame(name='diversity')
    age_of_citations = pd.Series(age_of_citations).to_frame(name='aoc')


    # Outer join using merge
    result = pd.merge(citation_diversity, age_of_citations, left_index=True, right_index=True, how='outer')

    result.to_parquet(os.path.join(args.data_dir, "NLP", "arXiv", "citation_age_and_diversity.parquet"))



    # Convert the relevant arXivId list to a set once before the loop
    relevant_arxiv_ids = set(mask[mask['contains_keyword']]['arXivId'])

    # Filter using the precomputed set of arXivIds
    references_one_snapshot_filtered = references_one_snapshot[
        references_one_snapshot['arXivId'].isin(relevant_arxiv_ids)
    ]

    print(f"Year={year}, #papers after filtering={len(references_one_snapshot_filtered)}")

    age_of_citations_filtered = pd.Series(age_of_citations).to_frame("time_difference").reindex(
        np.array(list(relevant_arxiv_ids)))
    age_of_citations_filtered.dropna(inplace=True)

    age_of_citations_filtered = pd.merge(age_of_citations_filtered.reset_index(), arxiv_data[[
        "arXivId",
                                                                                                        "published"]],
                                left_on="index",
                                right_on="arXivId", how='left')

    # Ensure time_differences are lists of numbers
    age_of_citations_filtered['time_differences'] = age_of_citations_filtered['time_differences'].apply(
        lambda x: pd.Series(x).mean())

    # Extract the year-month from the 'published' column
    age_of_citations_filtered['year_month'] = age_of_citations_filtered['published'].dt.to_period('M')

    # Group by the 'year_month' and calculate the mean of 'time_differences'
    mean_time_differences = age_of_citations_filtered.groupby('year_month')['time_differences'].mean()
    # Convert the result back to a DataFrame if needed
    mean_time_differences = mean_time_differences.reset_index()

    median_time_differences = age_of_citations_filtered.groupby('year_month')['time_differences'].median()
    # Convert the result back to a DataFrame if needed
    median_time_differences = median_time_differences.reset_index()

    # Plot the line plot
    plt.figure(figsize=(10, 6))
    plt.plot(mean_time_differences['year_month'].astype(str), mean_time_differences['time_differences'], marker='o')

    # Set the y-axis to log scale
    plt.yscale('log')

    # Set the title and labels
    plt.title('Mean Time Differences Over Time')
    plt.xlabel('Year-Month')
    plt.ylabel('Mean Time Differences')

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


    # ------------------------------
    # archived code


    # Aggregate based on month

    keywords = load_keywords(args.data_dir, args.feature_name)

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
