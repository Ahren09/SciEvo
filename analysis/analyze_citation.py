import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import pytz
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
from datasets import DatasetDict, Dataset

import itertools

import const
from arguments import parse_args
from utility.metrics import (calculate_citation_diversity, simpsons_diversity_index, shannons_diversity_index,
                             gini_simpson_index, gini)
from utility.utils_data import load_semantic_scholar_papers, \
    load_semantic_scholar_references_parquet, load_keywords, load_arXiv_data, convert_arxiv_url_to_id
from utility.utils_misc import project_setup
from utility.utils_time import time_difference_in_days


if __name__ == "__main__":
    project_setup()
    args = parse_args()

    arXivID2References, arXivID2SemanticScholarPaper = {}, {}

    # references_one_year = load_semantic_scholar_references_parquet(start_year=1990, end_year=2024,
    #                                                                data_dir=args.data_dir)

    all_references = []
    
    semantic_scholar_papers = load_semantic_scholar_papers(args.data_dir)
    years = [1990, 2005, 2011] + list(np.arange(2016, 2025))
    for year in years:
        references_one_snapshot = load_semantic_scholar_references_parquet(args.data_dir, start_year=year)
        all_references.append(references_one_snapshot)
        
    all_references = pd.concat(all_references).reset_index(drop=True)
        
    arxiv_data = load_arXiv_data(args.data_dir)
    
    
    
    arxiv_dataset = Dataset.from_pandas(arxiv_data)
    semantic_scholar_dataset = Dataset.from_pandas(semantic_scholar_papers)
    references_dataset = Dataset.from_pandas(all_references)
    
    
    arxiv_dataset.push_to_hub("Ahren09/SciEvo", config_name="arxiv")
    semantic_scholar_dataset.push_to_hub("Ahren09/SciEvo", config_name="semantic_scholar")
    references_dataset.push_to_hub("Ahren09/SciEvo", config_name="references")
    
    # Extract 'ArXiv' externalIds and assign them to the 'arXivId' column
    semantic_scholar_papers['arXivId'] = semantic_scholar_papers['externalIds'].apply(lambda x: x.get('ArXiv', None))

    arxiv_data['arXivId'] = arxiv_data['id'].apply(convert_arxiv_url_to_id)


    semantic_scholar_papers['arXivId'] = semantic_scholar_papers['externalIds'].apply(lambda x: x.get('ArXiv', None))


    for feature_name in ['title', 'title_and_abstract']:
        arxiv_data[f'{feature_name}_keywords'] = arxiv_data[f'{feature_name}_keywords'].apply(
            lambda x: x if isinstance(x, (set, list)) else set(
                [paper_kwd.strip() for paper_kwd in x.lower().split(',')]))

    """
    # Mask w.r.t. topics extracted by LLMs
    path_mask = os.path.join(args.data_dir, "NLP", "arXiv", "topic_mask.parquet")

    if os.path.exists(path_mask):
        mask_df = pd.read_parquet(path_mask)

    else:
        mask_df = pd.DataFrame(index=arxiv_data.index)

        # for subject in const.SUBJECT

        # Iterate over each subject and topic in SUBJECT2KEYWORDS
        for subject, topics in const.SUBJECT2KEYWORDS.items():
            for topic, keywords in tqdm(topics.items(), desc=subject):
                # Initialize a column for each topic
                mask_df[topic] = arxiv_data[f'{args.feature_name}_keywords'].apply(
                    lambda x: any(kw in x for kw in (
                        keywords))
                )


        mask_df['arXivId'] = arxiv_data['arXivId']
        mask_df.to_parquet(path_mask)

    """

    # Mask w.r.t. subject areas extracted by LLMs
    path_mask = os.path.join(args.data_dir, "NLP", "arXiv", "subject_mask.parquet")

    if os.path.exists(path_mask):
        mask_df = pd.read_parquet(path_mask)

    else:
        mask = {}

        mask_df = pd.DataFrame(index=arxiv_data.index)
        for subject in const.ARXIV_SUBJECTS:
            # Create a boolean mask DataFrame indicating if any tag in each row matches desired categories
            mask[subject] = arxiv_data['tags'].apply(lambda tags: any(tag in const.ARXIV_SUBJECTS[subject] for tag in
                                                             tags))
            for tag in const.ARXIV_SUBJECTS[subject]:
                mask[tag] = arxiv_data['tags'].apply(lambda tags: tag in tags)

        mask_df = mask_df.concatenate([mask_df, pd.DataFrame(mask)], axis=1)
        mask_df['arXivId'] = arxiv_data['arXivId']
        mask_df = mask_df.drop_duplicates('arXivId')
        mask_df.to_parquet(path_mask)

    arxiv_data = arxiv_data.drop_duplicates('arXivId')


    # Set the timezone once
    timezone = pytz.timezone('UTC')

    

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

    # Count the age of citation for each of the 2 million papers
    # Count the ratio of citations

    path_citation_age_and_diversity = os.path.join(args.data_dir, "NLP", "arXiv", "citation_age_and_diversity.parquet")

    if os.path.exists(path_citation_age_and_diversity):
        result = pd.read_parquet(path_citation_age_and_diversity)

    else:
        age_of_citations, citation_diversity = {}, {}

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

        result['published'] = arxiv_data.set_index('arXivId').loc[list(set(result.index.tolist()) & set(arxiv_data[
                                                                                                            'arXivId']
                                                                                                        )), 'published']
        result.index = result.index.set_names("arXivId")
        result.to_parquet(path_citation_age_and_diversity)



    # Convert the relevant arXivId list to a set once before the loop

    aggregated_metrics = {}


    mAoC_by_month = {}

    print("TODO: rename columns")

    # ------------------------------
    # Temporal Diversity
    # Extract the year-month from the 'published' column
    result['published_year_month'] = result['published'].dt.to_period('M')

    aggregated_metrics = {}
    path = os.path.join(args.output_dir, "stats", "citation_diversity_and_aoc_by_subject.xlsx")

    if os.path.exists(path):
        aoc_df = pd.read_excel(path, sheet_name='AoC')

    else:

        data = {'subject': [], 'AoC': []}
        for subject in const.ARXIV_SUBJECTS:
            # Create a boolean mask DataFrame indicating if any tag in each row matches desired categories
            relevant_arxiv_ids = set(mask_df[mask_df[subject]]['arXivId'].tolist())
            relevant_arxiv_ids = list(relevant_arxiv_ids & set(result.index))

            result_one_subject = result.loc[relevant_arxiv_ids]

            aoc_list = list(itertools.chain.from_iterable(item for item in result_one_subject['aoc'] if item is not None and len(item) > 0))

            aoc_list = [x for x in aoc_list if x > 0.]



            result_one_subject['mean_aoc'] = result_one_subject['aoc'].apply(
                lambda x: np.mean(x) if isinstance(x, (np.ndarray, list)) else
                None)
            result_one_subject['std_aoc'] = result_one_subject['aoc'].apply(
                lambda x: np.std(x) if isinstance(x, (np.ndarray, list)) else None)
            result_one_subject['median_aoc'] = result_one_subject['aoc'].apply(
                lambda x: np.median(x) if isinstance(x, (np.ndarray,
                                                         list)) else None)

            aggregated_metrics[subject] = {
                'Mean AoC': np.mean(aoc_list),
                'Std AoC': np.std(aoc_list),
                'Median AoC': np.median(aoc_list)
            }
            data['subject'] += [subject] * len(result_one_subject)
            data['AoC'] += result_one_subject['mean_aoc'].tolist()

            for tag in const.ARXIV_SUBJECTS[subject]:
                relevant_arxiv_ids = set(mask_df[mask_df[tag]]['arXivId'].tolist())
                relevant_arxiv_ids = list(relevant_arxiv_ids & set(result.index))

                result_one_subject = result.loc[relevant_arxiv_ids]

                aoc_list = list(itertools.chain.from_iterable(
                    item for item in result_one_subject['aoc'] if item is not None and len(item) > 0))

                result_one_subject['mean_aoc'] = result_one_subject['aoc'].apply(
                    lambda x: np.mean(x) if isinstance(x, (np.ndarray, list)) else
                    None)

                data['subject'] += [subject] * len(result_one_subject)
                data['AoC'] += result_one_subject['mean_aoc'].tolist()




        sns.stripplot(
            data=pd.DataFrame(data), x="AoC", hue="subject",
            dodge=True, alpha=.05, zorder=1, legend=False,
        )

        # Lowest mAoC comes first
        aoc_df = pd.DataFrame(aggregated_metrics).T[['Mean AoC', 'Std AoC', 'Median AoC']].sort_values(
            by='Mean AoC',
            ascending=True)



        with pd.ExcelWriter(path) as writer:
            aoc_df.to_excel(writer, sheet_name='AoC')



    path = os.path.join(args.output_dir, "stats", "citation_diversity_and_aoc_by_topic.xlsx")

    aggregated_metrics = {}

    if os.path.exists(path):
        aoc_df = pd.read_excel(path, sheet_name='AoC')
        citation_diversity_df = pd.read_excel(path, sheet_name='Diversity')

    else:

        for subject, topics in const.SUBJECT2KEYWORDS.items():
            for topic, keywords in tqdm(topics.items(), desc=subject):

                print(f"Calculating metrics for keyword={topic}")
                relevant_arxiv_ids = set(mask_df[mask_df[topic]]['arXivId'].tolist())

                relevant_arxiv_ids = list(relevant_arxiv_ids & set(result.index))

                result_one_topic = result.loc[list(relevant_arxiv_ids)]
                result_one_topic = result_one_topic[~result_one_topic['diversity'].isna()]
                print(f"Topic={topic}, #papers={len(result_one_topic)}")

                result_one_topic['simpson'] = result_one_topic['diversity'].apply(lambda x: simpsons_diversity_index(x))
                result_one_topic['shannon'] = result_one_topic['diversity'].apply(lambda x: shannons_diversity_index(x))
                result_one_topic['gini'] = result_one_topic['diversity'].apply(lambda x: gini(x))

                aoc_list = list(itertools.chain.from_iterable(filter(None, result_one_topic['aoc'])))
                aoc_list = [x for x in aoc_list if x > 0.]


                """
                result_one_topic['mean_aoc'] = result_one_topic['aoc'].apply(lambda x: np.mean(x) if isinstance(x, (np.ndarray, list)) else
                None)
                result_one_topic['std_aoc'] = result_one_topic['aoc'].apply(lambda x: np.std(x) if isinstance(x, (np.ndarray, list)) else None)
                result_one_topic['median_aoc'] = result_one_topic['aoc'].apply(lambda x: np.median(x) if isinstance(x, (np.ndarray,
                                                                                                     list)) else None)
                
                """


                # Change the topic names for better visualization
                if " " in topic:
                    topic = " ".join([word.capitalize() for word in topic.split(' ')])


                aggregated_metrics[topic] = {
                    'Simpson': result_one_topic['simpson'].mean(),
                    'Shannon': result_one_topic['shannon'].mean(),
                    'Gini': result_one_topic['gini'].mean(),
                    'Mean AoC': np.mean(aoc_list),
                    'Std AoC': np.std(aoc_list),
                    'Median AoC': np.median(aoc_list)
                }

                mAoC_by_month[topic] = result_one_topic.groupby('published_year_month')['mean_aoc'].mean()


        # Lowest mAoC comes first
        aoc_df = pd.DataFrame(aggregated_metrics).T[['Mean AoC', 'Std AoC', 'Median AoC']].sort_values(by='Mean AoC',
        ascending=True)

        citation_diversity_df = pd.DataFrame(aggregated_metrics).T[['Simpson', 'Shannon', 'Gini']].sort_values(by='Simpson', ascending=True)


        with pd.ExcelWriter(path) as writer:
            aoc_df.to_excel(writer, sheet_name='AoC')
            citation_diversity_df.to_excel(writer, sheet_name='Diversity')

    # Plot line plots for AoC
    plt.figure(figsize=(10, 6))

    plt.plot(result_one_topic['year_month'].astype(str), result_one_topic['mean_aoc'],
             marker='o')

    # Set the y-axis to log scale
    # plt.yscale('log')

    # Set the title and labels
    plt.title('Mean Time Differences Over Time')
    plt.xlabel('Year-Month')
    plt.ylabel('Mean Time Differences')

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


    age_of_citations_filtered = pd.merge(age_of_citations_filtered.reset_index(), arxiv_data[[
        "arXivId",
                                                                                                        "published"]],
                                left_on="index",
                                right_on="arXivId", how='left')

    # Ensure time_differences are lists of numbers
    age_of_citations_filtered['time_differences'] = age_of_citations_filtered['time_differences'].apply(
        lambda x: pd.Series(x).mean())




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
