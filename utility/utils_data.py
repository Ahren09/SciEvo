
import json
import os
import os.path as osp
import pickle
import re
import time
import datetime
from os import path as osp
from typing import List

import pandas as pd
import pytz
from datasets import Dataset
from tqdm import tqdm

import const


def load_arXiv_data(data_dir: str, start_year: int = None, start_month: int = None, end_year: int
= None, end_month: int = None):
    """
    Load the arXiv metadata.
    Args:
        args: Arguments. We need the data_dir attribute.
        subset: str: Subset of the arXiv metadata to load. If None, the entire dataset is loaded.

    Returns:
        data (pd.DataFrame): DataFrame containing the arXiv metadata.
    """

    t0 = time.time()

    path = osp.join(data_dir, "NLP", "arXiv", "arXiv_metadata.parquet")

    data = pd.read_parquet(path)

    data[const.PUBLISHED] = pd.to_datetime(data[const.PUBLISHED], utc=True)

    if start_year is not None and start_month is not None and end_year is not None and end_month is not None:

        start_time = datetime.datetime(start_year, start_month, 1, tzinfo=pytz.utc)
        end_time = datetime.datetime(end_year, end_month, 1, tzinfo=pytz.utc)

        data = data[(data[const.PUBLISHED] >= start_time) & (data[const.PUBLISHED] < end_time)].reset_index(drop=True)




    for feature_name in ["title", "title_and_abstract"]:
        all_keywords = {}
        if f"{feature_name}_keywords" not in data.columns:
            for year in range(1985, 2025):
                keyword_path = os.path.join(data_dir, "NLP", "arXiv", f"{feature_name}_keywords_{year}.json")
                keywords = json.load(open(keyword_path))
                all_keywords.update(keywords)

            all_keywords = pd.Series(all_keywords).to_frame(f"{feature_name}_keywords").reset_index()
            all_keywords.columns = ["id", f"{feature_name}_keywords"]

            data = pd.merge(data, all_keywords, on='id', how='left')
            data.to_parquet(path)



    print(f"Loaded {len(data)} arXiv papers in {(time.time() - t0):.3f} secs.")

    for column_name in ['title_keywords', 'title_and_abstract_keywords', 'title', 'summary']:
        data[column_name].fillna("", inplace=True)

    # If needed, push to HuggingFace

    # hf_dataset = Dataset.from_pandas(data)
    # hf_dataset.push_to_hub("anonymous/SciEvo")

    return data


def get_embed_path(data_dir: str, feature_name: str, tokenization_mode: str, model_name, start: datetime.datetime,
                   end: datetime.datetime):

    if model_name == "word2vec":
        path = osp.join(data_dir, f"{feature_name}_{tokenization_mode}", model_name,
                        f"word2vec_{start.strftime(const.format_string)}-{end.strftime(const.format_string)}.model")

    elif model_name == "gcn":
        path = osp.join(data_dir, f"{feature_name}_{tokenization_mode}", model_name,
                        f"gcn_{start.year}.pkl")

    else:
        raise ValueError(f"Model name {model_name} not recognized")
    return path

def get_keywords_path(data_dir: str, attribute: str):
    path = osp.join(data_dir, "NLP", "arXiv", f"{attribute}_keywords.json")
    return path

def load_keywords(data_dir: str, attribute: str):
    if attribute == "title":
        num_keywords = 3

    elif attribute == "title_and_abstract":
        num_keywords = 15

    else:
        raise ValueError(f"Attribute {attribute} not recognized")

    path = get_keywords_path(data_dir, attribute)

    with open(path, "r") as f:
        keywords = json.load(f)

    entries = []
    arxiv_data = load_arXiv_data(data_dir)

    for k, v in tqdm(keywords.items(), desc="Adding Keywords"):
        entries.append((k, [keyword.lower().strip() for keyword in v.split(",")]))
    keywords = pd.DataFrame(entries, columns=["id", "keywords"]).set_index("id")
    keywords = keywords.join(arxiv_data[['id', 'published', 'tags_cleaned']].set_index("id"))

    return keywords


def write_pickle(data, filename):
    fp = open(filename, "wb")
    pickle.dump(data, fp)

def load_pickle(filename):
    fp = open(filename, "rb")
    return pickle.load(fp)


def get_arXiv_IDs_of_existing_papers(input):
    if isinstance(input, pd.DataFrame):
        # Input is the DataFrame containing the arXiv metadata
        df = input

    elif isinstance(input, str):
        # Input is the path to the parquet file that stores all arXiv metadata
        df = pd.read_parquet(input)

    else:
        raise ValueError(f"input type {type(input)} not recognized")

    existing_arxiv_ids_truncated = []

    existing_arxiv_ids = df[const.ID].apply(lambda x: x.split('arxiv.org/abs/')[1]).values

    for x in tqdm(existing_arxiv_ids, desc="Processing IDs"):
        if x[4] == '.' and int(x[:4]) >= 704:
            x = x.split('v')[0]
            assert len(x) in [9, 10], f"Error: {x} {len(x)}"

        else:
            pass
        existing_arxiv_ids_truncated.append(x)

    existing_arxiv_ids = existing_arxiv_ids_truncated

    return existing_arxiv_ids


def process_arxiv_entry(entry):
    paper = {}

    for field in ['id', 'title_llm_extracted_keyword', 'summary', 'arxiv_comment',
                  'published', 'updated', ]:
        paper[field] = entry.get(field, None)

    paper['authors'] = [author['name'] for author in
                        entry.get('authors', [])]
    paper['tags'] = [tag['term'] for tag in entry.get('tags', [])]

    return paper


def convert_arxiv_url_to_id(url: str):
    id = url.split("arxiv.org/abs/")[-1]
    if id[-2] == 'v':
        id = id[:-2]

    elif id[-3] == 'v':
        id = id[:-3]
    return id

def load_tag2papers(args):
    tag2papers = json.load(open(osp.join(args.data_dir, "tag2papers.json"), "r"))
    return tag2papers


def get_titles_or_abstracts_as_list(data: pd.DataFrame, column_name: str):
    """
    Process and return the titles or abstracts (`summary`) in the dataset as a list.
    Args:
        data (pd.DataFrame): DataFrame containing the data.
        column_name (str): Column name to extract data from.

    Returns (list): List of titles or abstracts. Each entry is a string.
    """
    assert column_name in ['title_llm_extracted_keyword', 'summary']

    feature_list = data[column_name].tolist()
    feature_list = [re.sub(" +", " ", entry.replace('\n', ' ')) for entry in feature_list]
    return feature_list


def load_semantic_scholar_references_parquet(
        data_dir: str,
        start_year: int,
        end_year: int = None,

) -> pd.DataFrame:
    """Loads Semantic Scholar references from parquet files within the specified year range.

    Args:
        start_year (int): The starting year of the range.
        end_year (int): The ending year of the range.
        data_dir (str): The directory where the parquet files are stored.

    Returns:
        pd.DataFrame: A DataFrame containing all the references from the specified year range.
    """
    if end_year is None:
        end_year = start_year + 1

    current_year = start_year
    all_references: List[pd.DataFrame] = []

    while current_year < end_year:
        print(f"Loading year {current_year}")

        if 1990 <= current_year <= 2004:
            path = osp.join(data_dir, "NLP", "semantic_scholar", "references_1990-2004.parquet")
            current_year = 2004
        elif 2005 <= current_year <= 2010:
            path = osp.join(data_dir, "NLP", "semantic_scholar", "references_2005-2010.parquet")
            current_year = 2010
        elif 2011 <= current_year <= 2015:
            path = osp.join(data_dir, "NLP", "semantic_scholar", "references_2011-2015.parquet")
            current_year = 2015
        else:
            path = osp.join(data_dir, "NLP", "semantic_scholar", f"references_{current_year}.parquet")

        current_year += 1

        references = pd.read_parquet(path)
        all_references.append(references)

    if len(all_references) == 1:
        return all_references[0]

    else:
        all_references_df = pd.concat(all_references, ignore_index=True)
        return all_references_df


def load_semantic_scholar_references_json(year, month, data_dir):
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


def load_semantic_scholar_papers(data_dir: str, start_year: int = None, start_month: int = None, end_year: int = None,
                                 end_month: int = None):
    path = osp.join(data_dir, "NLP", "semantic_scholar", f"semantic_scholar.parquet")

    data = pd.read_parquet(path)
    data = data.rename(columns={'publicationDate': 'semanticScholarPublicationDate'})
    data['arXivPublicationDate'] = pd.to_datetime(data['arXivPublicationDate'], utc=True)

    if start_year is not None and start_month is not None and end_year is not None and end_month is not None:
        start_time = datetime.datetime(start_year, start_month, 1, tzinfo=pytz.utc)
        end_time = datetime.datetime(end_year, end_month, 1, tzinfo=pytz.utc)

        data = data[(data['arXivPublicationDate'] >= start_time) & (data['arXivPublicationDate'] < end_time)].reset_index(drop=True)

    print(f"Loaded {len(data)} entries from Semantic Scholar.")

    data.semanticScholarPublicationDate = pd.to_datetime(data.semanticScholarPublicationDate, utc=True)
    data.arXivPublicationDate = pd.to_datetime(data.arXivPublicationDate, utc=True) # Some of these are null

    return data
