import datetime
import json
import os.path as osp
import pickle
import re
import time

import pandas as pd
import pytz
from tqdm import tqdm

import const


def load_arXiv_data(data_dir: str, subset: str = None, start_year: int = None, start_month: int = None, end_year: int
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
    if subset is None:
        # Load the full dataset
        data = pd.read_parquet(osp.join(data_dir, "NLP", "arXiv", "arXiv_metadata.parquet"))

        data[const.PUBLISHED] = pd.to_datetime(data[const.PUBLISHED], utc=True, format='mixed')

        if start_year is not None and start_month is not None and end_year is not None and end_month is not None:

            start_time = datetime.datetime(start_year, start_month, 1, tzinfo=pytz.utc)
            end_time = datetime.datetime(end_year, end_month, 1, tzinfo=pytz.utc)

            data = data[(data[const.PUBLISHED] >= start_time) & (data[const.PUBLISHED] < end_time)].reset_index(drop=True)




    else:
        if subset == "first_100":
            path = osp.join(data_dir, "arXiv_metadata_first_100_entries.xlsx")

        elif subset == "last_100":
            path = osp.join(data_dir, "arXiv_metadata_last_100_entries.xlsx")

        else:
            raise ValueError(f"subset {subset} not recognized")

        data = pd.read_excel(path)

    print(f"Loaded {len(data)} entries in {(time.time() - t0):.3f} secs.")

    data['title'].fillna("", inplace=True)
    data['summary'].fillna("", inplace=True)

    return data


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

    for field in ['id', 'title', 'summary', 'arxiv_comment',
                  'published', 'updated', ]:
        paper[field] = entry.get(field, None)

    paper['authors'] = [author['name'] for author in
                        entry.get('authors', [])]
    paper['tags'] = [tag['term'] for tag in entry.get('tags', [])]

    return paper


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
    assert column_name in ['title', 'summary']

    feature_list = data[column_name].tolist()
    feature_list = [re.sub(" +", " ", entry.replace('\n', ' ')) for entry in feature_list]
    return feature_list
