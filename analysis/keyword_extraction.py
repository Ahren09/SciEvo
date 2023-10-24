"""
This script performs keyword extraction from the arXiv dataset.

Requirements: `conda install -c conda-forge rake_nltk`
"""

import os
import os.path as osp
import re
import string
import sys
import warnings
from multiprocessing import Pool

import pandas as pd
import spacy
from nltk.corpus import stopwords
from rake_nltk import Rake
from tqdm import tqdm

sys.path.insert(0, osp.join(os.getcwd(), "src"))

import const
from arguments import args
from utility.utils_misc import project_setup

# Suppress FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)


def custom_tokenizer(text):
    """
    Custom tokenizer that processes the text and returns tokens.

    Args:
        text (str): Text to tokenize.

    Returns:
        list: A list of tokens.
    """
    return re.findall(r'\b\w[\w-]*\w\b|\b\w\b', text.lower())


class CustomRake(Rake):
    """
    Custom Rake class to facilitate keyword extraction.
    """

    def __init__(self, min_length=1, max_length=3):
        super().__init__(min_length=min_length, max_length=max_length,
                         word_tokenizer=custom_tokenizer)
        self._regex_pattern = re.compile(r"[\w-]+")


def extract_keywords(text):
    """
    Extract keywords from a given text using NLP and other methods.

    Args:
        text (str): Text to extract keywords from.

    Returns:
        tuple: A tuple containing keywords and unigrams.
    """
    text = text.replace('\n', ' ').strip()
    text = re.sub(' +', ' ', text)

    keywords = []
    for text_segment in text.split(":"):
        doc = nlp(text_segment)
        keywords += [chunk.text.strip(string.punctuation + " ") for chunk in doc.noun_chunks]

    unigrams = [token.text.strip(string.punctuation + " ") for token in doc
                if token.pos_ in ('VERB', 'NOUN', 'PROPN', 'ADJ')]

    keywords = list(set(keywords) - stopwords_set)
    unigrams = list(set(unigrams) - stopwords_set)

    return [w.lower() for w in keywords], [w.lower() for w in unigrams]


def process_rows(data):
    """
    Multiprocess rows of a DataFrame for keyword extraction.

    Args:
        data (pd.DataFrame): DataFrame to process.

    Returns:
        list: List of dictionaries containing extracted keywords.
    """
    with Pool(NUM_PROCESS) as pool:
        df_list = list(pool.imap(extract_keywords_for_row, data.iterrows(), chunksize=1))
    return df_list


def extract_keywords_for_row(tup):
    """
    Extract keywords for a given row in DataFrame.

    Args:
        tup (tuple): Tuple containing index and row data.

    Returns:
        dict: Dictionary with extracted keywords and other metadata.
    """
    idx_row, row = tup
    d = {}

    for col_name, col_data in {
        "title": ["title_keywords", "title_unigrams"],
        "summary": ["abstract_keywords", "abstract_unigrams"]
    }.items():
        keywords, unigrams = extract_keywords(row[col_name])

        d[col_data[0]] = keywords
        d[col_data[1]] = unigrams
        d[const.UPDATED] = row[const.UPDATED]
        d[const.ID] = row[const.ID]

    return d


if __name__ == "__main__":

    project_setup()

    # Load Spacy model
    nlp = spacy.load("en_core_web_md")

    # Initialize Stopwords
    stopwords_set = set(stopwords.words('english'))

    # Constants
    NUM_PROCESS = 15

    data = pd.read_pickle(osp.join(args.data_dir, "arXiv_metadata.pkl"))

    data['title'].fillna("", inplace=True)
    data['summary'].fillna("", inplace=True)

    if NUM_PROCESS == 1:
        d_list = [extract_keywords_for_row((idx_row, row)) for idx_row, row in
                  tqdm(data.iterrows(), desc="Extract keywords", total=len(data))]
    else:
        d_list = process_rows(data)

    keywords_df = pd.DataFrame(d_list)
    keywords_df['updated_datetime'] = pd.to_datetime(keywords_df[const.UPDATED])
    keywords_df = keywords_df.sort_values(by=['updated_datetime'], ascending=False)
    keywords_df.drop(columns=['updated_datetime'], inplace=True)
    keywords_df.to_pickle(osp.join(args.data_dir, "Sample_keywords.pkl"))

    print("Done!")
