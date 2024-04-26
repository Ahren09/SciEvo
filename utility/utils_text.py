import re

import pandas as pd
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
import os.path as osp

english_stopwords = stopwords.words("english") + ['']

    # Initialize Stopwords
stopwords_set = set(stopwords.words('english'))


def split_text_into_tokens(text):
    # Remove newlines and extra spaces
    text = text.lower().replace('\n', ' ').strip()
    text = re.sub(' +', ' ', text)

    # Lowercase and remove punctuation
    tokenizer = RegexpTokenizer(r'[\$\[\]\{\}\w\\\-_]+')
    tokens = tokenizer.tokenize(text.lower())
    tokens = [token.strip('.?!,') for token in tokens if token not in stopwords_set]
    return tokens


def load_semantic_scholar_data_one_month(data_dir, year: int, month: int):
    return pd.read_json(
        osp.join(data_dir, "NLP", "semantic_scholar", "semantic_scholar_2023_3.json"), orient='index')


def load_semantic_scholar_data(data_dir, start_year: int, start_month: int, end_year: int, end_month: int):
    df_li = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):

            # End month will not be included
            if (month < start_month and year <= start_year) or (month >= end_month and year >= end_year):
                continue

            print(f"Loading Year {year} Month {month}")

            df = load_semantic_scholar_data_one_month(data_dir, year, month)

            df_li += [df]

    df = pd.concat(df_li, ignore_index=True)
    df.publicationDate = pd.to_datetime(df.publicationDate, utc=True)

    return df