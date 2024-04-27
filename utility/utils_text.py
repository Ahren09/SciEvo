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
    df = pd.read_parquet(osp.join(data_dir, "NLP", "semantic_scholar",
                                                     f"semantic_scholar.parquet"))

    start_time = pd.Timestamp(year=start_year, month=start_month, day=1, tz='utc')
    end_time = pd.Timestamp(year=end_year, month=end_month, day=1, tz='utc')
    df = df[(df.arXivPublicationDate >= start_time) & (df.arXivPublicationDate < end_time)].reset_index(drop=True)


    print(f"Loaded {len(df)} entries from Semantic Scholar.")

    df.publicationDate = pd.to_datetime(df.publicationDate, utc=True)
    df.arXivPublicationDate = pd.to_datetime(df.arXivPublicationDate, utc=True)

    return df