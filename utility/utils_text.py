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

