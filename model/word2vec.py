import os
import os.path as osp
import re
import sys
import datetime
from pathlib import Path
from typing import Set

import nltk
import pandas as pd
import pytz
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from tqdm import tqdm

from model.vectorizer import CustomCountVectorizer
from utility.utils_data import load_data
from utility.utils_misc import project_setup

sys.path.append(osp.join(os.getcwd(), "src"))
import const
from utility.wordbank import ALL_EXCLUDED_WORDS

from arguments import parse_args

# Ignore FutureWarnings
pd.options.mode.chained_assignment = None


def tokenize_document(documents: list, stop_words: set) -> list:
    """
    Tokenize the documents and remove stop words.

    Args:
    documents (list): List of document strings.
    stop_words (set): Set of stop words to exclude.

    Returns:
    list: List of tokenized and filtered documents.
    """
    tokenized_docs = []
    for doc in documents:
        doc = re.sub(r'[~/\n]', ' ', doc)
        doc = re.sub(r'(?<=\w)\.(?=\w)', '. ', doc)
        tokens = nltk.word_tokenize(doc.lower())
        filtered_tokens = [token for token in tokens if token not in stop_words]
        tokenized_docs.append(filtered_tokens)
    return tokenized_docs

def process_documents(documents: list) -> list:
    """
    Preprocesses a list of documents.

    Args:
        documents (list): List of document strings.

    Returns:
        tokenized_documents (list): List of tokenized and filtered documents.
    """
    stop_words = set(stopwords.words('english'))
    return tokenize_document(documents, stop_words)


def build_retained_indices(vocabulary: dict, excluded_words: Set[str]) -> list:
    return [
        i for i, (word, _) in enumerate(vocabulary.items())
        if word not in excluded_words and not word.isnumeric()
    ]


def build_updated_vocabulary(old_vocab: list, retained_indices: list) -> dict:
    return {k: i for i, (k, v) in enumerate(old_vocab) if i in retained_indices}


def update_ngrams(n_grams: list, words: set) -> list:
    return [[ngram for ngram in example if ngram in words] for example in n_grams]


def update_vectorizer(vectorizer: CustomCountVectorizer, N: int = 1) -> CustomCountVectorizer:
    excluded_words = set(ALL_EXCLUDED_WORDS)
    retained_indices = build_retained_indices(vectorizer.vocabulary_d[N], excluded_words)

    updated_vocabulary = build_updated_vocabulary(
        list(vectorizer.vocabulary_d[N].items()), retained_indices
    )

    updated_n_grams = update_ngrams(vectorizer.n_grams_d[N], set(updated_vocabulary.keys()))

    # TODO
    updated_vectorizer = CustomCountVectorizer(n_range=range(N, N + 1), args=args)
    updated_vectorizer.dtm_d[N] = vectorizer.dtm_d[N][:, retained_indices]
    updated_vectorizer.vocabulary_d[N] = updated_vocabulary
    updated_vectorizer.n_grams_d[N] = updated_n_grams

    return updated_vectorizer

# def update_vectorizer(vectorizer: CustomCountVectorizer, N: int  = 1) -> CustomCountVectorizer:
#     updated_vocabulary = {}
#     old_vocab = list(vectorizer.vocabulary_d[N].items())
#
#     retained_1gram_indices = []
#
#     def build_updated_vocabulary(old_vocab: list, retained_indices: list) -> dict:
#         return {k: i for i, (k, v) in enumerate(old_vocab) if i in retained_indices}
#
#     for i, (k, v) in enumerate(tqdm(vectorizer.vocabulary_d[1].items())):
#         if k not in set(ALL_EXCLUDED_WORDS):
#             try:
#                 # The word should NOT be convertible to numbers
#                 int(k)
#
#             except:
#                 # print(k)
#                 retained_1gram_indices += [i]
#
#     for i in tqdm(retained_1gram_indices, "New Vocab"):
#         k, v = old_vocab[i]
#         updated_vocabulary[k] = len(updated_vocabulary)
#
#     words = set(updated_vocabulary.keys())
#
#     n_grams_d = {i: [] for i in range(1, 5)}
#
#     for example in tqdm(vectorizer.n_grams_d[N], desc="Generate ngrams"):
#         ngrams = []
#         for ngram in example:
#             if ngram in words:
#                 ngrams += [ngram]
#         n_grams_d[N] += [ngrams]
#
#     updated_vectorizer = CustomCountVectorizer(n_range=range(1, 2), args=args)
#     updated_vectorizer.dtm_d[N] = vectorizer.dtm_d[N][:, retained_1gram_indices]
#     updated_vectorizer.vocabulary_d[N] = updated_vocabulary
#     updated_vectorizer.n_grams_d[N] = n_grams_d[N]
#
#     return updated_vectorizer


def main():

    print("Loading data...", end='\r')
    df = load_data(args, subset="last_10000")
    print("Done!")

    df['published'] = pd.to_datetime(df['published'])

    vectorizer = CustomCountVectorizer(n_range=range(1, 3), args=args)

    vectorizer.load()

    data = load_data(args)
    vectorizer = update_vectorizer(vectorizer, N=1)
    data.sort_values('published', ascending=True, inplace=True)

    for start_year in range(1991, 2024):

        for start_month in range(1, 13):

            # Treat all papers before 1990 as one single snapshot
            if start_year < 1990:
                start = datetime.datetime(1970, 1, 1, 0, 0, 0, tzinfo=pytz.utc)

                end = datetime.datetime(1990, 1, 1, 0, 0, 0, tzinfo=pytz.utc)
                break

            elif start_year == 2023 and start_month == 11:
                break

            else:
                if start_month == 12:
                    # Turn to January next year
                    end = datetime.datetime(start_year + 1, 1, 1, 0, 0, 0, tzinfo=pytz.utc)

                else:

                    # Turn to the next month in the same year
                    end = datetime.datetime(start_year, start_month + 1, 1, 0, 0, 0, tzinfo=pytz.utc)

                start = datetime.datetime(start_year, start_month, 1, 0, 0, 0, tzinfo=pytz.utc)



            mask = (data['published'] >= start) & (data['published'] < end)
            df_snapshot = data[mask]

            print(f"Generating embeds from {start} to {end}: {len(df_snapshot)} documents")


            abstracts = df_snapshot['summary'].tolist()
            print(f"Generating embeds from {start_date} to {end_date}: {len(abstracts)} documents")

            tokenized_docs = process_documents(abstracts)



            try:
                model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=4, min_count=3,
                                 sg=1, negative=5, epochs=50, workers=15, seed=42)
            except:
                model = Word2Vec(sentences=tokenized_docs, size=100, window=4,
                                 min_count=3, sg=1, negative=5, iter=50, workers=15, seed=42)


            similar_words = model.wv.most_similar('computer', topn=20)
            print(f"({start}-{end}) Words most similar to 'computer':")
            for word, similarity in similar_words:
                print(f"\t{word}: {similarity:.4f}")


            if args.save_model:
                filename = f"word2vec_{start_date}-{end_date}.model"
                print(f"Saving model to {filename}")
                model.save(osp.join(model_path, filename))

if __name__ == "__main__":
    project_setup()

    args = parse_args()
    model_path = osp.join("checkpoints", "word2vec")
    os.makedirs(model_path, exist_ok=True)
    main()
