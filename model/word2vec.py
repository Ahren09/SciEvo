import datetime
import os
import os.path as osp
import re
import sys
from typing import Set

import nltk
import numpy as np
import pandas as pd
import pytz
import scipy.sparse as sp
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from tqdm import tqdm

from model.vectorizer import CustomCountVectorizer
from utility.utils_data import load_arXiv_data
from utility.utils_misc import project_setup

sys.path.append(osp.join(os.getcwd(), "src"))
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
    print(f"Excluded words: {excluded_words}")
    retained_indices = build_retained_indices(vectorizer.vocabulary_d[N], excluded_words)

    print("Update vocabulary")

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


def update_vectorizer_naive_implementation(vectorizer: CustomCountVectorizer, N: int = 1) -> CustomCountVectorizer:
    updated_vocabulary = {}
    old_vocab = list(vectorizer.vocabulary_d[N].items())

    retained_1gram_indices = []

    for i, (k, v) in enumerate(tqdm(vectorizer.vocabulary_d[N].items())):
        if k not in set(ALL_EXCLUDED_WORDS):
            try:
                # The word should NOT be convertible to numbers
                int(k)

            except:
                # print(k)
                retained_1gram_indices += [i]

    for i in tqdm(retained_1gram_indices, "New Vocab"):
        k, v = old_vocab[i]
        updated_vocabulary[k] = len(updated_vocabulary)

    words = set(updated_vocabulary.keys())

    n_grams_d = {i: [] for i in range(1, 5)}

    for example in tqdm(vectorizer.n_grams_d[N], desc="Generate ngrams"):
        ngrams = []
        for ngram in example:
            if ngram in words:
                ngrams += [ngram]
        n_grams_d[N] += [ngrams]

    updated_vectorizer = CustomCountVectorizer(n_range=range(1, 2), args=args)
    updated_vectorizer.dtm_d[N] = vectorizer.dtm_d[N][:, retained_1gram_indices]
    updated_vectorizer.vocabulary_d[N] = updated_vocabulary
    updated_vectorizer.n_grams_d[N] = n_grams_d[N]

    return updated_vectorizer


def main():
    print("Loading data...", end='\r')
    df = load_arXiv_data(args, subset="last_10000")
    print("Done!")

    df['published'] = pd.to_datetime(df['published'], utc=True)

    N = 1

    vectorizer = CustomCountVectorizer(n_range=range(N, N + 1), args=args)

    vectorizer.load()

    data = load_arXiv_data(args)
    mask_valid_abstract = np.array([True if isinstance(abs, str) else False for abs in data['summary']])
    vectorizer = update_vectorizer_naive_implementation(vectorizer, N=N)
    data.sort_values('published', ascending=True, inplace=True)

    total = 0
    total_entries = 0
    format_string = "%Y-%m-%d"

    print(sum(mask_valid_abstract))

    graphs_li = []

    for start_year in range(1991, 2024):

        for start_month in range(1, 13):

            # Treat all papers before 1990 as one single snapshot
            if start_year < 1992:
                start = datetime.datetime(1970, 1, 1, 0, 0, 0, tzinfo=pytz.utc)

                end = datetime.datetime(1992, 1, 1, 0, 0, 0, tzinfo=pytz.utc)

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

            mask_time = ((data['published'] >= start) & (data['published'] < end)).values

            # mask: (1984440, )
            mask = mask_time & mask_valid_abstract
            total += mask.sum()
            total_entries += 1
            data_snapshot = data[mask]

            # (N, V), N = number of documents, V = number of words
            # vectorizer.dtm_d[N] is a CSR matrix, efficient for row slicing / operations
            coincidence_matrix = vectorizer.dtm_d[N][mask_time[mask_valid_abstract]]
            print(
                f"Generating embeds from {start.strftime(format_string)} to {end.strftime(format_string)}: {len(data_snapshot)} documents, total {total}, coincidence matrix {coincidence_matrix.shape}")

            if DO_WORD2VEC:
                abstracts = data_snapshot['summary'].tolist()

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
                    filename = f"word2vec_{start.strftime(format_string)}-{end.strftime(format_string)}.model"
                    print(f"Saving model to {filename}")
                    model.save(osp.join(model_path, filename))


            elif DO_GNN:
                # V = number of words

                # (V, V)
                co_occurrence_matrix = coincidence_matrix.T.dot(coincidence_matrix)

                # Remove the diagonal entries (i.e., word co-occurrence with itself)
                co_occurrence_matrix.setdiag(0)

                # Eliminate zero entries to maintain sparse structure
                co_occurrence_matrix.eliminate_zeros()

                sp.save_npz(osp.join(f'graph_{total_entries}_{start.strftime(format_string)}_'
                                     f'{end.strftime(format_string)}.npz'),
                            co_occurrence_matrix)

            if start_year < 1992:
                break

    print(f"Total entries: {total_entries}")


if __name__ == "__main__":
    DO_WORD2VEC = False
    DO_GNN = True

    project_setup()

    args = parse_args()
    model_path = osp.join("checkpoints", "word2vec")
    os.makedirs(model_path, exist_ok=True)
    main()
