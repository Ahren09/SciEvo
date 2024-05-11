import datetime
import json
import os
import os.path as osp
import pickle
import re
import sys
from copy import deepcopy
from typing import Set

import nltk
import numpy as np
import pandas as pd
import pytz
import scipy.sparse as sp
from gensim.models import Word2Vec, KeyedVectors
from nltk.corpus import stopwords
from tqdm import tqdm

import const
from utility.utils_time import TimeIterator

sys.path.append(osp.join(os.getcwd(), "src"))

from model.vectorizer import CustomCountVectorizer
from utility.utils_data import load_arXiv_data
from utility.utils_misc import project_setup

# from utility.wordbank import ALL_EXCLUDED_WORDS

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

    data = load_arXiv_data(args.data_dir, start_year=1990, start_month=1, end_year=2024, end_month=4)
    print("Done!")

    total = 0
    total_entries = 0

    # TODO
    iterator = TimeIterator(2021, 2024, start_month=6, end_month=3, snapshot_type='monthly')

    for (start, end) in iterator:

        mask = ((data['published'] >= start) & (data['published'] < end)).values
        total += mask.sum()
        total_entries += 1

        data_snapshot = data[mask]

        docs = data_snapshot[args.feature_name].tolist()
        tokenized_docs = process_documents(docs)


        # (N, V), N = number of documents, V = number of words
        # vectorizer.dtm_d[N] is a CSR matrix, efficient for row slicing / operations
        print(
            f"Generating embeds from {start.strftime(const.format_string)} to {end.strftime(const.format_string)}: {len(tokenized_docs)} documents, total {total}")

        if DO_WORD2VEC:

            try:
                model = Word2Vec(sentences=tokenized_docs, vector_size=args.embed_dim, window=4, min_count=3,
                                 sg=1, negative=5, epochs=50, workers=args.num_workers, seed=42)
            except:
                model = Word2Vec(sentences=tokenized_docs, size=args.embed_dim, window=4,
                                 min_count=3, sg=1, negative=5, iter=50, workers=15, seed=42)

            similar_words = model.wv.most_similar('computer', topn=20)
            print(f"({start}-{end}) Words most similar to 'computer':")
            for word, similarity in similar_words:
                print(f"\t{word}: {similarity:.4f}")

            if args.save_model:
                filename = f"word2vec_{start.strftime(const.format_string)}-{end.strftime(const.format_string)}.model"
                print(f"Saving model to {filename}")
                model.save(osp.join(model_path, filename))


            elif DO_GNN:
                # Archived

                # V = number of words

                # (V, V)
                co_occurrence_matrix = coincidence_matrix.T.dot(coincidence_matrix)

                # Remove the diagonal entries (i.e., word co-occurrence with itself)
                co_occurrence_matrix.setdiag(0)

                # Eliminate zero entries to maintain sparse structure
                co_occurrence_matrix.eliminate_zeros()

                sp.save_npz(osp.join(f'graph_{total_entries}_{start.strftime(const.format_string)}_'
                                     f'{end.strftime(const.format_string)}.npz'),
                            co_occurrence_matrix)


    print(f"Total entries: {total_entries}")


if __name__ == "__main__":
    DO_WORD2VEC = True
    DO_GNN = False

    const.format_string = "%Y-%m-%d"

    project_setup()
    args = parse_args()
    model_path = osp.join("checkpoints", args.feature_name, "word2vec")
    os.makedirs(model_path, exist_ok=True)
    main()


    # Load each model and construct a combined vocab


    all_shared_idx_word_li = []

    shared_wi, shared_iw = None, None

    all_embeds = []

    for start_year in range(1994, 2025):

        # for start_month in range(1, 13):
        start_month = 1

        # Treat all papers before 1990 as one single snapshot
        if start_year == 1994:
            start = datetime.datetime(1970, 1, 1, 0, 0, 0, tzinfo=pytz.utc)

            end = datetime.datetime(1995, 1, 1, 0, 0, 0, tzinfo=pytz.utc)


        else:

            """
            # For monthly snapshots
            if start_month == 12:
                # Turn to January next year
                end = datetime.datetime(start_year + 1, 1, 1, 0, 0, 0, tzinfo=pytz.utc)

            else:

                # Turn to the next month in the same year
                end = datetime.datetime(start_year, start_month + 1, 1, 0, 0, 0, tzinfo=pytz.utc)

            """

            end = datetime.datetime(start_year + 1, start_month, 1, 0, 0, 0, tzinfo=pytz.utc)

            start = datetime.datetime(start_year, start_month, 1, 0, 0, 0, tzinfo=pytz.utc)




        filename = f"word2vec_{start.strftime(const.format_string)}-{end.strftime(const.format_string)}.model"

        print(f"Loading model from {filename} ...", end='\r')

        model = Word2Vec.load(osp.join(model_path, filename))

        # Create iw and wi for the current model
        iw = list(model.wv.index_to_key)
        wi = {word: idx for idx, word in enumerate(iw)}

        if shared_iw is None and shared_wi is None:
            # First snapshot
            # Update shared_wi and shared_iw
            shared_iw = deepcopy(iw)
            shared_wi = deepcopy(wi)
            shared_idx_word_li = list(range(len(iw)))

        else:
            # The rest of the snapshots

            shared_idx_word_li = []

            for idx_word, word in enumerate(iw):
                if word not in shared_wi:
                    shared_wi[word] = len(shared_wi)
                    shared_iw.append(word)

                else:
                    pass

                shared_idx_word = shared_wi[word]

                shared_idx_word_li += [shared_idx_word]

                print(f"{word}\t{shared_idx_word}")



        print(f"Loading model from {filename} Done!")
        all_shared_idx_word_li += [np.array(shared_idx_word_li)]

        all_embeds += [model.wv.vectors]

    os.makedirs(osp.join(args.checkpoint_dir, const.WORD2VEC), exist_ok=True)

    with open(osp.join(args.checkpoint_dir, const.WORD2VEC, 'shared_vocab.pkl'), 'wb') as f:
        pickle.dump({
            "wi": shared_wi,
            "iw": shared_iw
        }, f)

    for i in range(len(all_embeds)):
        print(f"Embedding {i}: {all_embeds[i].shape}")

        # Create a memory-mapped file with zero initialization
        memmap_path = osp.join(args.checkpoint_dir, const.WORD2VEC, f'data_{i}.memmap')
        memmap_array = np.memmap(memmap_path, dtype=np.float32, mode='w+', shape=(len(shared_wi), args.embed_dim))
        memmap_array[:] = 0  # Initialize the array with zeros

        memmap_array[all_shared_idx_word_li[i]] = all_embeds[i]
















