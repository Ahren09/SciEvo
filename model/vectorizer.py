import itertools
import os
import string
import sys
import os.path as osp
import pickle
import time
import numpy as np
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer

sys.path.insert(0, os.path.abspath('..'))


def preprocess_text(text):
    """
    Lowercases the text.
    """
    return text.lower()


def should_keep_ngram(n_gram, stop_set):
    """
    Checks if all words in an n-gram are stop words.
    """
    return not all(word in stop_set for word in n_gram.split())


def extract_ngrams_from_text(text, n, stop_set):
    """
    Extracts n-grams from a single document.
    """
    tokens = text.split()
    tokens = [t.strip(string.punctuation) for t in tokens]

    n_grams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(n_gram) for n_gram in n_grams if should_keep_ngram(" ".join(n_gram), stop_set)]


def extract_ngrams_parallel(documents, n_range, stop_set, num_workers=4):
    """
    Parallelizes the extraction of n-grams from multiple documents.

    Args:
        documents (list): List of documents, e.g. abstracts.
        n_range (range): Range of n-gram lengths to extract. We usually use n_range=range(1, 5). for 1-gram to 4-gram
        stop_set (set): Set of stop words.


    """
    n_grams_d = {}
    
    for n in n_range:
        futures = []
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            print(f"Extracting {n}-grams with {num_workers} workers...")
            for doc in documents:
                futures.append(executor.submit(extract_ngrams_from_text, preprocess_text(doc), n, stop_set))
            
            n_grams_for_n = []
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Extracting {n}-grams"):
                n_grams_for_n.append(future.result())

        n_grams_d[n] = n_grams_for_n


    # Combine the n-grams of different lengths for each document
    return n_grams_d


def filter_ngrams_by_min_df(n_grams_d, n_range=range(1, 5), min_df=5):
    """
    Filters out n-grams that appear less than `min_df` times across all documents.
    """

    filtered_n_grams_d = {}
    for n in n_range:
        global_counter = Counter(itertools.chain.from_iterable(n_grams_d[n]))
        filtered_n_grams_d[n] = [[n_gram for n_gram in doc if global_counter[n_gram] >= min_df] for doc in n_grams_d[n]]

        print(f"#{n}grams {len(filtered_n_grams_d[n])}")

    return filtered_n_grams_d

def build_count_vectorizer_matrix(n_grams_d):
    """
    Constructs a CountVectorizer-like matrix from the list of n-grams for each document.
    """
    counters = [Counter(doc_n_grams) for doc_n_grams in n_grams_d]
    vocabulary = {}
    for counter in counters:
        for n_gram in counter.keys():
            if n_gram not in vocabulary:
                vocabulary[n_gram] = len(vocabulary)

    num_docs = len(counters)
    num_terms = len(vocabulary)
    indptr = [0]
    indices = []
    data = []
    for counter in counters:
        for term, freq in counter.items():
            index = vocabulary[term]
            indices.append(index)
            data.append(freq)
        indptr.append(len(indices))

    return sp.csr_matrix((data, indices, indptr), shape=(num_docs, num_terms), dtype=np.int32)


class BaseVectorizer:
    def __init__(self, n_range, args):
        self.n_grams_d = {}
        self.dtm_d = {}
        self.vocabulary_d = {}
        self.n_range = n_range
        self.args = args

    def build_dtm(self, n_grams_d):

        for n in self.n_range:
            counters = [Counter(doc_n_grams) for doc_n_grams in n_grams_d[n]]
            vocabulary = {}
            for counter in counters:
                for n_gram in counter.keys():
                    if n_gram not in vocabulary:
                        vocabulary[n_gram] = len(vocabulary)

            num_docs = len(counters)
            num_terms = len(vocabulary)
            indptr = [0]
            indices = []
            data = []
            for counter in counters:
                for term, freq in counter.items():
                    index = vocabulary[term]
                    indices.append(index)
                    data.append(freq)
                indptr.append(len(indices))

            self.dtm_d[n] = sp.csr_matrix((data, indices, indptr), shape=(num_docs, num_terms), dtype=int)

            self.vocabulary_d[n] = vocabulary

    def save(self, path=None):

        path = osp.join(self.args.output_dir, self.__class__.__name__) if path is None else path
        print(f"Saving vectorizer to {path}...")
        os.makedirs(path, exist_ok=True)

        for n in self.n_range:
            print(f"Saving {n}-gram...")
            sp.save_npz(osp.join(path, f'{n}gram_csr_matrix.npz'), self.dtm_d[n])
            with open(osp.join(path, f"{n}gram_d.pkl"), 'wb') as f:
                pickle.dump(self.n_grams_d[n], f)

            with open(osp.join(path, f"{n}gram_vocab.pkl"), 'wb') as f:
                pickle.dump(self.vocabulary_d[n], f)

        print("Done!")

    def load(self, path=None):
        path = osp.join(self.args.output_dir, self.__class__.__name__) if path is None else path
        print(f"Loading vectorizer from {path}...")
        for n in self.n_range:
            print(f"Loading {n}-gram...")
            self.dtm_d[n] = sp.load_npz(osp.join(path, f'{n}gram_csr_matrix.npz'))
            with open(osp.join(path, f"{n}gram_d.pkl"), 'rb') as f:
                self.n_grams_d[n] = pickle.load(f)

            with open(osp.join(path, f"{n}gram_vocab.pkl"), 'rb') as f:
                self.vocabulary_d[n] = pickle.load(f)

        print("Done!")


class CustomCountVectorizer(BaseVectorizer):
    def fit_transform(self, documents, n_range, stop_set, num_workers=4):
        t0 = time.time()
        self.n_grams_d = extract_ngrams_parallel(documents, n_range, stop_set, num_workers)
        self.n_grams_d = filter_ngrams_by_min_df(self.n_grams_d, self.n_range, 5)
        self.build_dtm(self.n_grams_d)
        print(f"Fit and transform takes {time.time() - t0:.2f} secs")
        return self.dtm_d


class CustomTfidfVectorizer(BaseVectorizer):
    def fit_transform(self, documents, n_range, stop_set, num_workers=4):
        t0 = time.time()
        self.n_grams_d = extract_ngrams_parallel(documents, n_range, stop_set, num_workers)
        self.n_grams_d = filter_ngrams_by_min_df(self.n_grams_d, self.n_range, 5)
        self.build_dtm(self.n_grams_d)

        # Apply TF-IDF transformation
        transformer = TfidfTransformer()

        print("TODO: Fix this")
        raise NotImplementedError
        self.dtm_d = transformer.fit_transform(self.dtm_d)

        print(f"Fit and transform takes {time.time() - t0:.2f} secs")

        return self.dtm
