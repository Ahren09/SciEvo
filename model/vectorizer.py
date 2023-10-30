import itertools
import os
import os.path as osp
import pickle
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer


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
    n_grams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(n_gram) for n_gram in n_grams if should_keep_ngram(" ".join(n_gram), stop_set)]


def extract_ngrams_parallel(documents, n_range, stop_set, num_workers=4):
    """
    Parallelizes the extraction of n-grams from multiple documents.
    """
    n_grams_list = []
    for n in n_range:
        # Use ProcessPoolExecutor instead of ThreadPoolExecutor for multi-processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            n_grams_for_n = list(
                executor.map(lambda doc: extract_ngrams_from_text(preprocess_text(doc), n, stop_set), documents))
        n_grams_list.append(n_grams_for_n)

    # Combine the n-grams of different lengths for each document
    combined_n_grams_list = [list(itertools.chain.from_iterable(doc_n_grams)) for doc_n_grams in zip(*n_grams_list)]
    return combined_n_grams_list


def filter_ngrams_by_min_df(n_grams_list, min_df=5):
    """
    Filters out n-grams that appear less than `min_df` times across all documents.
    """
    global_counter = Counter(itertools.chain.from_iterable(n_grams_list))
    return [[n_gram for n_gram in doc if global_counter[n_gram] >= min_df] for doc in n_grams_list]


def build_count_vectorizer_matrix(n_grams_list):
    """
    Constructs a CountVectorizer-like matrix from the list of n-grams for each document.
    """
    counters = [Counter(doc_n_grams) for doc_n_grams in n_grams_list]
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

    return sp.csr_matrix((data, indices, indptr), shape=(num_docs, num_terms), dtype=int)


class BaseVectorizer:
    def __init__(self, args):
        self.n_grams_list = None
        self.dtm = None
        self.args = args

    def build_dtm(self, n_grams_list):
        counters = [Counter(doc_n_grams) for doc_n_grams in n_grams_list]
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

        self.dtm = sp.csr_matrix((data, indices, indptr), shape=(num_docs, num_terms), dtype=int)

    def save(self, path=None):

        path = osp.join(self.args.output_dir, self.__class__.__name__) if path is None else path
        print(f"Saving vectorizer to {path}...")
        os.makedirs(path, exist_ok=True)

        sp.save_npz(osp.join(path, 'csr_matrix.npz'), self.dtm)
        with open(osp.join(path, "n_grams_list.pkl"), 'wb') as f:
            pickle.dump(self.n_grams_list, f)

    def load(self, path):
        path = osp.join(self.args.output_dir, self.__class__.__name__) if path is None else path
        print(f"Loading vectorizer from {path}...")
        self.dtm = sp.load_npz(osp.join(path, 'csr_matrix.npz'))
        with open(osp.join(path, "n_grams_list.pkl"), 'rb') as f:
            self.n_grams_list = pickle.load(f)


class CustomCountVectorizer(BaseVectorizer):
    def fit_transform(self, documents, n_range, stop_set, num_workers=4):
        t0 = time.time()
        self.n_grams_list = extract_ngrams_parallel(documents, n_range, stop_set, num_workers)
        self.n_grams_list = filter_ngrams_by_min_df(self.n_grams_list, 5)
        self.build_dtm(self.n_grams_list)
        print(f"Fit and transform takes {time.time() - t0:.2f} secs")
        return self.dtm


class CustomTfidfVectorizer(BaseVectorizer):
    def fit_transform(self, documents, n_range, stop_set, num_workers=4):
        t0 = time.time()
        self.n_grams_list = extract_ngrams_parallel(documents, n_range, stop_set, num_workers)
        self.n_grams_list = filter_ngrams_by_min_df(self.n_grams_list, 5)
        self.build_dtm(self.n_grams_list)

        # Apply TF-IDF transformation
        transformer = TfidfTransformer()
        self.dtm = transformer.fit_transform(self.dtm)

        print(f"Fit and transform takes {time.time() - t0:.2f} secs")

        return self.dtm
