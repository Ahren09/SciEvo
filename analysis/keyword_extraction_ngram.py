"""
Extract keywords from the titles / abstracts (summary) using n-gram.
"""
import os
import os.path as osp
import pickle
import sys
import time

import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from model.vectorizer import CustomCountVectorizer, CustomTfidfVectorizer

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))
from arguments import parse_args
from utility.utils_data import load_arXiv_data
from utility.utils_misc import project_setup


def get_vectorizer(args, data=None):
    """
    Get a vectorizer object for extracting n-grams from the text.

    Args:
        args (argparse.Namespace): Arguments.

    Returns:
        vectorizer (sklearn.feature_extraction.text.CountVectorizer): Vectorizer object.
    """
    path = osp.join(args.output_dir, 'tfidf_vectorizer.pkl')

    if osp.exists(path):
        with open(path, 'rb') as f:
            vectorizer = pickle.load(f)
            return vectorizer

    else:

        abstracts = [abs for abs in data['summary'].tolist() if isinstance(abs, str)]

        if args.num_workers == 1:
            # Initialize a CountVectorizer/TfidfVectorizer with ngram_range=(1, 4)

            # vectorizer = CountVectorizer(ngram_range=(1, 4), min_df=5)

            # NOTE: If we use ngram_range=(1, 4), it will be generating n-grams with length 1, 2, 3, 4
            vectorizer = TfidfVectorizer(ngram_range=(1, 4), min_df=5, stop_words='english')

            # Fit and transform the summaries to get feature matrix
            vectorizer.fit_transform(abstracts)

            with open(path, 'wb') as f:
                pickle.dump(vectorizer, f)

        else:
            from model.vectorizer import CustomCountVectorizer
            vectorizer = CustomCountVectorizer(n_range=range(1, 2), args=args)

            dtm = vectorizer.fit_transform(abstracts, range(1, 2), stopwords, num_workers=args.num_workers)

            vectorizer.save()

    return vectorizer


def main():
    df = load_arXiv_data(args.data_dir, subset="last_100" if args.debug else None)

    # vectorizer = get_vectorizer(args, data=df)

    vectorizer = CustomTfidfVectorizer(n_range=range(1, 2), args=args)
    dtm = vectorizer.fit_transform(df.summary.values, range(1, 2), stopwords, num_workers=args.num_workers)
    vectorizer.load()






    # Get feature names (i.e., the n-grams)
    # feature_names = vectorizer.get_feature_names_out()


if __name__ == "__main__":
    project_setup()
    args = parse_args()


    stopwords = set(stopwords.words("english"))
    main()
