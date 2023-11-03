"""
Extract keywords from the titles / abstracts (summary) using n-gram.
"""
import os
import os.path as osp
import pickle
import sys
import time

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))
from arguments import parse_args
from utility.utils_data import load_data
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

            t0 = time.time()
            # Fit and transform the summaries to get feature matrix
            vectorizer.fit_transform(abstracts)
            print(f"Time taken to fit and transform the summaries: {time.time() - t0:.2f} seconds")

            with open(path, 'wb') as f:
                pickle.dump(vectorizer, f)

        else:
            from model.vectorizer import CustomCountVectorizer
            vectorizer = CustomCountVectorizer(n_range=range(1, 5), args=args)
            vectorizer.load()

            dtm = vectorizer.fit_transform(abstracts, range(1, 5), stopwords, num_workers=args.num_workers)

            vectorizer.save()

            #
            # # Extract n-grams of length 1 to 4 in parallel
            # n_grams_list = extract_ngrams_parallel(abstracts, range(1, 5), stopwords, num_workers=args.num_workers)
            #
            # # Filter n-grams by minimum document frequency
            # n_grams_list = filter_ngrams_by_min_df(n_grams_list, 5)
            #
            # # Build CountVectorizer-like matrix
            # dtm = build_count_vectorizer_matrix(n_grams_list)

    return vectorizer


def main():
    df = load_data(args, subset="last_100" if args.debug else None)
    vectorizer = get_vectorizer(args, data=df)

    # Get feature names (i.e., the n-grams)
    # feature_names = vectorizer.get_feature_names_out()


if __name__ == "__main__":
    project_setup()
    args = parse_args()
    if args.debug:
        print("=" * 10, "RUNNING DEBUG MODE", "=" * 10)

    stopwords = set(stopwords.words("english"))
    main()
