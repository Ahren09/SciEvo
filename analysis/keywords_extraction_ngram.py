"""
Extract keywords from the titles / abstracts (summary) using n-gram.
"""

import os
import os.path as osp
import pickle
import re
import time
from functools import lru_cache

from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from arguments import parse_args
from utility.utils_data import load_data
from utility.utils_misc import project_setup

stop_set = set(stopwords.words())
lemmatizer = WordNetLemmatizer()


@lru_cache(maxsize=10000)
def lemmatize(word):
    return lemmatizer.lemmatize(word.lower().strip())


def tokenize(text):
    text = re.sub('[^a-zA-Z0-9]+', ' ', text)
    words = [lemmatize(w) for w in word_tokenize(text)]
    return [w for w in words if w not in stop_set]







def get_vectorizer(args, data=None):
    """
    Get a vectorizer object for extracting n-grams from the text.

    Args:
        args (argparse.Namespace): Arguments.

    Returns:
        vectorizer (sklearn.feature_extraction.text.CountVectorizer): Vectorizer object.
    """
    path = osp.join(args.output_dir, 'tfidf_vectorizer.pkl')

    if False:  # osp.exists(path):
        with open(path, 'rb') as f:
            vectorizer = pickle.load(f)
            return vectorizer

    else:
        if USE_STEMMER:

            english_stemmer = Stemmer.Stemmer('en')

            class StemmedTfidfVectorizer(TfidfVectorizer):
                def build_analyzer(self):
                    analyzer = super(TfidfVectorizer, self).build_analyzer()
                    return lambda doc: english_stemmer.stemWords(analyzer(doc))

            vectorizer = StemmedTfidfVectorizer(tokenizer=tokenize, min_df=5, stop_words='english', analyzer='word',
                                           ngram_range=(
                                               1, 4))

        else:


            # Initialize a CountVectorizer/TfidfVectorizer with ngram_range=(1, 4)

            # vectorizer = CountVectorizer(ngram_range=(1, 4), min_df=5)
            vectorizer = TfidfVectorizer(tokenizer=tokenize, ngram_range=(1, 4), min_df=5, stop_words='english')

        abstracts = [abs for abs in data['summary'].tolist() if isinstance(abs, str)]

        t0 = time.time()
        # Fit and transform the summaries to get feature matrix
        vectorizer.fit_transform(abstracts)
        print(f"Time taken to fit and transform the summaries: {time.time() - t0:.2f} seconds")

        with open(path, 'wb') as f:
            pickle.dump(vectorizer, f)

    return vectorizer


def main():
    df = load_data(args)
    vectorizer = get_vectorizer(args, data=df)

    # Get feature names (i.e., the n-grams)
    feature_names = vectorizer.get_feature_names_out()

    # Sum up the counts of each n-gram across all documents
    # Sort n-grams and counts by frequency
    sorted_indices = ngram_counts.argsort()[::-1]
    sorted_ngrams = [(feature_names[i], ngram_counts[i]) for i in sorted_indices]


if __name__ == "__main__":
    project_setup()
    args = parse_args()

    import Stemmer
    USE_STEMMER = True

    os.makedirs(args.output_dir, exist_ok=True)
    main()
