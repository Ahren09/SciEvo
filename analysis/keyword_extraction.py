"""
This script performs keyword extraction from the arXiv dataset.

Requirements: `conda install -c conda-forge rake_nltk`
"""
import os
import os.path as osp
import re
import string
import sys
import warnings
from collections import Counter
from multiprocessing import Pool

import pandas as pd
import spacy
from nltk import BigramCollocationFinder, BigramAssocMeasures, TrigramCollocationFinder, TrigramAssocMeasures, \
    RegexpTokenizer
from nltk.corpus import stopwords
from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from utility.utils_data import load_arXiv_data, get_titles_or_abstracts_as_list

sys.path.insert(0, os.path.abspath('..'))

import const
from arguments import parse_args
from utility.utils_misc import project_setup

# Suppress FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)


def custom_tokenizer(text):
    """
    Custom tokenizer that processes the text and returns tokens.

    Args:
        text (str): Text to tokenize.

    Returns:
        list: A list of tokens.
    """
    return re.findall(r'\b\w[\w-]*\w\b|\b\w\b', text.lower())


class CustomRake(Rake):
    """
    Custom Rake class to facilitate keyword extraction.
    """

    def __init__(self, min_length=1, max_length=3):
        super().__init__(min_length=min_length, max_length=max_length,
                         word_tokenizer=custom_tokenizer)
        self._regex_pattern = re.compile(r"[\w-]+")


def extract_keywords(text):
    """
    Extract keywords from a given text using NLP and other methods.

    Args:
        text (str): Text to extract keywords from.

    Returns:
        tuple: A tuple containing keywords and unigrams.
    """
    text = text.replace('\n', ' ').strip()
    text = re.sub(' +', ' ', text)

    keywords = []
    for text_segment in text.split(":"):
        doc = nlp(text_segment)
        keywords += [chunk.text.strip(string.punctuation + " ") for chunk in doc.noun_chunks]

    unigrams = [token.text.strip(string.punctuation + " ") for token in doc
                if token.pos_ in ('VERB', 'NOUN', 'PROPN', 'ADJ')]

    unigrams = list(set(unigrams) - stopwords_set)
    keywords = list(set(keywords) - stopwords_set - set(unigrams))

    return [w.lower() for w in keywords], [w.lower() for w in unigrams]


def process_rows(data, num_workers: int = 4):
    """
    Multiprocess rows of a DataFrame for keyword extraction.

    Args:
        data (pd.DataFrame): DataFrame to process.

    Returns:
        list: List of dictionaries containing extracted keywords.
    """
    with Pool(num_workers) as pool:
        df_list = list(pool.imap(extract_keywords_for_row, data.iterrows(), chunksize=1))
    return df_list


def extract_keywords_for_row(tup):
    """
    Extract keywords for a given row in DataFrame.

    Args:
        tup (tuple): Tuple containing index and row data.

    Returns:
        dict: Dictionary with extracted keywords and other metadata.
    """
    idx_row, row = tup
    d = {}

    for col_name, col_data in {
        "title": ["title_keywords", "title_unigrams"],
        "summary": ["abstract_keywords", "abstract_unigrams"]
    }.items():
        keywords, unigrams = extract_keywords(row[col_name])

        d[col_data[0]] = keywords
        d[col_data[1]] = unigrams

    d[const.UPDATED] = row[const.UPDATED]
    d[const.ID] = row[const.ID]

    return d


def extract_unigrams_from_abstract(data):
    abstracts = get_titles_or_abstracts_as_list(data, "summary")

    # Custom tokenizer that does not split on hyphens
    def custom_tokenizer(text):
        # Split text by spaces and other delimiters that are not hyphens
        pattern = r'[$]?[\w.\$\[\]\{\}\(\)\_\-]+[$]?'  # r'\b[\w.$_-]+\b'
        tokens = re.findall(pattern, text.lower())  # Adjust regex as needed
        return tokens

    tfidf_vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, stop_words=english_stopwords +
                                                                              const.ACADEMIC_STOP_WORDS)

    # Fit and transform the 'summary' text
    tfidf_matrix = tfidf_vectorizer.fit_transform(abstracts)

    # Get feature names
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Transpose the TF-IDF matrix so that each row corresponds to a word
    tfidf_matrix_transposed = tfidf_matrix.transpose()

    # Convert to array and sum TF-IDF scores across all documents for each word
    word_scores = tfidf_matrix_transposed.sum(axis=1)

    # Zip together the word scores with their corresponding feature names
    scores = zip(feature_names, word_scores)

    # Sort the scores in descending order
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Print top 10 words with highest TF-IDF scores
    for score, feature in sorted_scores[:10]:
        print(f"{feature}: {score}")

    all_abstract_words = []

    # Temporarily select this threshold
    TFIDF_THRESHOLD = 1.
    words_kept = set([feature for feature, score in sorted_scores if score >= TFIDF_THRESHOLD])
    for abs in tqdm(abstracts, desc="Extract words from abstract"):
        abstract_words = custom_tokenizer(abs)
        abstract_words = [word for word in abstract_words if word not in english_stopwords]
        all_abstract_words += [abstract_words]

    return all_abstract_words


def preprocess_text(text):
    # Remove newlines and extra spaces
    text = text.replace('\n', ' ').strip()
    text = re.sub(' +', ' ', text)

    # Lowercase and remove punctuation

    tokenizer = RegexpTokenizer(r'[\$\[\]\{\}\w\\\-_]+')
    tokens = tokenizer.tokenize(text.lower())
    return tokens


def extract_unigram_keywords(text):
    """
    Archived due to speed issues
    Args:
        text:

    Returns:

    """
    keywords = []
    for text_segment in text.split(":"):
        doc = nlp(text_segment)
        keywords += [chunk.text.strip(string.punctuation + " ") for chunk in doc.noun_chunks]

    unigrams = [token.text.strip(string.punctuation + " ") for token in doc
                if token.pos_ in ('VERB', 'NOUN', 'PROPN', 'ADJ')]

    unigrams = list(set(unigrams) - stopwords_set)
    return [w.lower() for w in unigrams if w != '']


def extract_ngrams(features_list):
    # Tokenize and preprocess each abstract
    all_tokens = []
    unigrams_all_examples = []

    for i, entry in enumerate(tqdm(features_list, desc="Extract 1grams", total=len(features_list))):
        tokens_one_example = preprocess_text(entry)
        all_tokens += [tokens_one_example]
        unigrams_all_examples += [[token for token in tokens_one_example if token not in stopwords_set]]

    # Extract unigrams
    count_unigrams = Counter([token for tokens_one_example in all_tokens for token in tokens_one_example])

    count = sum(1 for count in count_unigrams.values() if count >= 3)

    tokens_flatten = [token for tokens_one_example in all_tokens for token in tokens_one_example]

    # Extract bigrams
    bigram_finder = BigramCollocationFinder.from_words(tokens_flatten)

    bigram_finder.apply_word_filter(lambda x: any(stop_word == x for stop_word in
                                                  {'$', 'a', 'are', 'by', 'for', 'how', 'in', 'is', 'not', 'of', 'on',
                                                   'than', 'the', 'to', 'what', 'when', 'with'}))
    bigram_finder.apply_freq_filter(3)  # only bigrams that appear 3+ times

    bigrams = bigram_finder.nbest(BigramAssocMeasures.likelihood_ratio, 50000)
    bigrams = set(bigrams)

    bigrams_all_examples = []

    for tokens_one_example in all_tokens:

        bigrams_one_examples = []
        for i in range(len(tokens_one_example) - 1):
            bigram = (tokens_one_example[i], tokens_one_example[i + 1])

            if bigram in bigrams:
                bigrams_all_examples += [bigram]



    # Extract trigrams
    trigram_finder = TrigramCollocationFinder.from_words(tokens_flatten)

    # only trigrams that appear 3+ times
    trigram_finder.apply_freq_filter(3)

    filtered_trigrams = [(w1, w2, w3) for w1, w2, w3 in trigram_finder.ngram_fd if
                         w3 not in {'the', 'of', 'from', 'in', 'on',
                                    'to', 'for', 'with', }]

    trigrams = trigram_finder.nbest(TrigramAssocMeasures.likelihood_ratio, 50000)

    return unigrams, bigrams, trigrams


if __name__ == "__main__":

    project_setup()
    args = parse_args()

    # Load Spacy model
    nlp = spacy.load("en_core_web_md")

    english_stopwords = stopwords.words("english") + ['']

    # Initialize Stopwords
    stopwords_set = set(stopwords.words('english'))

    if args.debug:
        data = pd.read_json("/Users/ahren/Workspace/Course/CS7450/CS7450_Homeworks/HW4/data/arXiv_2023_3-4.json")

    else:
        data = load_arXiv_data(args.data_dir)

    # all_abstract_words = extract_unigrams_from_abstract()
    # json.dump(all_abstract_words, open("all_abstract_words.json", "w"), indent=2)

    keywords = extract_ngrams(get_titles_or_abstracts_as_list(data, "title"))

    if args.num_workers == 1:
        d_list = [extract_keywords_for_row((idx_row, row)) for idx_row, row in
                  tqdm(data.iterrows(), desc="Extract keywords", total=len(data))]
    else:
        d_list = process_rows(data, num_workers=args.num_workers)

    keywords_df = pd.DataFrame(d_list)
    keywords_df['updated_datetime'] = pd.to_datetime(keywords_df[const.UPDATED])
    keywords_df = keywords_df.sort_values(by=['updated_datetime'], ascending=False)
    keywords_df.drop(columns=['updated_datetime'], inplace=True)
    keywords_df.to_pickle(osp.join(args.data_dir, "Sample_keywords.pkl"))

    print("Done!")
