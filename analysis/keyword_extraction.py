"""
ARCHIVED

This script performs keyword extraction from the arXiv dataset using ngrams and Rake.

Requirements: `conda install -c conda-forge rake_nltk`
"""
import json
import os
import os.path as osp
import re
import string
import sys
import warnings
from collections import Counter
from multiprocessing import Pool
from typing import List

import pandas as pd
from nltk import BigramCollocationFinder, BigramAssocMeasures, TrigramCollocationFinder, TrigramAssocMeasures
from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

from utility.utils_data import load_arXiv_data, get_titles_or_abstracts_as_list
from utility.utils_text import split_text_into_tokens, stopwords_set, english_stopwords, load_semantic_scholar_data

import const
from arguments import parse_args
from utility.utils_misc import project_setup

# Suppress FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

nlp = None  # Placeholder for Spacy model


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


def extract_keywords_spacy(text):
    global nlp

    # Load Spacy model if not already loaded
    if nlp is None:
        import spacy
        nlp = spacy.load("en_core_web_md")

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
        "title_llm_extracted_keyword": ["title_keywords", "title_unigrams"],
        "summary": ["abstract_keywords", "abstract_unigrams"]
    }.items():
        keywords, unigrams = extract_keywords_spacy(row[col_name])

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


def extract_words(features_list: List[str]):
    tokens = []
    for i, entry in enumerate(tqdm(features_list, desc="Extract individual words", total=len(features_list))):
        tokens_one_example = split_text_into_tokens(entry)
        tokens += [tokens_one_example]

    return tokens


def extract_1grams(tokens: List[List[str]]):
    unigrams = []
    for i in range(len(tokens)):
        unigrams += [[token for token in tokens[i] if token not in stopwords_set]]

    count_unigrams = Counter([token for tokens_one_example in unigrams for token in tokens_one_example])

    num_unigrams = sum(1 for count in count_unigrams.values() if count >= args.min_occurrences)
    print(f"Number of unigrams (unfiltered): {num_unigrams}")

    tokens = [[token for token in tokens_one_example if count_unigrams[token] >= args.min_occurrences] for
              tokens_one_example
              in unigrams]

    tokens_flatten = [token for tokens_one_example in tokens for token in tokens_one_example]

    # hard-cap #unigrams at a reasonable number
    unigrams_set = pd.DataFrame(Counter(tokens_flatten).most_common(200000), columns=["keyword", "count"])
    unigrams_set = unigrams_set[unigrams_set['count'] >= args.min_occurrences][:200000].keyword.tolist()
    print(f"Number of unigrams (filtered): {len(unigrams_set)}")

    unigrams, unigrams_filtered = [], []
    for tokens_one_example in tqdm(tokens, desc="Construct 1-grams"):

        unigrams_one_examples, unigrams_one_examples_filtered = [], []
        for i in range(len(tokens_one_example)):
            unigram = tokens_one_example[i]

            if unigram in unigrams_set:
                unigrams_one_examples_filtered += [unigram]
            # unigrams_one_examples += [unigram]

        # unigrams += [unigrams_one_examples]
        unigrams_filtered += [unigrams_one_examples_filtered]

    assert len(unigrams_filtered) == len(tokens)

    return unigrams, unigrams_filtered


def extract_2grams(tokens: List[List[str]]):
    tokens_flatten = [token for tokens_one_example in tokens for token in tokens_one_example]

    # Extract bigrams
    bigram_finder = BigramCollocationFinder.from_words(tokens_flatten)

    bigram_finder.apply_word_filter(lambda x: any(stop_word == x for stop_word in
                                                  {'$', 'a', 'are', 'by', 'for', 'how', 'in', 'is', 'not', 'of', 'on',
                                                   'than', 'the', 'to', 'what', 'when', 'with'}))
    bigram_finder.apply_freq_filter(args.min_occurrences)  # only bigrams that appear 3+ times

    bigrams_set = set(bigram_finder.nbest(BigramAssocMeasures.likelihood_ratio, 200000))

    bigrams, bigrams_filtered = [], []
    for tokens_one_example in tqdm(tokens, desc="Construct 2-grams"):

        bigrams_one_examples, bigrams_one_examples_filtered = [], []
        for i in range(len(tokens_one_example) - 1):
            bigram = (tokens_one_example[i], tokens_one_example[i + 1])

            if bigram in bigrams_set:
                bigrams_one_examples_filtered += [bigram]
            # bigrams_one_examples += [bigram]

        # bigrams += [bigrams_one_examples]
        bigrams_filtered += [bigrams_one_examples_filtered]

    assert len(bigrams_filtered) == len(tokens)
    return bigrams_filtered


def extract_3grams(tokens: List[List[str]]):
    """
    Extract unigrams, bigrams, and trigrams from the given list of features.

    """

    # Tokenize and preprocess each abstract

    tokens_flatten = [token for tokens_one_example in tokens for token in tokens_one_example]

    # Extract trigrams
    trigram_finder = TrigramCollocationFinder.from_words(tokens_flatten)

    # only trigrams that appear 3+ times
    trigram_finder.apply_freq_filter(args.min_occurrences)

    trigrams_set = set(trigram_finder.nbest(TrigramAssocMeasures.likelihood_ratio, 200000))

    trigrams_set = [(w1, w2, w3) for w1, w2, w3 in trigrams_set if
                    w3 not in {'the', 'of', 'from', 'in', 'on',
                               'to', 'for', 'with', }]

    trigrams, trigrams_filtered = [], []
    for tokens_one_example in tqdm(tokens, desc="Construct 3-grams"):
        trigrams_one_examples, trigrams_one_examples_filtered = [], []
        for i in range(len(tokens_one_example) - 2):
            trigram = (tokens_one_example[i], tokens_one_example[i + 1], tokens_one_example[i + 2])

            if trigram in trigrams_set:
                trigrams_one_examples_filtered += [trigram]
            trigrams_one_examples += [trigram]

        # trigrams += [trigrams_one_examples]
        trigrams_filtered += [trigrams_one_examples_filtered]

    assert len(trigrams_filtered) == len(tokens)

    return trigrams_filtered


if __name__ == "__main__":

    project_setup()
    args = parse_args()
    START_YEAR, START_MONTH = 1990, 1
    END_YEAR, END_MONTH = 2024, 4

    semantic_scholar_data = load_semantic_scholar_data(args.data_dir, START_YEAR, START_MONTH, END_YEAR, END_MONTH)

    if args.debug:
        data = pd.read_json("/Users/ahren/Workspace/Course/CS7450/CS7450_Homeworks/HW4/data/arXiv_2023_3-4.json")



    else:
        data = load_arXiv_data(args.data_dir, start_year=START_YEAR, start_month=START_MONTH, end_year=END_YEAR,
                               end_month=END_MONTH)

    # all_abstract_words = extract_unigrams_from_abstract()
    # json.dump(all_abstract_words, open("all_abstract_words.json", "w"), indent=2)

    for col in ["title_llm_extracted_keyword", "summary"]:
        feature_list = get_titles_or_abstracts_as_list(data, col)
        tokens = extract_words(feature_list)
        json.dump(tokens, open(osp.join(args.output_dir, f"{col}_tokens.json"), "w"), indent=2)

        unigrams_filtered = extract_1grams(tokens)
        # json.dump(unigrams, open(osp.join(args.output_dir, f"{col}_unigrams.json"), "w"), indent=2)
        json.dump(unigrams_filtered, open(osp.join(args.output_dir, f"{col}_unigrams_filtered.json"), "w"), indent=2)

        bigrams_filtered = extract_2grams(tokens)
        # json.dump(bigrams, open(osp.join(args.output_dir, f"{col}_bigrams.json"), "w"), indent=2)
        json.dump(bigrams_filtered, open(osp.join(args.output_dir, f"{col}_bigrams_filtered.json"), "w"), indent=2)

        trigrams_filtered = extract_3grams(tokens)
        # json.dump(trigrams, open(osp.join(args.output_dir, f"{col}_trigrams.json"), "w"), indent=2)
        json.dump(trigrams_filtered, open(osp.join(args.output_dir, f"{col}_trigrams_filtered.json"), "w"), indent=2)

    exit(0)

    # For each subject and each example, match the extracted keywords with the subject keywords
