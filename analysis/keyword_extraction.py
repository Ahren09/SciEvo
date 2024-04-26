"""
This script performs keyword extraction from the arXiv dataset.

Requirements: `conda install -c conda-forge rake_nltk`
"""
import os
import os.path as osp
import re
import string
import sys
import traceback
import warnings
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
import spacy
from matplotlib import pyplot as plt
from nltk import BigramCollocationFinder, BigramAssocMeasures, TrigramCollocationFinder, TrigramAssocMeasures
from nltk.corpus import stopwords
from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from utility.utils_data import load_arXiv_data, get_titles_or_abstracts_as_list
from utility.utils_text import split_text_into_tokens, stopwords_set, english_stopwords, load_semantic_scholar_data

sys.path.insert(0, os.path.abspath('..'))

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
        "title": ["title_keywords", "title_unigrams"],
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


def extract_ngrams(features_list):
    """
    Extract unigrams, bigrams, and trigrams from the given list of features.

    """

    # Tokenize and preprocess each abstract
    all_tokens = []
    unigrams = []

    for i, entry in enumerate(tqdm(features_list, desc="Extract 1grams", total=len(features_list))):
        tokens_one_example = split_text_into_tokens(entry)
        all_tokens += [tokens_one_example]
        unigrams += [[token for token in tokens_one_example if token not in stopwords_set]]

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

    bigrams_set = set(bigram_finder.nbest(BigramAssocMeasures.likelihood_ratio, 50000))



    bigrams = []
    for tokens_one_example in tqdm(all_tokens, desc="Construct bigrams"):

        bigrams_one_examples = []
        for i in range(len(tokens_one_example) - 1):
            bigram = (tokens_one_example[i], tokens_one_example[i + 1])

            if bigram in bigrams_set:
                bigrams_one_examples += [bigram]

        bigrams += [bigrams_one_examples]



    # Extract trigrams
    trigram_finder = TrigramCollocationFinder.from_words(tokens_flatten)

    # only trigrams that appear 3+ times
    trigram_finder.apply_freq_filter(3)

    trigrams_set = set(trigram_finder.nbest(TrigramAssocMeasures.likelihood_ratio, 100000))

    trigrams_set = [(w1, w2, w3) for w1, w2, w3 in trigrams_set if
                         w3 not in {'the', 'of', 'from', 'in', 'on',
                                    'to', 'for', 'with', }]

    trigrams = []
    for tokens_one_example in tqdm(all_tokens, desc="Construct trigrams"):
        trigrams_one_examples = []
        for i in range(len(tokens_one_example) - 2):
            trigram = (tokens_one_example[i], tokens_one_example[i + 1], tokens_one_example[i + 2])

            if trigram in trigrams_set:
                trigrams_one_examples += [trigram]

        trigrams += [trigrams_one_examples]

    assert len(unigrams) == len(bigrams) == len(trigrams)

    return unigrams, bigrams, trigrams


if __name__ == "__main__":

    project_setup()
    args = parse_args()
    START_YEAR, START_MONTH = 2022, 10
    END_YEAR, END_MONTH = 2023, 1


    semantic_scholar_data = load_semantic_scholar_data(args.data_dir, START_YEAR, START_MONTH, END_YEAR, END_MONTH)

    # semantic_scholar_data.explode(const.FOS)["paperId"]


    if args.debug:
        data = pd.read_json("/Users/ahren/Workspace/Course/CS7450/CS7450_Homeworks/HW4/data/arXiv_2023_3-4.json")



    else:
        data = load_arXiv_data(args.data_dir, start_year=START_YEAR, start_month=START_MONTH, end_year=END_YEAR,
                               end_month=END_MONTH)

    # all_abstract_words = extract_unigrams_from_abstract()
    # json.dump(all_abstract_words, open("all_abstract_words.json", "w"), indent=2)



    unigrams, bigrams, trigrams = extract_ngrams(get_titles_or_abstracts_as_list(data, "title"))


    assert len(data) == len(unigrams) == len(bigrams) == len(trigrams)


    keyword2SSId_list = {subject: defaultdict(set) for subject in const.SUBJECT2KEYWORDS}


    G = {}

    # Temporal Graph
    for subject in const.SUBJECT2KEYWORDS:
        for keywords_tuple in const.SUBJECT2KEYWORDS[subject]:
            assert keywords_tuple[0] not in G
            G[keywords_tuple[0]] = defaultdict(set)


    for ngrams in [unigrams, bigrams, trigrams]:

        """
        Structure of keyword2SSId_list: 
        {
            subject -> {
                keyword -> [list of paper],
                keyword -> [list of paper],
                ...
            },
            subject -> {
                keyword -> [list of paper],
                keyword -> [list of paper],
                ...
            },
        }
        
        """

        # Pre-extract necessary data from DataFrame for faster speed
        fos_data = semantic_scholar_data[const.FOS].tolist()
        semantic_scholar_id_data = semantic_scholar_data[const.PAPERID].tolist()
        published_data = data[const.PUBLISHED].dt.strftime("%Y-%m").tolist()

        # Construct temporal graph

        for idx_example, ngrams_one_example in enumerate(tqdm(ngrams)):

            try:

                subject_one_paper = fos_data[idx_example]

                semantic_scholar_id = semantic_scholar_id_data[const.PAPERID]


                published_year_and_month = published_data[idx_example].strftime(
                    "%Y-%m")



                for subject in subject_one_paper:

                    if not subject in const.SUBJECT2KEYWORDS:
                        continue

                    for keywords_tuple in const.SUBJECT2KEYWORDS[subject]:

                        common_keywords = set(ngrams_one_example) & set(keywords_tuple)

                        if common_keywords:
                            print("=====================================")
                            print(f"ngrams: {ngrams_one_example}")
                            print(f"Subject: {subject}")
                            print(f"Common keywords: {common_keywords}")

                            # Each tuple in `keywords_tuple` contains a list of related keywords
                            # We use the first keyword in each tuple to represent the group

                            representative_keyword = keywords_tuple[0]

                            keyword2SSId_list[subject][representative_keyword].add(semantic_scholar_id)

                            G[representative_keyword][published_year_and_month].add(semantic_scholar_id)



            except:
                traceback.print_exc()


        stack_data = []
        snapshot_names = []
        for keyword in const.KEYWORD2ID:
            num_papers = []

            for year in range(START_YEAR, END_YEAR + 1):
                for month in range(1, 13):
                    if (month < START_MONTH and year <= START_YEAR) or (month >= END_MONTH and year >= END_YEAR):
                        continue

                    snapshot_name = f"{year}-{month:02d}"
                    num_papers += [len(G[keyword][snapshot_name])]

                    if snapshot_name not in snapshot_names:
                        snapshot_names += [snapshot_name]

            stack_data += [num_papers]

        stack_data = np.array(stack_data)

        # Mask out keywords that have no papers
        mask = np.array(stack_data).sum(axis=1) > 20

        labels = np.array(list(const.KEYWORD2ID.keys()))[mask].tolist()
        plt.stackplot(np.arange(len(snapshot_names)), *stack_data[mask], labels=labels)
        plt.legend(loc='upper left')
        plt.show()



        for subject, keyword2SSId in keyword2SSId_list.items():
            print(f"Subject: {subject}")
            for keyword, SSId_list in keyword2SSId.items():
                print(f"Keyword: {keyword} #Papers: {len(SSId_list)}")


    # For each subject and each example, match the extracted keywords with the subject keywords

    if args.num_workers == 1:
        d_list = [extract_keywords_for_row((idx_row, row)) for idx_row, row in
                  tqdm(data.iterrows(), desc="Extract keywords", total=len(data))]
    else:
        d_list = process_rows(data, num_workers=args.num_workers)

    keywords_df = pd.DataFrame(d_list)
    keywords_df['updated_datetime'] = pd.to_datetime(keywords_df[const.UPDATED], utc=True)
    keywords_df = keywords_df.sort_values(by=['updated_datetime'], ascending=False)
    keywords_df.drop(columns=['updated_datetime'], inplace=True)
    keywords_df.to_pickle(osp.join(args.data_dir, "Sample_keywords.pkl"))

    print("Done!")
