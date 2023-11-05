import os
import os.path as osp
import re
import sys
from pathlib import Path

import nltk
import pandas as pd
from gensim.models import Word2Vec
from nltk.corpus import stopwords

from model.vectorizer import CustomCountVectorizer
from utility.utils_data import load_data
from utility.utils_misc import project_setup

sys.path.append(osp.join(os.getcwd(), "src"))
import const
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

def main():

    print("Loading data...", end='\r')
    df = load_data(args, subset="last_10000")
    print("Done!")

    df['published'] = pd.to_datetime(df['published'])

    vectorizer = CustomCountVectorizer(n_range=range(1, 3), args=args)

    vectorizer.load()

    for start_year in range(2023, 2024):

        for start_month in range(7, 11):

            if start_month == 12:
                end_year = start_year + 1
                end_month = 1
            else:
                end_year = start_year
                end_month = start_month + 1

            start_date = f"{start_year}-{start_month:02d}-01"
            end_date = f"{end_year}-{end_month:02d}-01"


            mask = (df['published'] >= f"{start_date}") & (df['published'] < f"{end_date}")
            df_snapshot = df[mask]

            abstracts = df_snapshot['summary'].tolist()
            print(f"Generating embeds from {start_date} to {end_date}: {len(abstracts)} documents")

            tokenized_docs = process_documents(abstracts)

            try:
                model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=4, min_count=3,
                                 sg=1, negative=5, epochs=50, workers=15, seed=42)
            except:
                model = Word2Vec(sentences=tokenized_docs, size=100, window=4,
                                 min_count=3, sg=1, negative=5, iter=50, workers=15, seed=42)


            similar_words = model.wv.most_similar('computer', topn=20)
            print(f"({start_year}-{end_year}) Words most similar to 'computer':")
            for word, similarity in similar_words:
                print(f"\t{word}: {similarity:.4f}")


            if args.save_model:
                filename = f"word2vec_{start_date}-{end_date}.model"
                print(f"Saving model to {filename}")
                model.save(osp.join(model_path, filename))

if __name__ == "__main__":
    project_setup()

    args = parse_args()
    model_path = osp.join("checkpoints", "word2vec")
    os.makedirs(model_path, exist_ok=True)
    main()
