import os

from sklearn.feature_extraction.text import CountVectorizer

from arguments import parse_args
from utility.utils_data import load_data
from utility.utils_misc import project_setup


def main():
    # Initialize and train a CountVectorizer with some sample data
    vectorizer = CountVectorizer(ngram_range=(1, 4), min_df=5, stop_words='english')
    df = load_data(args)

    # New document
    new_doc = ["This is a new document that we want to check for specific keywords."]

    # Transform the new document using the trained vectorizer
    new_X = vectorizer.transform(new_doc)

    # Convert the sparse matrix row to a dense array
    new_X_dense = new_X.toarray()[0]

    # List of keywords to check for
    keywords_to_check = ['new', 'document', 'specific', 'keywords']

    # Find the indices of the keywords in the feature names
    keyword_indices = [vectorizer.get_feature_names_out().tolist().index(kw) for kw in keywords_to_check if
                       kw in vectorizer.get_feature_names_out()]

    # Check if the keywords appear in the new document
    for kw, idx in zip(keywords_to_check, keyword_indices):
        if new_X_dense[idx] > 0:
            print(f"The keyword '{kw}' appears in the new document.")
        else:
            print(f"The keyword '{kw}' does not appear in the new document.")


if __name__ == "__main__":
    project_setup()
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main()
