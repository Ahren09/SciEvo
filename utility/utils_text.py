import re

from nltk import RegexpTokenizer
from nltk.corpus import stopwords

english_stopwords = stopwords.words("english") + ['']

    # Initialize Stopwords
stopwords_set = set(stopwords.words('english'))


def split_text_into_tokens(text):
    # Remove newlines and extra spaces
    text = text.replace('\n', ' ').strip()
    text = re.sub(' +', ' ', text)

    # Lowercase and remove punctuation
    tokenizer = RegexpTokenizer(r'[\$\[\]\{\}\w\\\-_]+')
    tokens = tokenizer.tokenize(text.lower())
    tokens = [token.strip('.?!,') for token in tokens if token not in stopwords_set]
    return tokens
