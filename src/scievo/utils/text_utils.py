"""Text processing utilities for SciEvo package.

This module provides functions for text preprocessing, tokenization, and
keyword extraction from academic literature.
"""

import re
from typing import List, Set

import pandas as pd
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
import os.path as osp

# Initialize English stopwords
english_stopwords = stopwords.words("english") + ['']
stopwords_set = set(stopwords.words('english'))


def split_text_into_tokens(text: str) -> List[str]:
    """Split text into clean tokens for analysis.
    
    Processes text by removing punctuation, converting to lowercase,
    and filtering out stopwords.
    
    Args:
        text: Raw text string to tokenize.
        
    Returns:
        List of cleaned tokens.
        
    Example:
        >>> tokens = split_text_into_tokens("Deep Learning in NLP!")
        >>> print(tokens)
        ['deep', 'learning', 'nlp']
    """
    # Remove newlines and extra spaces
    text = text.lower().replace('\n', ' ').strip()
    text = re.sub(' +', ' ', text)

    # Tokenize and clean
    tokenizer = RegexpTokenizer(r'[\$\[\]\{\}\w\\\-_]+')
    tokens = tokenizer.tokenize(text.lower())
    tokens = [token.strip('.?!,') for token in tokens if token not in stopwords_set]
    return tokens


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and normalizing format.
    
    Args:
        text: Raw text string to clean.
        
    Returns:
        Cleaned text string.
        
    Example:
        >>> clean_text("  Hello   world!  \\n")
        'Hello world!'
    """
    if not text:
        return ""
    
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text.strip())
    return text


def extract_keywords(text: str, min_length: int = 2) -> List[str]:
    """Extract keywords from text using simple tokenization.
    
    Args:
        text: Input text to extract keywords from.
        min_length: Minimum length for keywords to include.
        
    Returns:
        List of extracted keywords.
        
    Example:
        >>> keywords = extract_keywords("machine learning algorithms")
        >>> print(keywords)
        ['machine', 'learning', 'algorithms']
    """
    tokens = split_text_into_tokens(text)
    return [token for token in tokens if len(token) >= min_length]


def preprocess_academic_text(text: str, remove_numbers: bool = True) -> str:
    """Preprocess academic text for analysis.
    
    Performs comprehensive text cleaning specifically designed for
    academic literature processing.
    
    Args:
        text: Raw academic text.
        remove_numbers: Whether to remove numeric tokens.
        
    Returns:
        Preprocessed text string.
        
    Example:
        >>> text = "The algorithm achieves 95.2% accuracy on dataset."
        >>> preprocess_academic_text(text)
        'algorithm achieves accuracy dataset'
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove numbers if specified
    if remove_numbers:
        text = re.sub(r'\d+', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove stopwords
    tokens = text.split()
    tokens = [token for token in tokens if token not in stopwords_set and len(token) > 1]
    
    return ' '.join(tokens)


def load_semantic_scholar_data_one_month(data_dir: str, year: int, month: int) -> pd.DataFrame:
    """Load Semantic Scholar data for a specific month.
    
    Args:
        data_dir: Base directory containing the data.
        year: Year of the data to load.
        month: Month of the data to load.
        
    Returns:
        DataFrame containing the loaded data.
        
    Note:
        This function currently loads a hardcoded file path and should be
        updated to use the provided parameters.
    """
    return pd.read_json(
        osp.join(data_dir, "NLP", "semantic_scholar", "semantic_scholar_2023_3.json"), 
        orient='index'
    )