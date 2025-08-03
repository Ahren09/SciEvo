"""Text processing models for SciEvo package.

This module contains models for text processing, keyword extraction,
and document vectorization for academic literature analysis.
"""

import os
import os.path as osp
import pickle
import re
from copy import deepcopy
from typing import List, Set, Dict, Any, Optional

import nltk
import numpy as np
import pandas as pd
import scipy.sparse as sp
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from tqdm import tqdm

from ..config.constants import DATE_FORMAT, WORD2VEC
from ..utils.time_utils import TimeIterator


class Word2VecEmbedding:
    """Word2Vec embedding model for temporal analysis of academic literature.
    
    This class provides functionality to train Word2Vec models on temporal
    snapshots of academic literature and track semantic evolution over time.
    
    Attributes:
        embed_dim: Dimension of the word embeddings.
        window: Context window size for Word2Vec training.
        min_count: Minimum word frequency for inclusion in vocabulary.
        workers: Number of worker threads for training.
        seed: Random seed for reproducibility.
    """
    
    def __init__(
        self,
        embed_dim: int = 100,
        window: int = 4,
        min_count: int = 3,
        workers: int = 4,
        seed: int = 42
    ):
        """Initialize Word2Vec embedding model.
        
        Args:
            embed_dim: Dimension of word embeddings.
            window: Size of context window.
            min_count: Minimum frequency for word inclusion.
            workers: Number of training threads.
            seed: Random seed for reproducibility.
        """
        self.embed_dim = embed_dim
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.seed = seed
        self.models = {}
        self.shared_vocab = None
    
    def tokenize_documents(self, documents: List[str], stop_words: Optional[Set[str]] = None) -> List[List[str]]:
        """Tokenize documents and remove stop words.
        
        Args:
            documents: List of document strings to tokenize.
            stop_words: Set of stop words to exclude. If None, uses NLTK English stopwords.
            
        Returns:
            List of tokenized and filtered documents.
            
        Example:
            >>> model = Word2VecEmbedding()
            >>> docs = ["machine learning algorithms", "deep neural networks"]
            >>> tokens = model.tokenize_documents(docs)
            >>> print(tokens[0])
            ['machine', 'learning', 'algorithms']
        """
        if stop_words is None:
            stop_words = set(stopwords.words('english'))
        
        tokenized_docs = []
        for doc in documents:
            # Clean text
            doc = re.sub(r'[~/\n]', ' ', doc)
            doc = re.sub(r'(?<=\w)\.(?=\w)', '. ', doc)
            
            # Tokenize and filter
            tokens = nltk.word_tokenize(doc.lower())
            filtered_tokens = [token for token in tokens if token not in stop_words]
            tokenized_docs.append(filtered_tokens)
        
        return tokenized_docs
    
    def train_temporal_models(
        self,
        data: pd.DataFrame,
        start_year: int = 1985,
        end_year: int = 2025,
        feature_column: str = 'title',
        save_path: Optional[str] = None
    ) -> Dict[str, Word2Vec]:
        """Train Word2Vec models for each temporal snapshot.
        
        Args:
            data: DataFrame containing the documents with temporal information.
            start_year: Starting year for temporal analysis.
            end_year: Ending year for temporal analysis.
            feature_column: Column name containing the text to analyze.
            save_path: Directory to save trained models.
            
        Returns:
            Dictionary mapping time periods to trained Word2Vec models.
            
        Example:
            >>> model = Word2VecEmbedding()
            >>> models = model.train_temporal_models(df, start_year=2020, end_year=2023)
            >>> print(f"Trained {len(models)} temporal models")
        """
        iterator = TimeIterator(start_year, end_year, snapshot_type='yearly')
        
        for start, end in iterator:
            # Filter data for current time window
            mask = ((data['published'] >= start) & (data['published'] < end))
            data_snapshot = data[mask]
            
            if len(data_snapshot) == 0:
                continue
            
            # Prepare documents
            docs = data_snapshot[feature_column].tolist()
            tokenized_docs = self.tokenize_documents(docs)
            
            if len(tokenized_docs) == 0:
                continue
            
            print(f"Training Word2Vec for {start.year}: {len(tokenized_docs)} documents")
            
            # Train Word2Vec model
            try:
                model = Word2Vec(
                    sentences=tokenized_docs,
                    vector_size=self.embed_dim,
                    window=self.window,
                    min_count=self.min_count,
                    sg=1,  # Skip-gram
                    negative=5,
                    epochs=50,
                    workers=self.workers,
                    seed=self.seed
                )
            except TypeError:
                # Fallback for older gensim versions
                model = Word2Vec(
                    sentences=tokenized_docs,
                    size=self.embed_dim,
                    window=self.window,
                    min_count=self.min_count,
                    sg=1,
                    negative=5,
                    iter=50,
                    workers=self.workers,
                    seed=self.seed
                )
            
            # Store model
            period_key = f"{start.year}"
            self.models[period_key] = model
            
            # Save model if path provided
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                filename = f"word2vec_{start.year}.model"
                model.save(osp.join(save_path, filename))
        
        return self.models
    
    def build_shared_vocabulary(self) -> Dict[str, Any]:
        """Build a shared vocabulary across all temporal models.
        
        Creates a unified vocabulary that includes all words from all time periods,
        enabling temporal comparison of embeddings.
        
        Returns:
            Dictionary containing shared vocabulary mappings.
            
        Example:
            >>> model = Word2VecEmbedding()
            >>> # ... train models ...
            >>> shared_vocab = model.build_shared_vocabulary()
            >>> print(f"Shared vocabulary size: {len(shared_vocab['iw'])}")
        """
        if not self.models:
            raise ValueError("No models trained. Call train_temporal_models first.")
        
        shared_iw = []  # index to word
        shared_wi = {}  # word to index
        
        # Build shared vocabulary from all models
        for period_key, model in self.models.items():
            for word in model.wv.index_to_key:
                if word not in shared_wi:
                    shared_wi[word] = len(shared_wi)
                    shared_iw.append(word)
        
        self.shared_vocab = {
            'wi': shared_wi,
            'iw': shared_iw
        }
        
        return self.shared_vocab
    
    def get_temporal_embeddings(self, word: str) -> Dict[str, np.ndarray]:
        """Get embeddings for a word across all time periods.
        
        Args:
            word: Word to get temporal embeddings for.
            
        Returns:
            Dictionary mapping time periods to embedding vectors.
            
        Example:
            >>> embeddings = model.get_temporal_embeddings("machine")
            >>> for period, embedding in embeddings.items():
            ...     print(f"{period}: {embedding[:3]}")  # First 3 dimensions
        """
        temporal_embeddings = {}
        
        for period_key, model in self.models.items():
            if word in model.wv.key_to_index:
                temporal_embeddings[period_key] = model.wv[word]
        
        return temporal_embeddings
    
    def find_similar_words(self, word: str, period: str, topn: int = 10) -> List[tuple]:
        """Find words most similar to a given word in a specific time period.
        
        Args:
            word: Target word to find similarities for.
            period: Time period identifier.
            topn: Number of similar words to return.
            
        Returns:
            List of (word, similarity_score) tuples.
            
        Example:
            >>> similar = model.find_similar_words("learning", "2023", topn=5)
            >>> for word, score in similar:
            ...     print(f"{word}: {score:.3f}")
        """
        if period not in self.models:
            raise ValueError(f"No model found for period: {period}")
        
        model = self.models[period]
        
        if word not in model.wv.key_to_index:
            return []
        
        return model.wv.most_similar(word, topn=topn)
    
    def save_shared_vocabulary(self, save_path: str) -> None:
        """Save shared vocabulary to disk.
        
        Args:
            save_path: Path to save the vocabulary file.
            
        Example:
            >>> model.save_shared_vocabulary("checkpoints/shared_vocab.pkl")
        """
        if self.shared_vocab is None:
            self.build_shared_vocabulary()
        
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(self.shared_vocab, f)


class TextVectorizer:
    """Custom text vectorizer for academic literature analysis.
    
    Provides functionality for converting text documents to numerical
    representations suitable for machine learning analysis.
    """
    
    def __init__(self, ngram_range: tuple = (1, 1), min_df: int = 1, max_df: float = 1.0):
        """Initialize the text vectorizer.
        
        Args:
            ngram_range: Range of n-grams to extract.
            min_df: Minimum document frequency for term inclusion.
            max_df: Maximum document frequency for term inclusion.
        """
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.vocabulary = {}
        self.document_term_matrix = None
    
    def fit_transform(self, documents: List[str]) -> sp.csr_matrix:
        """Fit vectorizer to documents and transform them.
        
        Args:
            documents: List of text documents to vectorize.
            
        Returns:
            Sparse document-term matrix.
        """
        # Implementation would go here
        # This is a placeholder for the actual vectorization logic
        pass
    
    def transform(self, documents: List[str]) -> sp.csr_matrix:
        """Transform documents using fitted vectorizer.
        
        Args:
            documents: List of text documents to transform.
            
        Returns:
            Sparse document-term matrix.
        """
        # Implementation would go here
        pass


def process_documents(documents: List[str]) -> List[List[str]]:
    """Preprocess a list of documents for analysis.
    
    Args:
        documents: List of document strings to process.
        
    Returns:
        List of tokenized and filtered documents.
        
    Example:
        >>> docs = ["Machine learning is powerful", "Deep learning works well"]
        >>> processed = process_documents(docs)
        >>> print(processed[0])
        ['machine', 'learning', 'powerful']
    """
    stop_words = set(stopwords.words('english'))
    
    tokenized_docs = []
    for doc in documents:
        # Clean and tokenize
        doc = re.sub(r'[~/\n]', ' ', doc)
        doc = re.sub(r'(?<=\w)\.(?=\w)', '. ', doc)
        tokens = nltk.word_tokenize(doc.lower())
        
        # Filter tokens
        filtered_tokens = [token for token in tokens if token not in stop_words]
        tokenized_docs.append(filtered_tokens)
    
    return tokenized_docs