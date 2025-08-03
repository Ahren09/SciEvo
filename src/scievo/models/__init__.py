"""Machine learning models for SciEvo package.

This module contains implementations of various models for analyzing
academic literature, including embedding models, graph neural networks,
and text processing models.
"""

from .embeddings import *
from .graph_models import *
from .text_models import *

__all__ = [
    # Embedding models
    "Word2VecEmbedding",
    "GraphEmbedding",
    
    # Graph models  
    "GCNModel",
    "GraphConvGRU",
    
    # Text models
    "TextVectorizer",
    "KeywordExtractor"
]