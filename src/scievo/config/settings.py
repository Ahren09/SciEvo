"""Settings and configuration management for SciEvo package.

This module contains argument parsing and configuration settings for the SciEvo
scientometric analysis package.
"""

import argparse
import os
import os.path as osp
from typing import Tuple

from .constants import WORD2VEC, GCN


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for SciEvo analysis.
    
    Returns:
        argparse.Namespace: Parsed arguments containing all configuration options.
        
    Example:
        >>> args = parse_args()
        >>> print(args.model_name)
        'gcn'
    """
    parser = argparse.ArgumentParser(
        description="SciEvo: Longitudinal Scientometric Dataset Analysis"
    )
    
    # Analysis parameters
    parser.add_argument(
        '--base_year', 
        type=int, 
        default=2023,
        help="Base year for temporal analysis"
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=256,
        help="Batch size for model training and inference"
    )
    parser.add_argument(
        '--checkpoint_dir', 
        type=str, 
        default="checkpoints",
        help="Directory to store model checkpoints"
    )
    parser.add_argument(
        '--data_dir', 
        type=str,
        default="data",
        help="Directory containing the processed dataset"
    )
    parser.add_argument(
        '--debug', 
        action='store_true',
        help="Enable debug mode for verbose output"
    )
    parser.add_argument(
        '--do_plotly', 
        action='store_true',
        help="Use Plotly for interactive visualizations"
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default="cuda",
        help="Device for model training (cuda/cpu)"
    )
    parser.add_argument(
        '--do_visual', 
        action='store_true',
        help="Generate visualization outputs"
    )
    
    # Model parameters
    parser.add_argument(
        '--embedding_dim', 
        type=int, 
        default=50,
        help="Dimension of the embedding vectors"
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=50, 
        help="Number of training epochs"
    )
    parser.add_argument(
        '--random_seed', 
        type=int, 
        default=42, 
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        '--embed_dim', 
        type=int, 
        default=100, 
        help="Dimension of the generated embeddings"
    )
    parser.add_argument(
        '--graph_backend', 
        type=str, 
        default="networkx", 
        choices=["networkx", "rapids"],
        help="Backend for graph computations"
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=1e-2,
        help="Learning rate for model training"
    )
    parser.add_argument(
        '--model_name', 
        type=str, 
        choices=[WORD2VEC, GCN],  
        default=GCN,
        help="Model type for embedding generation"
    )
    
    # Processing parameters  
    parser.add_argument(
        '--num_workers', 
        type=int, 
        default=16,
        help="Number of worker processes for data loading"
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default="outputs", 
        help="Directory to store analysis results and outputs"
    )
    parser.add_argument(
        '--save_every', 
        type=int, 
        default=20, 
        help="Save model checkpoint every N epochs"
    )
    parser.add_argument(
        '--save_model', 
        action='store_true', 
        help="Save trained model checkpoints"
    )
    
    # Data filtering parameters
    parser.add_argument(
        '--start_year', 
        type=int, 
        default=None, 
        help="Start year for data filtering"
    )
    parser.add_argument(
        '--end_year', 
        type=int,
        default=None, 
        help="End year for data filtering"
    )
    parser.add_argument(
        '--load_from_cache', 
        action='store_true', 
        help="Load processed dataset from cache if available"
    )
    parser.add_argument(
        '--step_size', 
        type=int, 
        default=50,
        help="Step size for learning rate scheduler"
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    
    # Feature extraction parameters
    parser.add_argument(
        '--feature_name', 
        type=str, 
        choices=["title", "abstract", "title_and_abstract"],
        default='title',
        help="Text features to use for analysis"
    )
    parser.add_argument(
        '--tokenization_mode', 
        type=str, 
        choices=["unigram", "llm_extracted_keyword"], 
        default='llm_extracted_keyword',
        help="Tokenization method for text processing"
    )
    parser.add_argument(
        '--min_occurrences', 
        type=int, 
        default=3,
        help="Minimum occurrences for keyword inclusion"
    )
    
    # Visualization parameters
    parser.add_argument(
        '--graphistry_personal_key_id', 
        type=str, 
        default='',
        help="Graphistry personal key ID for graph visualizations"
    )
    parser.add_argument(
        '--graphistry_personal_key_secret', 
        type=str, 
        default='',
        help="Graphistry personal key secret for graph visualizations"
    )

    args = parser.parse_args()

    # Expand user paths and create directories
    args.data_dir = osp.expanduser(args.data_dir)
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def get_default_config() -> dict:
    """Get default configuration dictionary.
    
    Returns:
        dict: Default configuration parameters.
    """
    return {
        "base_year": 2023,
        "batch_size": 256,
        "embedding_dim": 50,
        "epochs": 50,
        "lr": 1e-2,
        "model_name": GCN,
        "feature_name": "title",
        "tokenization_mode": "llm_extracted_keyword",
        "min_occurrences": 3,
        "random_seed": 42
    }