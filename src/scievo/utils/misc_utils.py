"""Miscellaneous utilities for SciEvo package.

This module provides general utility functions for project setup,
random seed management, and common operations.
"""

import os
import os.path as osp
import random
import warnings
from typing import Optional

import numpy as np


def check_cwd() -> None:
    """Check if current working directory is the SciEvo project root.
    
    Raises:
        AssertionError: If not running from the SciEvo project directory.
        
    Example:
        >>> check_cwd()  # Should pass if in SciEvo/ directory
    """
    basename = osp.basename(osp.normpath(os.getcwd()))
    assert basename.lower() in ["scievo"], \
        "Please run this file from parent directory (SciEvo/)"


def project_setup() -> None:
    """Set up the project environment with default configurations.
    
    Configures pandas display options, suppresses warnings, and sets
    random seed for reproducibility.
    
    Example:
        >>> project_setup()  # Sets up project environment
    """
    check_cwd()
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    try:
        import pandas as pd
        pd.set_option('display.max_rows', 40)
        pd.set_option('display.max_columns', 20)
    except ImportError:
        pass
    
    set_random_seed(42)


def set_random_seed(seed: int, use_torch: bool = True) -> None:
    """Set random seeds for reproducibility across different libraries.
    
    Args:
        seed: Random seed value to use.
        use_torch: Whether to set PyTorch random seed (if available).
        
    Example:
        >>> set_random_seed(42)  # Sets seed for numpy, random, and torch
    """
    random.seed(seed)
    np.random.seed(seed)

    if use_torch:
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass


def ensure_dir_exists(directory: str) -> str:
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Path to the directory.
        
    Returns:
        The directory path (for chaining operations).
        
    Example:
        >>> output_dir = ensure_dir_exists("outputs/analysis")
    """
    os.makedirs(directory, exist_ok=True)
    return directory


def get_project_root() -> str:
    """Get the project root directory path.
    
    Returns:
        Path to the project root directory.
        
    Example:
        >>> root = get_project_root()
        >>> print(root)
        '/path/to/SciEvo'
    """
    current_file = osp.abspath(__file__)
    # Navigate up from src/scievo/utils/misc_utils.py to project root
    return osp.dirname(osp.dirname(osp.dirname(osp.dirname(current_file))))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: The numerator value.
        denominator: The denominator value.
        default: Default value to return if denominator is zero.
        
    Returns:
        The division result or default value.
        
    Example:
        >>> result = safe_divide(10, 2)  # Returns 5.0
        >>> result = safe_divide(10, 0)  # Returns 0.0
    """
    return numerator / denominator if denominator != 0 else default


def format_number(num: float, precision: int = 2) -> str:
    """Format a number for display with appropriate precision.
    
    Args:
        num: Number to format.
        precision: Number of decimal places.
        
    Returns:
        Formatted number string.
        
    Example:
        >>> formatted = format_number(3.14159, 2)
        >>> print(formatted)
        '3.14'
    """
    if abs(num) >= 1e6:
        return f"{num/1e6:.{precision}f}M"
    elif abs(num) >= 1e3:
        return f"{num/1e3:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"


def validate_year_range(start_year: int, end_year: int) -> None:
    """Validate that year range is logical.
    
    Args:
        start_year: Starting year.
        end_year: Ending year.
        
    Raises:
        ValueError: If year range is invalid.
        
    Example:
        >>> validate_year_range(2020, 2023)  # Valid
        >>> validate_year_range(2023, 2020)  # Raises ValueError
    """
    if start_year > end_year:
        raise ValueError(f"Start year ({start_year}) must be <= end year ({end_year})")
    
    if start_year < 1900 or end_year > 2030:
        raise ValueError("Year range should be between 1900 and 2030")