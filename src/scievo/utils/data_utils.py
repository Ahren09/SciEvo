"""Data loading and processing utilities for SciEvo package.

This module provides functions for loading, processing, and managing
the SciEvo dataset and related data files.
"""

import json
import os
import os.path as osp
import pickle
import re
import time
import datetime
from typing import List, Optional, Dict, Any, Union

import pandas as pd
import pytz
from datasets import Dataset
from tqdm import tqdm

from ..config.constants import ARXIV_SUBJECTS_LIST, DATE_FORMAT


def load_arXiv_data(
    data_dir: str, 
    start_year: Optional[int] = None, 
    start_month: Optional[int] = None, 
    end_year: Optional[int] = None, 
    end_month: Optional[int] = None
) -> pd.DataFrame:
    """Load arXiv metadata with optional date filtering.
    
    Args:
        data_dir: Directory containing the arXiv data files.
        start_year: Starting year for filtering (inclusive).
        start_month: Starting month for filtering (inclusive).
        end_year: Ending year for filtering (inclusive).  
        end_month: Ending month for filtering (inclusive).
        
    Returns:
        DataFrame containing the filtered arXiv metadata.
        
    Example:
        >>> df = load_arXiv_data("data/", start_year=2020, end_year=2023)
        >>> print(f"Loaded {len(df)} papers")
    """
    # This is a placeholder implementation
    # The actual implementation should load from the SciEvo dataset
    data_file = osp.join(data_dir, "arxiv_metadata.parquet")
    
    if osp.exists(data_file):
        df = pd.read_parquet(data_file)
    else:
        # Fallback to CSV or other formats
        csv_file = osp.join(data_dir, "arxiv_metadata.csv")
        if osp.exists(csv_file):
            df = pd.read_csv(csv_file)
        else:
            raise FileNotFoundError(f"No data files found in {data_dir}")
    
    # Apply date filtering if specified
    if any([start_year, start_month, end_year, end_month]):
        df = filter_by_date_range(df, start_year, start_month, end_year, end_month)
    
    return df


def filter_by_date_range(
    df: pd.DataFrame,
    start_year: Optional[int] = None,
    start_month: Optional[int] = None, 
    end_year: Optional[int] = None,
    end_month: Optional[int] = None,
    date_column: str = 'published'
) -> pd.DataFrame:
    """Filter DataFrame by date range.
    
    Args:
        df: Input DataFrame to filter.
        start_year: Starting year for filtering.
        start_month: Starting month for filtering.
        end_year: Ending year for filtering.
        end_month: Ending month for filtering.
        date_column: Name of the date column to filter on.
        
    Returns:
        Filtered DataFrame.
        
    Example:
        >>> filtered_df = filter_by_date_range(df, start_year=2020, end_year=2023)
    """
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found in DataFrame")
    
    # Convert date column to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    mask = pd.Series([True] * len(df))
    
    if start_year is not None:
        start_month = start_month or 1
        start_date = pd.Timestamp(start_year, start_month, 1)
        mask &= df[date_column] >= start_date
    
    if end_year is not None:
        end_month = end_month or 12
        # Get last day of the month
        if end_month == 12:
            end_date = pd.Timestamp(end_year + 1, 1, 1) - pd.Timedelta(days=1)
        else:
            end_date = pd.Timestamp(end_year, end_month + 1, 1) - pd.Timedelta(days=1)
        mask &= df[date_column] <= end_date
    
    return df[mask]


def load_dataset(file_path: str, format: str = 'auto') -> Union[pd.DataFrame, Dict, List]:
    """Load dataset from various file formats.
    
    Args:
        file_path: Path to the data file.
        format: File format ('auto', 'parquet', 'csv', 'json', 'pickle').
        
    Returns:
        Loaded data in appropriate format.
        
    Example:
        >>> data = load_dataset("data/papers.parquet")
        >>> print(type(data))
    """
    if format == 'auto':
        format = osp.splitext(file_path)[1].lower().lstrip('.')
    
    if format in ['parquet', 'pq']:
        return pd.read_parquet(file_path)
    elif format in ['csv']:
        return pd.read_csv(file_path)
    elif format in ['json', 'jsonl']:
        with open(file_path, 'r', encoding='utf-8') as f:
            if format == 'jsonl':
                return [json.loads(line) for line in f]
            else:
                return json.load(f)
    elif format in ['pickle', 'pkl']:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported format: {format}")


def save_processed_data(
    data: Union[pd.DataFrame, Dict, List], 
    file_path: str, 
    format: str = 'auto'
) -> None:
    """Save processed data to file.
    
    Args:
        data: Data to save.
        file_path: Output file path.
        format: File format ('auto', 'parquet', 'csv', 'json', 'pickle').
        
    Example:
        >>> save_processed_data(df, "outputs/processed_data.parquet")
    """
    # Ensure output directory exists
    os.makedirs(osp.dirname(file_path), exist_ok=True)
    
    if format == 'auto':
        format = osp.splitext(file_path)[1].lower().lstrip('.')
    
    if format in ['parquet', 'pq']:
        if isinstance(data, pd.DataFrame):
            data.to_parquet(file_path)
        else:
            raise ValueError("Parquet format requires pandas DataFrame")
    elif format in ['csv']:
        if isinstance(data, pd.DataFrame):
            data.to_csv(file_path, index=False)
        else:
            raise ValueError("CSV format requires pandas DataFrame")
    elif format in ['json']:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    elif format in ['jsonl']:
        with open(file_path, 'w', encoding='utf-8') as f:
            if isinstance(data, list):
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            else:
                raise ValueError("JSONL format requires list of objects")
    elif format in ['pickle', 'pkl']:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    else:
        raise ValueError(f"Unsupported format: {format}")


def merge_datasets(*datasets: pd.DataFrame, on: Optional[str] = None) -> pd.DataFrame:
    """Merge multiple datasets.
    
    Args:
        *datasets: Variable number of DataFrames to merge.
        on: Column name to merge on (if None, uses index).
        
    Returns:
        Merged DataFrame.
        
    Example:
        >>> merged = merge_datasets(df1, df2, on='paper_id')
    """
    if len(datasets) < 2:
        raise ValueError("At least 2 datasets required for merging")
    
    result = datasets[0]
    for df in datasets[1:]:
        if on is not None:
            result = result.merge(df, on=on, how='outer')
        else:
            result = result.merge(df, left_index=True, right_index=True, how='outer')
    
    return result


def validate_dataset(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate that dataset contains required columns.
    
    Args:
        df: DataFrame to validate.
        required_columns: List of required column names.
        
    Returns:
        True if all required columns are present.
        
    Raises:
        ValueError: If required columns are missing.
        
    Example:
        >>> validate_dataset(df, ['title', 'abstract', 'published'])
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataset by removing duplicates and handling missing values.
    
    Args:
        df: Input DataFrame to clean.
        
    Returns:
        Cleaned DataFrame.
        
    Example:
        >>> clean_df = clean_dataset(raw_df)
    """
    # Remove duplicates
    df_clean = df.drop_duplicates()
    
    # Handle missing values in text columns
    text_columns = df_clean.select_dtypes(include=['object']).columns
    for col in text_columns:
        df_clean[col] = df_clean[col].fillna('')
    
    # Handle missing values in numeric columns
    numeric_columns = df_clean.select_dtypes(include=['number']).columns
    for col in numeric_columns:
        df_clean[col] = df_clean[col].fillna(0)
    
    return df_clean


def sample_dataset(
    df: pd.DataFrame, 
    n_samples: Optional[int] = None, 
    frac: Optional[float] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """Sample from dataset for testing or analysis.
    
    Args:
        df: Input DataFrame to sample from.
        n_samples: Number of samples to take.
        frac: Fraction of dataset to sample.
        random_state: Random seed for reproducibility.
        
    Returns:
        Sampled DataFrame.
        
    Example:
        >>> sample_df = sample_dataset(df, n_samples=1000)
    """
    if n_samples is not None and frac is not None:
        raise ValueError("Specify either n_samples or frac, not both")
    
    if n_samples is not None:
        return df.sample(n=min(n_samples, len(df)), random_state=random_state)
    elif frac is not None:
        return df.sample(frac=frac, random_state=random_state)
    else:
        raise ValueError("Must specify either n_samples or frac")


def get_dataset_info(df: pd.DataFrame) -> Dict[str, Any]:
    """Get comprehensive information about a dataset.
    
    Args:
        df: DataFrame to analyze.
        
    Returns:
        Dictionary containing dataset statistics and information.
        
    Example:
        >>> info = get_dataset_info(df)
        >>> print(f"Dataset has {info['num_rows']} rows")
    """
    info = {
        'num_rows': len(df),
        'num_columns': len(df.columns),
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicated_rows': df.duplicated().sum()
    }
    
    # Add date range if date columns exist
    date_columns = df.select_dtypes(include=['datetime']).columns
    if len(date_columns) > 0:
        for col in date_columns:
            info[f'{col}_range'] = {
                'min': df[col].min(),
                'max': df[col].max()
            }
    
    return info