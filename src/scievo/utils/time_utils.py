"""Time-related utilities for SciEvo package.

This module provides utilities for handling temporal data, date parsing,
and time-based iteration for longitudinal analysis.
"""

import datetime
import os
import time
from typing import Tuple, Iterator

import pytz


class TimeIterator:
    """Iterator for generating time windows for temporal analysis.
    
    Supports both monthly and yearly snapshots for longitudinal studies
    of academic literature evolution.
    
    Attributes:
        start_year: Starting year for iteration.
        end_year: Ending year for iteration.
        start_month: Starting month (for monthly snapshots).
        end_month: Ending month (for monthly snapshots).
        snapshot_type: Type of snapshot ('monthly' or 'yearly').
    """
    
    def __init__(
        self, 
        start_year: int, 
        end_year: int, 
        start_month: int = 1, 
        end_month: int = 1, 
        snapshot_type: str = 'yearly'
    ):
        """Initialize the TimeIterator.
        
        Args:
            start_year: Starting year for the iteration.
            end_year: Ending year for the iteration.
            start_month: Starting month (only used for monthly snapshots).
            end_month: Ending month (only used for monthly snapshots).
            snapshot_type: Type of snapshot, either 'monthly' or 'yearly'.
            
        Raises:
            AssertionError: If snapshot_type is invalid or monthly parameters
                are provided for yearly snapshots.
            
        Example:
            >>> iterator = TimeIterator(2020, 2023, snapshot_type='yearly')
            >>> for start, end in iterator:
            ...     print(f"{start.year} - {end.year}")
        """
        self.start_year = start_year
        self.end_year = end_year
        self.start_month = start_month
        self.end_month = end_month

        assert snapshot_type in ['monthly', 'yearly'], \
            "Invalid snapshot type. Must be 'monthly' or 'yearly'"

        if snapshot_type == "yearly":
            assert start_month == 1 and end_month == 1, \
                "Invalid start_month for yearly snapshot"

        self.snapshot_type = snapshot_type
        self.current_year = start_year
        self.current_month = 1 if snapshot_type == 'yearly' else start_month

    def __iter__(self) -> Iterator[Tuple[datetime.datetime, datetime.datetime]]:
        """Return iterator object."""
        return self

    def __next__(self) -> Tuple[datetime.datetime, datetime.datetime]:
        """Get next time window.
        
        Returns:
            Tuple of (start_datetime, end_datetime) for the next time window.
            
        Raises:
            StopIteration: When iteration is complete.
        """
        if self.snapshot_type == 'monthly':
            return self._next_monthly_snapshot()
        elif self.snapshot_type == 'yearly':
            return self._next_yearly_snapshot()
        else:
            raise ValueError("Unsupported snapshot type: " + self.snapshot_type)

    def _next_monthly_snapshot(self) -> Tuple[datetime.datetime, datetime.datetime]:
        """Generate next monthly snapshot window.
        
        Returns:
            Tuple of start and end datetime objects for the month.
            
        Raises:
            StopIteration: When monthly iteration is complete.
        """
        if (self.current_year > self.end_year or 
            (self.current_year == self.end_year and self.current_month >= self.end_month)):
            raise StopIteration

        start = datetime.datetime(
            self.current_year, self.current_month, 1, 0, 0, 0, tzinfo=pytz.utc
        )
        
        if self.current_month == 12:
            end = datetime.datetime(
                self.current_year + 1, 1, 1, 0, 0, 0, tzinfo=pytz.utc
            )
        else:
            end = datetime.datetime(
                self.current_year, self.current_month + 1, 1, 0, 0, 0, tzinfo=pytz.utc
            )

        self.current_month += 1
        if self.current_month > 12:
            self.current_month = 1
            self.current_year += 1

        return (start, end)

    def _next_yearly_snapshot(self) -> Tuple[datetime.datetime, datetime.datetime]:
        """Generate next yearly snapshot window.
        
        Special handling for pre-1995 data, which is combined into a single snapshot.
        
        Returns:
            Tuple of start and end datetime objects for the year.
            
        Raises:
            StopIteration: When yearly iteration is complete.
        """
        if self.current_year >= self.end_year:
            raise StopIteration

        # Combine all papers before 1995 into one single snapshot
        if self.current_year <= 1994:
            start = datetime.datetime(1985, 1, 1, 0, 0, 0, tzinfo=pytz.utc)
            end = datetime.datetime(1995, 1, 1, 0, 0, 0, tzinfo=pytz.utc)
            self.current_year = 1994
        else:
            start = datetime.datetime(
                self.current_year, 1, 1, 0, 0, 0, tzinfo=pytz.utc
            )
            end = datetime.datetime(
                self.current_year + 1, 1, 1, 0, 0, 0, tzinfo=pytz.utc
            )

        self.current_year += 1
        return (start, end)


def time_difference_in_days(date1: str, date2: str) -> int:
    """Calculate time difference in days between two dates.
    
    Args:
        date1: First date in 'YYYY-MM-DD' format.
        date2: Second date in 'YYYY-MM-DD' format.
        
    Returns:
        Number of days between the dates (date1 - date2).
        
    Example:
        >>> days = time_difference_in_days('2023-01-15', '2023-01-10')
        >>> print(days)
        5
    """
    date1_obj = datetime.datetime.strptime(date1, '%Y-%m-%d')
    date2_obj = datetime.datetime.strptime(date2, '%Y-%m-%d')
    return (date1_obj - date2_obj).days


def get_file_times(filepath: str) -> Tuple[str, str]:
    """Get file creation and modification times.
    
    Args:
        filepath: Path to the file to examine.
        
    Returns:
        Tuple of (creation_time, modified_time) as readable strings.
        
    Example:
        >>> creation, modified = get_file_times('/path/to/file.txt')
        >>> print(f"Created: {creation}, Modified: {modified}")
    """
    # Getting the creation time (Windows) and last modified time (Unix)
    creation_time = os.path.getctime(filepath)
    modified_time = os.path.getmtime(filepath)

    # Converting time in seconds since the epoch to a readable format
    readable_creation = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(creation_time)
    )
    readable_modified = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(modified_time)
    )
    
    print(f"File: {filepath} Creation: {readable_creation}, "
          f"Last Modified: {readable_modified}")

    return readable_creation, readable_modified


def parse_date(date_str: str, format_str: str = "%Y-%m-%d") -> datetime.datetime:
    """Parse date string into datetime object.
    
    Args:
        date_str: Date string to parse.
        format_str: Format string for parsing (default: "%Y-%m-%d").
        
    Returns:
        Parsed datetime object.
        
    Example:
        >>> dt = parse_date("2023-01-15")
        >>> print(dt.year)
        2023
    """
    return datetime.datetime.strptime(date_str, format_str)


def get_year_from_date(date_str: str) -> int:
    """Extract year from date string.
    
    Args:
        date_str: Date string in 'YYYY-MM-DD' format.
        
    Returns:
        Year as integer.
        
    Example:
        >>> year = get_year_from_date("2023-01-15")
        >>> print(year)
        2023
    """
    return parse_date(date_str).year


def filter_by_year_range(data, date_column: str, start_year: int, end_year: int):
    """Filter data by year range.
    
    Args:
        data: Data object with date column (e.g., DataFrame).
        date_column: Name of the date column.
        start_year: Starting year (inclusive).
        end_year: Ending year (inclusive).
        
    Returns:
        Filtered data within the specified year range.
        
    Example:
        >>> filtered_df = filter_by_year_range(df, 'published', 2020, 2023)
    """
    if hasattr(data, 'loc'):  # DataFrame-like object
        years = data[date_column].apply(get_year_from_date)
        return data.loc[(years >= start_year) & (years <= end_year)]
    else:
        raise ValueError("Unsupported data type for filtering")


if __name__ == "__main__":
    # Example usage
    iterator = TimeIterator(2021, 2025, start_month=1, end_month=1, snapshot_type='yearly')
    for start, end in iterator:
        print("Start:", start, "End:", end)