import datetime

import pytz


class TimeIterator:
    def __init__(self, start_year, end_year, start_month: int = 1, end_month: int = 1, snapshot_type='yearly'):
        self.start_year = start_year
        self.end_year = end_year

        self.start_month = start_month
        self.end_month = end_month

        assert snapshot_type in ['monthly', 'yearly'], "Invalid snapshot type. Must be 'monthly' or 'yearly'"

        if snapshot_type == "yearly":
            assert start_month == 1 and end_month == 1, "Invalid start_month for yearly snapshot"

        self.snapshot_type = snapshot_type
        self.current_year = start_year
        self.current_month = 1 if snapshot_type == 'yearly' else start_month

    def __iter__(self):
        return self

    def __next__(self):
        if self.snapshot_type == 'monthly':
            return self._next_monthly_snapshot()
        elif self.snapshot_type == 'yearly':
            return self._next_yearly_snapshot()
        else:
            raise ValueError("Unsupported snapshot type: " + self.snapshot_type)

    def _next_monthly_snapshot(self):
        if self.current_year > self.end_year or (self.current_year == self.end_year and self.current_month >= self.end_month):
            raise StopIteration

        start = datetime.datetime(self.current_year, self.current_month, 1, 0, 0, 0, tzinfo=pytz.utc)
        if self.current_month == 12:
            end = datetime.datetime(self.current_year + 1, 1, 1, 0, 0, 0, tzinfo=pytz.utc)
        else:
            end = datetime.datetime(self.current_year, self.current_month + 1, 1, 0, 0, 0, tzinfo=pytz.utc)

        self.current_month += 1
        if self.current_month > 12:
            self.current_month = 1
            self.current_year += 1

        return (start, end)

    def _next_yearly_snapshot(self):
        if self.current_year >= self.end_year:
            raise StopIteration

        # Combine all papers before 1995 into one single snapshot
        if self.current_year <= 1994:
            start = datetime.datetime(1985, 1, 1, 0, 0, 0, tzinfo=pytz.utc)
            end = datetime.datetime(1995, 1, 1, 0, 0, 0, tzinfo=pytz.utc)
            self.current_year = 1994

        else:
            start = datetime.datetime(self.current_year, 1, 1, 0, 0, 0, tzinfo=pytz.utc)
            end = datetime.datetime(self.current_year + 1, 1, 1, 0, 0, 0, tzinfo=pytz.utc)

        self.current_year += 1

        return (start, end)


def time_difference_in_days(date1, date2):
    """
    calculate time difference in days between two dates

    """
    date1 = datetime.strptime(date1, '%Y-%m-%d')
    date2 = datetime.strptime(date2, '%Y-%m-%d')
    return (date1 - date2).days


import os
import time


def get_file_times(filepath):
    # Getting the creation time (Windows) and last modified time (Unix)
    creation_time = os.path.getctime(filepath)
    modified_time = os.path.getmtime(filepath)

    # Converting time in seconds since the epoch to a readable format
    readable_creation = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(creation_time))
    readable_modified = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(modified_time))
    print(f"File: {filepath} Creation: {readable_creation}, Last Modified: {readable_modified}")

    return readable_creation, readable_modified

if __name__ == "__main__":
    # Example usage:
    # iterator = TimeIterator(2021, 2025, start_month=11, end_month=3, snapshot_type='monthly')
    iterator = TimeIterator(2021, 2025, start_month=1, end_month=1, snapshot_type='yearly')
    for start, end in iterator:
        print("Start:", start, "End:", end)
