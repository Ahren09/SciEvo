from datetime import datetime
def time_difference_in_days(date1, date2):
    """
    calculate time difference in days between two dates

    """
    date1 = datetime.strptime(date1, '%Y-%m-%d')
    date2 = datetime.strptime(date2, '%Y-%m-%d')
    return (date1 - date2).days