from collections import Counter
from typing import Dict, List, Union

import numpy as np

import const


def gini(x):
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x) ** 2 * np.mean(x))


def simpsons_diversity_index(proportions):
    """Calculate Simpson's Diversity Index, a.k.a. Gini-Simpson Index."""
    return 1 - np.sum(proportions ** 2)


def shannons_diversity_index(proportions):
    """Calculate Shannon's Diversity Index."""
    return -np.sum(proportions * np.log(proportions + 1e-10))  # adding small value to avoid log(0)


def gini_simpson_index(proportions):
    """Calculate Gini-Simpson Index."""
    return np.sum(proportions * (1 - proportions))


def herfindahl_hirschman_index(proportions):
    """Calculate Herfindahl-Hirschman Index, i.e. 1 - Simpson's Diversity Index"""
    return np.sum(proportions ** 2)


def normalized_entropy(proportions):
    """Calculate Normalized Entropy."""
    n = len(proportions)
    if n == 1:  # All papers in one field
        return 0
    log_n = np.log(n)
    return -np.sum(proportions * np.log(proportions + 1e-10)) / log_n

def berger_parker_index(categories):
    """Calculate Berger-Parker Index for a list of categories."""

    if isinstance(categories, dict):
        category_counts = Counter(categories)

    elif isinstance(categories, dict):
        category_counts = categories

    else:
        raise NotImplementedError("Input must be a list or a dictionary.")

    max_abundance = max(category_counts.values())
    total = sum(category_counts.values())
    berger_parker = max_abundance / total
    return berger_parker



def calculate_citation_diversity(inputs: Union[List, Dict]):

    if isinstance(inputs, list):

        count = {subject: 0 for subject in const.SEMANTIC_SCHOLAR_STATS.keys()}
        for subject_areas_of_one_reference in inputs:
            for subject in subject_areas_of_one_reference:
                count[subject] += 1 / len(subject_areas_of_one_reference)

    elif isinstance(inputs, dict):
        count = {subject: inputs.get(subject, 0) for subject in const.SEMANTIC_SCHOLAR_STATS.keys()}


    if sum(count.values()) == 0:
        return None

    proportions = np.array(list(count.values())) / sum(count.values())

    simpsons_DI = simpsons_diversity_index(proportions)
    shannons_DI = shannons_diversity_index(proportions)
    # GSI = gini_simpson_index(proportions)
    # HHI = herfindahl_hirschman_index(proportions)

    NE = normalized_entropy(proportions)
    BPI = berger_parker_index(count)

    G = gini(proportions)

    return {
        'simpsons_diversity_index': simpsons_DI,
        'shannons_diversity_index': shannons_DI,
        'normalized_entropy': NE,
        "berger_parker_index": BPI,
        "gini": G,
    }

if __name__ == "__main__":
    fields = ['n/a', 'Medicine', 'Biology', 'Physics', 'Engineering', 'Computer Science']
    counts = np.array([10, 15, 5, 20, 50])  # example counts for some fields
    total = counts.sum()
    proportions = counts / total

    print("Simpson's Diversity Index:", simpsons_diversity_index(proportions))
    print("Shannon's Diversity Index:", shannons_diversity_index(proportions))
    print("Gini-Simpson Index:", gini_simpson_index(proportions))
    print("Herfindahl-Hirschman Index:", herfindahl_hirschman_index(proportions))
    print("Normalized Entropy:", normalized_entropy(proportions))
