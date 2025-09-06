"""SciEvo: A Longitudinal Scientometric Dataset Analysis Package.

This package provides tools and utilities for analyzing academic literature
evolution using the SciEvo dataset, which spans over 30 years of arXiv publications.
"""

__version__ = "0.1.0"
__author__ = "SciEvo Team"

from . import config
from . import data
from . import models  
from . import analysis
from . import visualization
from . import utils

__all__ = [
    "config",
    "data", 
    "models",
    "analysis",
    "visualization",
    "utils"
]