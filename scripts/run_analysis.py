#!/usr/bin/env python3
"""Main script for running SciEvo analysis.

This script provides a command-line interface for running various
analysis tasks on the SciEvo dataset.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scievo.config.settings import parse_args
from scievo.utils.logging_utils import setup_logger
from scievo.utils.misc_utils import set_random_seed


def main():
    """Main entry point for SciEvo analysis."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logger(
        "scievo_analysis",
        log_file=os.path.join(args.output_dir, "analysis.log") if args.output_dir else None
    )
    
    logger.info("Starting SciEvo Analysis")
    logger.info(f"Arguments: {vars(args)}")
    
    # Set random seed for reproducibility
    set_random_seed(args.random_seed)
    
    try:
        # Import analysis modules (lazy import to avoid circular dependencies)
        from scievo.analysis import run_analysis_pipeline
        
        # Run the analysis pipeline
        results = run_analysis_pipeline(args)
        
        logger.info("Analysis completed successfully")
        logger.info(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise
    
    return results


if __name__ == "__main__":
    main()