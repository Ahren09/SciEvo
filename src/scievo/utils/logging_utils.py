"""Logging utilities for SciEvo package.

This module provides logging configuration and utilities for tracking
experiments and analysis progress.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any


def setup_logger(
    name: str, 
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Set up a logger with optional file output.
    
    Args:
        name: Name of the logger.
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional file path for logging output.
        format_string: Custom format string for log messages.
        
    Returns:
        Configured logger instance.
        
    Example:
        >>> logger = setup_logger("experiment", log_file="experiment.log")
        >>> logger.info("Starting analysis...")
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def configure_default_logging(level: int = logging.INFO) -> None:
    """Configure default logging for the entire application.
    
    Args:
        level: Logging level to set globally.
        
    Example:
        >>> configure_default_logging(logging.DEBUG)
    """
    logging.basicConfig(
        level=level, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def log_experiment_config(logger: logging.Logger, config: Dict[str, Any]) -> None:
    """Log experiment configuration parameters.
    
    Args:
        logger: Logger instance to use.
        config: Configuration dictionary to log.
        
    Example:
        >>> config = {"model": "gcn", "epochs": 50, "lr": 0.01}
        >>> log_experiment_config(logger, config)
    """
    logger.info("=" * 50)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("=" * 50)
    
    for key, value in config.items():
        logger.info(f"{key}: {value}")
    
    logger.info("=" * 50)


def log_dataset_info(logger: logging.Logger, dataset_stats: Dict[str, Any]) -> None:
    """Log dataset information and statistics.
    
    Args:
        logger: Logger instance to use.
        dataset_stats: Dictionary containing dataset statistics.
        
    Example:
        >>> stats = {"num_papers": 10000, "num_citations": 50000}
        >>> log_dataset_info(logger, stats)
    """
    logger.info("-" * 30)
    logger.info("DATASET INFORMATION")
    logger.info("-" * 30)
    
    for key, value in dataset_stats.items():
        logger.info(f"{key}: {value}")
    
    logger.info("-" * 30)


def log_model_performance(
    logger: logging.Logger, 
    metrics: Dict[str, float],
    epoch: Optional[int] = None
) -> None:
    """Log model performance metrics.
    
    Args:
        logger: Logger instance to use.
        metrics: Dictionary of performance metrics.
        epoch: Optional epoch number for training logs.
        
    Example:
        >>> metrics = {"accuracy": 0.95, "loss": 0.1}
        >>> log_model_performance(logger, metrics, epoch=10)
    """
    epoch_str = f"Epoch {epoch} - " if epoch is not None else ""
    
    metric_strs = [f"{key}: {value:.4f}" for key, value in metrics.items()]
    logger.info(f"{epoch_str}Performance: {' | '.join(metric_strs)}")


def create_experiment_logger(
    experiment_name: str,
    output_dir: str = "outputs/logs"
) -> logging.Logger:
    """Create a logger for a specific experiment.
    
    Args:
        experiment_name: Name of the experiment.
        output_dir: Directory to store log files.
        
    Returns:
        Configured logger for the experiment.
        
    Example:
        >>> logger = create_experiment_logger("citation_analysis")
        >>> logger.info("Starting citation analysis...")
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{experiment_name}_{timestamp}.log"
    log_path = os.path.join(output_dir, log_filename)
    
    return setup_logger(
        name=experiment_name,
        log_file=log_path,
        format_string='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


class ProgressLogger:
    """Logger for tracking progress of long-running operations.
    
    Attributes:
        logger: The underlying logger instance.
        total: Total number of items to process.
        current: Current number of processed items.
        log_interval: Interval for logging progress updates.
    """
    
    def __init__(
        self, 
        logger: logging.Logger, 
        total: int, 
        log_interval: int = 100
    ):
        """Initialize the progress logger.
        
        Args:
            logger: Logger instance to use.
            total: Total number of items to process.
            log_interval: How often to log progress updates.
        """
        self.logger = logger
        self.total = total
        self.current = 0
        self.log_interval = log_interval
        self.start_time = datetime.now()
    
    def update(self, increment: int = 1) -> None:
        """Update progress counter and log if necessary.
        
        Args:
            increment: Number of items processed in this update.
        """
        self.current += increment
        
        if self.current % self.log_interval == 0 or self.current == self.total:
            percentage = (self.current / self.total) * 100
            elapsed_time = datetime.now() - self.start_time
            
            self.logger.info(
                f"Progress: {self.current}/{self.total} ({percentage:.1f}%) "
                f"- Elapsed: {elapsed_time}"
            )
    
    def finish(self) -> None:
        """Log completion message."""
        total_time = datetime.now() - self.start_time
        self.logger.info(f"Completed processing {self.total} items in {total_time}")