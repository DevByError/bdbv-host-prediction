"""
Custom logger with structured logging.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "bdbv", log_level: str = "INFO", log_dir: Optional[Path] = None
) -> logging.Logger:
    """
    Setup logger with file and console handlers.

    Args:
        name: Logger name
        log_level: Logging level
        log_dir: Directory for log files

    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(log_dir / f"{name}_{timestamp}.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class Logger:
    """
    Structured logger for ML workflows.
    """

    def __init__(self, name: str = "bdbv", log_dir: Optional[Path] = None):
        self.logger = setup_logger(name, log_dir=log_dir)
        self.metrics = {}

    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a metric."""
        self.metrics[name] = {
            "value": value,
            "step": step,
            "timestamp": datetime.now().isoformat(),
        }
        self.logger.info(
            f"Metric: {name} = {value:.4f} {f'at step {step}' if step else ''}"
        )

    def log_params(self, params: dict):
        """Log parameters."""
        self.logger.info("Parameters:")
        for key, value in params.items():
            self.logger.info(f"  {key}: {value}")

    def log_data_stats(self, df, name: str = "dataset"):
        """Log dataset statistics."""
        self.logger.info(f"{name} stats:")
        self.logger.info(f"  Samples: {len(df)}")
        self.logger.info(f"  Features: {len(df.columns)}")
        self.logger.info(f"  Missing values: {df.isnull().sum().sum()}")

    def get_metrics(self) -> dict:
        """Get all logged metrics."""
        return self.metrics
