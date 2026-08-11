"""
Logging configuration for production-grade debugging.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler


def setup_logging(log_dir: Path, name: str = "bdbv_host_prediction") -> logging.Logger:
    """
    Setup comprehensive logging with both file and console handlers.

    Args:
        log_dir: Directory to store log files
        name: Logger name

    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    logger.handlers = []

    # Create log directory
    log_dir.mkdir(parents=True, exist_ok=True)

    # Timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # File handler - all logs
    file_handler = RotatingFileHandler(
        log_dir / f"{name}_{timestamp}.log", maxBytes=10_000_000, backupCount=5  # 10MB
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    file_handler.setFormatter(file_format)

    # Console handler - info and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_format)

    # Error file handler - errors only
    error_handler = RotatingFileHandler(
        log_dir / f"errors_{timestamp}.log", maxBytes=1_000_000, backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_format)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.addHandler(error_handler)

    return logger


# Convenience function
def get_logger(name: str = "bdbv_host_prediction") -> logging.Logger:
    """Get the configured logger"""
    from config.config import config

    return setup_logging(config.paths.LOGS_DIR, name)
