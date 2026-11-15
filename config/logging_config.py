"""Logger setup: rotating files plus console."""

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler


def setup_logging(log_dir: Path, name: str = "bdbv_host_prediction") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # drop anything a previous call left behind
    logger.handlers = []

    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    file_handler = RotatingFileHandler(
        log_dir / f"{name}_{timestamp}.log", maxBytes=10_000_000, backupCount=5  # 10MB
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    file_handler.setFormatter(file_format)

    # console stays at INFO, debug output is unreadable otherwise
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_format)

    error_handler = RotatingFileHandler(
        log_dir / f"errors_{timestamp}.log", maxBytes=1_000_000, backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_format)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.addHandler(error_handler)

    return logger


def get_logger(name: str = "bdbv_host_prediction") -> logging.Logger:
    from config.config import config

    return setup_logging(config.paths.LOGS_DIR, name)
