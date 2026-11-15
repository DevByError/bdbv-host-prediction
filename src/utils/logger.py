"""Logging setup plus a small helper for recording run metrics."""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "bdbv", log_level: str = "INFO", log_dir: Optional[Path] = None
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # drop anything attached by an earlier call, otherwise lines get doubled
    logger.handlers = []

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # file gets DEBUG, console stays at INFO
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
    def __init__(self, name: str = "bdbv", log_dir: Optional[Path] = None):
        self.logger = setup_logger(name, log_dir=log_dir)
        self.metrics = {}

    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        self.metrics[name] = {
            "value": value,
            "step": step,
            "timestamp": datetime.now().isoformat(),
        }
        self.logger.info(
            f"Metric: {name} = {value:.4f} {f'at step {step}' if step else ''}"
        )

    def log_params(self, params: dict):
        self.logger.info("Parameters:")
        for key, value in params.items():
            self.logger.info(f"  {key}: {value}")

    def log_data_stats(self, df, name: str = "dataset"):
        self.logger.info(f"{name} stats:")
        self.logger.info(f"  Samples: {len(df)}")
        self.logger.info(f"  Features: {len(df.columns)}")
        self.logger.info(f"  Missing values: {df.isnull().sum().sum()}")

    def get_metrics(self) -> dict:
        return self.metrics
