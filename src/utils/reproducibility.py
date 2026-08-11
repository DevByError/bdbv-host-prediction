"""
Reproducibility utilities for consistent random seeds.
"""

import random
import numpy as np
import os


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_random_state():
    """
    Get current random state for reproducibility.

    Returns:
        dict: Current random states
    """
    return {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
    }


class Reproducible:
    """
    Context manager for reproducible code blocks.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.original_state = None

    def __enter__(self):
        self.original_state = {
            "random": random.getstate(),
            "numpy": np.random.get_state(),
        }
        set_seed(self.seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        random.setstate(self.original_state["random"])
        np.random.set_state(self.original_state["numpy"])
