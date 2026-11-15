"""Seeding helpers. Call set_seed once at the top of an entry point, before anything draws."""

import random
import numpy as np
import os
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic_torch: bool = True) -> Dict[str, bool]:
    """Returns which libraries were actually seeded, e.g. {"torch": False} where torch is absent.

    deterministic_torch costs some GPU speed but is what makes GNN runs bit-for-bit repeatable.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    seeded = {"python_random": True, "numpy": True, "torch": False}

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        seeded["torch"] = True
    except ImportError:
        pass

    logger.info(f"Random seed set to {seed} (seeded: {seeded})")
    return seeded


def get_random_state():
    return {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
    }


class Reproducible:
    """Seeds on entry, puts the caller's random state back on exit."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.original_state = None

    def __enter__(self):
        self.original_state = {
            "random": random.getstate(),
            "numpy": np.random.get_state(),
        }
        set_seed(self.seed, deterministic_torch=False)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        random.setstate(self.original_state["random"])
        np.random.set_state(self.original_state["numpy"])
