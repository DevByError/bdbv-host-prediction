"""Config objects, logging setup and global seeding."""

import logging
from typing import Optional

from .config import Config, config
from .logging_config import setup_logging, get_logger

logger = logging.getLogger(__name__)


def set_all_seeds(seed: int = 42) -> None:
    """Seed python, numpy, torch and tf in one go. Call this first thing."""
    import random
    import numpy as np
    import os

    random.seed(seed)
    logger.info(f"Set Python random.seed({seed})")

    np.random.seed(seed)
    logger.info(f"Set numpy.random.seed({seed})")

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # deterministic cudnn costs us some speed, worth it here
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        logger.info(f"Set torch.manual_seed({seed})")
        logger.info("Enabled deterministic behavior in PyTorch (may reduce performance)")
    except ImportError:
        logger.debug("PyTorch not available, skipping torch seed setup")

    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Set PYTHONHASHSEED={seed}")

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        logger.info(f"Set tensorflow.random.set_seed({seed})")
    except ImportError:
        logger.debug("TensorFlow not available, skipping TF seed setup")

    # scipy rides on the numpy generator, nothing extra to do
    logger.info(f"Reproducibility: All random seeds set to {seed}")
    logger.info("IMPORTANT: For complete reproducibility, also:")
    logger.info("  1. Run on same hardware/OS")
    logger.info("  2. Use same library versions (see requirements.txt)")
    logger.info("  3. Document CUDA/GPU version if using GPU")
    logger.info("  4. Consider using Docker for exact environment")


def get_reproducibility_info() -> dict:
    """Seed plus library/hardware versions, for dumping into run metadata."""
    import sys
    import numpy as np

    info = {
        "seed": config.seed,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "random_state_set": True,
    }

    try:
        import torch
        info["pytorch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["cuda_device_count"] = torch.cuda.device_count()
    except ImportError:
        info["pytorch_version"] = "Not installed"

    try:
        import tensorflow as tf
        info["tensorflow_version"] = tf.__version__
    except ImportError:
        info["tensorflow_version"] = "Not installed"

    return info


__all__ = [
    "Config",
    "config",
    "setup_logging",
    "get_logger",
    "set_all_seeds",
    "get_reproducibility_info",
]
