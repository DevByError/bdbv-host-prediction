"""
BDBV Host Prediction - Source Package
"""

# Import modules
from . import data
from . import features
from . import models
from . import evaluation
from . import explainability
from . import visualization
from . import pipelines

# Import utils conditionally (handles torch import)
try:
    from . import utils
except ImportError:
    # Create a placeholder utils module
    import sys
    from types import ModuleType

    utils = ModuleType("utils")
    sys.modules["src.utils"] = utils

__all__ = [
    "data",
    "features",
    "models",
    "evaluation",
    "explainability",
    "visualization",
    "pipelines",
    "utils",
]
