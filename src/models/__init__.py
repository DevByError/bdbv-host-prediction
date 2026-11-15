"""
Machine learning models for BDBV Host Prediction.
"""

from .base import BaseModel
from .classical import (
    RandomForestModel,
    XGBoostModel,
    LogisticRegressionModel,
    SVMModel,
)
from .ensemble import EnsembleModel
from .hyperparameter import HyperparameterOptimizer
from .trainer import ModelTrainer

# GNNModel requires torch + torch_geometric, both optional heavy
# dependencies. Importing gnn.py itself never raises (it guards its own
# torch import internally and only raises ImportError when GNNModel is
# instantiated), so this import is always safe.
from .gnn import GNNModel, TORCH_AVAILABLE

__all__ = [
    "BaseModel",
    "RandomForestModel",
    "XGBoostModel",
    "LogisticRegressionModel",
    "SVMModel",
    "EnsembleModel",
    "GNNModel",
    "TORCH_AVAILABLE",
    "HyperparameterOptimizer",
    "ModelTrainer",
]
