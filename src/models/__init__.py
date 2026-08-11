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

# Temporarily disable GNN due to torch DLL issues
# from .gnn import GNNModel

__all__ = [
    "BaseModel",
    "RandomForestModel",
    "XGBoostModel",
    "LogisticRegressionModel",
    "SVMModel",
    "EnsembleModel",
    # "GNNModel",  # Temporarily disabled
    "HyperparameterOptimizer",
    "ModelTrainer",
]
