"""
Model training with cross-validation and hyperparameter tuning.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from sklearn.base import clone
from sklearn.model_selection import train_test_split
import logging

from src.models.hyperparameter import HyperparameterOptimizer
from src.models.classical import (
    RandomForestModel,
    XGBoostModel,
    LogisticRegressionModel,
)

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Complete model training pipeline with optimization.
    """

    def __init__(
        self,
        model_name: str = "random_forest",
        optimize: bool = True,
        n_trials: int = 50,
        random_state: int = 42,
    ):
        """
        Initialize model trainer.

        Args:
            model_name: Model to train
            optimize: Whether to perform hyperparameter optimization
            n_trials: Number of optimization trials
            random_state: Random seed
        """
        self.model_name = model_name
        self.optimize = optimize
        self.n_trials = n_trials
        self.random_state = random_state

        self.model = None
        self.best_params = None
        self.training_history = None

    def train(self, X: np.ndarray, y: np.ndarray, val_size: float = 0.2) -> Dict:
        """
        Train model with optional optimization.

        Args:
            X: Features
            y: Labels
            val_size: Validation set size

        Returns:
            Dict: Training results
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_size, random_state=self.random_state, stratify=y
        )

        # Get model class and parameter grid
        model_class, param_grid = self._get_model_config()

        if self.optimize and param_grid:
            # Optimize hyperparameters
            logger.info(f"Optimizing {self.model_name}...")

            optimizer = HyperparameterOptimizer(
                model_class, param_grid, scoring="f1", random_state=self.random_state
            )

            result = optimizer.optuna_optimize(X_train, y_train, n_trials=self.n_trials)

            self.best_params = optimizer.best_params
            self.model = optimizer.best_model

        else:
            # Train with default parameters
            logger.info(f"Training {self.model_name} with default parameters...")

            self.model = model_class()
            self.model.fit(X_train, y_train)
            self.best_params = self.model.get_params()

        # Evaluate on validation set
        y_pred = self.model.predict(X_val)

        from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

        results = {
            "model_name": self.model_name,
            "best_params": self.best_params,
            "val_accuracy": accuracy_score(y_val, y_pred),
            "val_f1": f1_score(y_val, y_pred),
        }

        if hasattr(self.model, "predict_proba"):
            y_prob = self.model.predict_proba(X_val)[:, 1]
            results["val_roc_auc"] = roc_auc_score(y_val, y_prob)

        logger.info(f"Validation F1: {results['val_f1']:.4f}")

        return results

    def _get_model_config(self) -> Tuple:
        """Get model class and parameter grid."""
        if self.model_name == "random_forest":

            class RFWithParams(RandomForestModel):
                def __init__(self, **kwargs):
                    super().__init__(kwargs)

            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }

            return RandomForestModel, param_grid

        elif self.model_name == "xgboost":

            class XGBWithParams(XGBoostModel):
                def __init__(self, **kwargs):
                    super().__init__(kwargs)

            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.3],
                "subsample": [0.6, 0.8, 1.0],
            }

            return XGBoostModel, param_grid

        elif self.model_name == "logistic_regression":

            class LRWithParams(LogisticRegressionModel):
                def __init__(self, **kwargs):
                    super().__init__(kwargs)

            param_grid = {
                "C": [0.01, 0.1, 1.0, 10.0],
                "solver": ["lbfgs", "liblinear"],
                "penalty": ["l2"],
            }

            return LogisticRegressionModel, param_grid

        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def cross_validate(self, X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> Dict:
        """
        Perform cross-validation.

        Args:
            X: Features
            y: Labels
            n_folds: Number of folds

        Returns:
            Dict: CV results
        """
        from sklearn.model_selection import cross_val_score, StratifiedKFold

        if self.model is None:
            raise ValueError("Model must be trained first")

        cv = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=self.random_state
        )

        scores = cross_val_score(self.model, X, y, cv=cv, scoring="f1")

        return {
            "mean_score": scores.mean(),
            "std_score": scores.std(),
            "scores": scores.tolist(),
            "n_folds": n_folds,
        }

    def save_model(self, path: str):
        """Save trained model."""
        if self.model is None:
            raise ValueError("Model must be trained first")

        import joblib

        joblib.dump(
            {
                "model": self.model,
                "best_params": self.best_params,
                "model_name": self.model_name,
            },
            path,
        )

        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load trained model."""
        import joblib

        data = joblib.load(path)

        self.model = data["model"]
        self.best_params = data["best_params"]
        self.model_name = data["model_name"]

        logger.info(f"Model loaded from {path}")
