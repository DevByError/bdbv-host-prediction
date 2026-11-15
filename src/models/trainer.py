"""Train one model, optionally tuning it first."""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import logging

from src.models.hyperparameter import HyperparameterOptimizer
from src.models.classical import (
    RandomForestModel,
    XGBoostModel,
    LogisticRegressionModel,
)

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Tune + fit one model, returned as a BaseModel wrapper.

    Search runs on the raw estimator class and only the resulting params get
    wrapped, since HyperparameterOptimizer can't take the wrappers.
    """

    def __init__(
        self,
        model_name: str = "random_forest",
        optimize: bool = True,
        n_trials: int = 50,
        random_state: int = 42,
    ):
        valid_names = {"random_forest", "xgboost", "logistic_regression"}
        if model_name not in valid_names:
            raise ValueError(f"model_name must be one of {valid_names}, got {model_name}")

        self.model_name = model_name
        self.optimize = optimize
        self.n_trials = n_trials
        self.random_state = random_state

        self.model = None  # BaseModel wrapper, set by train()
        self.best_params = None
        self.training_history = None

    def train(self, X: np.ndarray, y: np.ndarray, val_size: float = 0.2) -> Dict:
        """Returns val_accuracy/val_f1 (and val_roc_auc when probabilities exist)."""
        if len(X) == 0 or len(X) != len(y):
            raise ValueError("X and y must be non-empty and the same length")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_size, random_state=self.random_state, stratify=y
        )

        raw_model_class, param_grid, base_model_class = self._get_model_config()

        if self.optimize and param_grid:
            logger.info(f"Optimizing {self.model_name}...")

            optimizer = HyperparameterOptimizer(
                raw_model_class,
                param_grid,
                scoring="f1",
                random_state=self.random_state,
            )

            optimizer.optuna_optimize(X_train, y_train, n_trials=self.n_trials)

            self.best_params = optimizer.best_params
            self.model = optimizer.to_base_model(base_model_class)
            self.model.fit(X_train, y_train)

        else:
            logger.info(f"Training {self.model_name} with default parameters...")

            self.model = base_model_class()
            self.model.fit(X_train, y_train)
            self.best_params = self.model.get_params()

        y_pred = self.model.predict(X_val)

        from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

        results = {
            "model_name": self.model_name,
            "best_params": self.best_params,
            "val_accuracy": accuracy_score(y_val, y_pred),
            "val_f1": f1_score(y_val, y_pred, zero_division=0),
        }

        if hasattr(self.model, "predict_proba"):
            y_prob = self.model.predict_proba(X_val)[:, 1]
            results["val_roc_auc"] = roc_auc_score(y_val, y_prob)

        logger.info(f"Validation F1: {results['val_f1']:.4f}")

        return results

    def _get_model_config(self) -> Tuple:
        """(raw estimator class, search space, BaseModel subclass to wrap it in)."""
        if self.model_name == "random_forest":
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }
            return RandomForestClassifier, param_grid, RandomForestModel

        elif self.model_name == "xgboost":
            if not XGB_AVAILABLE:
                raise ImportError("xgboost is not installed")
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.3],
                "subsample": [0.6, 0.8, 1.0],
            }
            return xgb.XGBClassifier, param_grid, XGBoostModel

        elif self.model_name == "logistic_regression":
            param_grid = {
                "C": [0.01, 0.1, 1.0, 10.0],
                "solver": ["lbfgs"],
                "penalty": ["l2"],
            }
            return LogisticRegression, param_grid, LogisticRegressionModel

        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def cross_validate(self, X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> Dict:
        # CV refits clones per fold, so this re-tunes nothing - config only
        if self.model is None:
            raise ValueError("Model must be trained first")

        from src.evaluation.cross_validation import CrossValidator

        cv = CrossValidator(n_folds=n_folds, random_state=self.random_state)
        return cv.cross_validate(self.model, X, y, scoring="f1")

    def save_model(self, path):
        if self.model is None:
            raise ValueError("Model must be trained first")
        from pathlib import Path

        self.model.save(Path(path))

    def load_model(self, path):
        if self.model is None:
            _, _, base_model_class = self._get_model_config()
            self.model = base_model_class()
        from pathlib import Path

        self.model.load(Path(path))
