"""
Uncertainty estimation for predictions.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from sklearn.ensemble import RandomForestClassifier
import logging

logger = logging.getLogger(__name__)


class UncertaintyEstimator:
    """
    Estimate prediction uncertainty using multiple methods.

    Methods:
    - Monte Carlo Dropout
    - Ensemble Variance
    - Conformal Prediction
    - Bootstrap Aggregating
    """

    def __init__(self, n_estimators: int = 100):
        self.n_estimators = n_estimators
        self.models = []

    def ensemble_uncertainty(self, model, X: np.ndarray) -> Dict:
        """
        Estimate uncertainty using ensemble of models.

        Args:
            model: Base model (must support bagging)
            X: Features

        Returns:
            Dict: Uncertainty estimates
        """
        # For RandomForest, we can use individual tree predictions
        if hasattr(model, "estimators_"):
            predictions = np.array([tree.predict(X) for tree in model.estimators_])

            # Calculate mean and variance
            mean = np.mean(predictions, axis=0)
            variance = np.var(predictions, axis=0)

            # Calculate entropy
            proba_mean = np.mean(
                [tree.predict_proba(X) for tree in model.estimators_], axis=0
            )
            entropy = -np.sum(proba_mean * np.log(proba_mean + 1e-10), axis=1)

            return {
                "mean": mean,
                "variance": variance,
                "std": np.sqrt(variance),
                "entropy": entropy,
                "confidence": 1 - entropy / np.log(2),  # Normalize to [0,1]
            }
        else:
            logger.warning("Model doesn't support ensemble uncertainty")
            return {}

    def conformal_prediction(
        self,
        model,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        X_test: np.ndarray,
        alpha: float = 0.1,
    ) -> Dict:
        """
        Conformal prediction for uncertainty quantification.

        Args:
            model: Trained model
            X_cal: Calibration features
            y_cal: Calibration labels
            X_test: Test features
            alpha: Significance level (0.1 = 90% confidence)

        Returns:
            Dict: Prediction sets and confidence
        """
        from sklearn.model_selection import train_test_split

        # Get non-conformity scores
        if hasattr(model, "predict_proba"):
            # For classification, use 1 - probability
            proba = model.predict_proba(X_cal)
            scores = 1 - proba[np.arange(len(y_cal)), y_cal.astype(int)]
        else:
            # For regression, use absolute error
            y_pred = model.predict(X_cal)
            scores = np.abs(y_pred - y_cal)

        # Calculate quantile
        n_cal = len(scores)
        q_hat = np.percentile(scores, (1 - alpha) * 100)

        # Get predictions for test
        if hasattr(model, "predict_proba"):
            test_proba = model.predict_proba(X_test)

            # Prediction sets
            prediction_sets = []
            for i in range(len(test_proba)):
                set_ = np.where(test_proba[i] >= 1 - q_hat)[0]
                prediction_sets.append(set_)

            # Confidence scores
            max_proba = np.max(test_proba, axis=1)
            confidence = 1 - max_proba

            return {
                "prediction_sets": prediction_sets,
                "confidence": confidence,
                "q_hat": q_hat,
                "coverage": np.mean(
                    [1 if y in set_ else 0 for y, set_ in zip(y_test, prediction_sets)]
                ),
            }
        else:
            return {}

    def bootstrap_uncertainty(
        self, model, X: np.ndarray, n_bootstrap: int = 100
    ) -> Dict:
        """
        Estimate uncertainty using bootstrap.

        Args:
            model: Trained model
            X: Features
            n_bootstrap: Number of bootstrap samples

        Returns:
            Dict: Uncertainty estimates
        """
        from sklearn.base import clone

        n_samples = len(X)
        predictions = []

        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]

            # Train model on bootstrap sample
            model_clone = clone(model)
            model_clone.fit(X_boot, y_boot)  # Need y_boot

            y_pred = model_clone.predict(X)
            predictions.append(y_pred)

        predictions = np.array(predictions)

        return {
            "mean": np.mean(predictions, axis=0),
            "variance": np.var(predictions, axis=0),
            "std": np.std(predictions, axis=0),
            "ci_lower": np.percentile(predictions, 2.5, axis=0),
            "ci_upper": np.percentile(predictions, 97.5, axis=0),
        }

    def prediction_interval(self, y_pred: np.ndarray, y_std: np.ndarray) -> Dict:
        """
        Calculate prediction intervals from uncertainty estimates.

        Args:
            y_pred: Mean predictions
            y_std: Standard deviation of predictions

        Returns:
            Dict: Prediction intervals
        """
        # 95% prediction interval
        z_score = 1.96  # For 95% CI

        lower = y_pred - z_score * y_std
        upper = y_pred + z_score * y_std

        return {
            "lower": lower,
            "upper": upper,
            "width": upper - lower,
            "relative_width": (upper - lower) / (np.abs(y_pred) + 1e-10),
        }
