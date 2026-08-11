"""
Cross-validation strategies with advanced options.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    GroupKFold,
    LeaveOneOut,
    cross_val_score,
    cross_val_predict,
)
from sklearn.metrics import make_scorer, f1_score, accuracy_score, roc_auc_score
import logging

logger = logging.getLogger(__name__)


class CrossValidator:
    """
    Advanced cross-validation with multiple strategies.

    Supports:
    - Stratified K-Fold
    - Group K-Fold (for species-based splitting)
    - Leave-One-Out
    - Custom fold generation
    """

    def __init__(
        self, n_folds: int = 5, random_state: int = 42, strategy: str = "stratified"
    ):
        """
        Initialize cross-validator.

        Args:
            n_folds: Number of folds
            random_state: Random seed
            strategy: 'stratified', 'group', 'loo', 'standard'
        """
        self.n_folds = n_folds
        self.random_state = random_state
        self.strategy = strategy
        self.cv_splitter = None
        self._setup_cv()

    def _setup_cv(self):
        """Set up the CV splitter based on strategy."""
        if self.strategy == "stratified":
            self.cv_splitter = StratifiedKFold(
                n_splits=self.n_folds, shuffle=True, random_state=self.random_state
            )
        elif self.strategy == "standard":
            self.cv_splitter = KFold(
                n_splits=self.n_folds, shuffle=True, random_state=self.random_state
            )
        elif self.strategy == "loo":
            self.cv_splitter = LeaveOneOut()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def split(
        self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices.

        Args:
            X: Features
            y: Labels
            groups: Group labels for group CV

        Returns:
            List of (train_idx, test_idx) tuples
        """
        if self.strategy == "group" and groups is not None:
            cv = GroupKFold(n_splits=self.n_folds)
            return list(cv.split(X, y, groups))
        else:
            return list(self.cv_splitter.split(X, y))

    def cross_validate(
        self, model, X: np.ndarray, y: np.ndarray, scoring: str = "f1"
    ) -> Dict:
        """
        Perform cross-validation.

        Args:
            model: Model to evaluate
            X: Features
            y: Labels
            scoring: Scoring metric

        Returns:
            Dict: CV results
        """
        logger.info(f"Performing {self.n_folds}-fold CV with {scoring}")

        # Get split indices
        splits = self.split(X, y)

        scores = []
        predictions = []
        probabilities = []

        for fold, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)

            # Predict
            y_pred = model_clone.predict(X_test)
            predictions.extend(y_pred)

            # Probabilities
            if hasattr(model_clone, "predict_proba"):
                y_prob = model_clone.predict_proba(X_test)[:, 1]
                probabilities.extend(y_prob)

            # Score
            if scoring == "f1":
                score = f1_score(y_test, y_pred)
            elif scoring == "accuracy":
                score = accuracy_score(y_test, y_pred)
            elif scoring == "roc_auc" and hasattr(model_clone, "predict_proba"):
                score = roc_auc_score(y_test, y_prob)
            else:
                score = f1_score(y_test, y_pred)

            scores.append(score)

        results = {
            "fold_scores": scores,
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "n_folds": len(splits),
            "predictions": np.array(predictions),
            "probabilities": np.array(probabilities) if probabilities else None,
        }

        logger.info(
            f"CV complete: {results['mean_score']:.3f} ± {results['std_score']:.3f}"
        )

        return results

    def get_feature_importance_cv(
        self, model, X: np.ndarray, y: np.ndarray, feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Get feature importance across CV folds.

        Args:
            model: Model with feature_importances_
            X: Features
            y: Labels
            feature_names: Feature names

        Returns:
            pd.DataFrame: Feature importance with statistics
        """
        splits = self.split(X, y)

        importance_list = []

        for train_idx, test_idx in splits:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model_clone = clone(model)
            model_clone.fit(X_train, y_train)

            if hasattr(model_clone, "feature_importances_"):
                importance = model_clone.feature_importances_
            elif hasattr(model_clone, "coef_"):
                importance = np.abs(model_clone.coef_[0])
            else:
                continue

            importance_list.append(importance)

        if not importance_list:
            return pd.DataFrame()

        # Calculate statistics
        importance_array = np.array(importance_list)

        results = pd.DataFrame(
            {
                "feature": feature_names[: len(importance_array[0])],
                "mean": importance_array.mean(axis=0),
                "std": importance_array.std(axis=0),
                "min": importance_array.min(axis=0),
                "max": importance_array.max(axis=0),
            }
        )

        results = results.sort_values("mean", ascending=False)

        return results


from sklearn.base import clone
