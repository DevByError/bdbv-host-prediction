"""
Feature selection methods for dimensionality reduction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from sklearn.feature_selection import (
    SelectKBest,
    SelectPercentile,
    VarianceThreshold,
    RFE,
    RFECV,
    mutual_info_classif,
    f_classif,
    chi2,
    SelectFromModel,
)
from sklearn.ensemble import RandomForestClassifier
import logging

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Advanced feature selection with multiple methods.

    Methods:
    - Variance threshold
    - Univariate selection (f-test, mutual information)
    - Recursive feature elimination
    - Model-based selection
    - Correlation-based selection
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.selected_features = None
        self.selector = None

    def variance_threshold(self, X: pd.DataFrame, threshold: float = 0.01) -> List[str]:
        """
        Remove features with low variance.

        Args:
            X: Features
            threshold: Variance threshold

        Returns:
            List[str]: Selected feature names
        """
        logger.info(f"Applying variance threshold: {threshold}")

        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)

        selected = X.columns[selector.get_support()].tolist()

        logger.info(f"Selected {len(selected)} out of {len(X.columns)} features")

        return selected

    def univariate_selection(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        method: str = "mutual_info",
        n_features: Optional[int] = None,
        percentile: Optional[int] = None,
    ) -> List[str]:
        """
        Univariate feature selection.

        Args:
            X: Features
            y: Labels
            method: 'mutual_info', 'f_classif', 'chi2'
            n_features: Number of features to select
            percentile: Percentile of features to select

        Returns:
            List[str]: Selected feature names
        """
        logger.info(f"Applying univariate selection: {method}")

        # Choose scoring function
        if method == "mutual_info":
            score_func = mutual_info_classif
        elif method == "f_classif":
            score_func = f_classif
        elif method == "chi2":
            score_func = chi2
        else:
            raise ValueError(f"Unknown method: {method}")

        # Select selector
        if n_features is not None:
            selector = SelectKBest(score_func=score_func, k=n_features)
        elif percentile is not None:
            selector = SelectPercentile(score_func=score_func, percentile=percentile)
        else:
            selector = SelectKBest(score_func=score_func, k=min(20, X.shape[1]))

        selector.fit(X, y)

        selected = X.columns[selector.get_support()].tolist()

        logger.info(f"Selected {len(selected)} out of {len(X.columns)} features")

        return selected

    def recursive_feature_elimination(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        n_features: Optional[int] = None,
        estimator=None,
    ) -> List[str]:
        """
        Recursive Feature Elimination.

        Args:
            X: Features
            y: Labels
            n_features: Number of features to select
            estimator: Estimator for RFE

        Returns:
            List[str]: Selected feature names
        """
        logger.info("Applying recursive feature elimination")

        if estimator is None:
            estimator = RandomForestClassifier(
                n_estimators=50, random_state=self.random_state
            )

        if n_features is None:
            n_features = min(20, X.shape[1] // 2)

        selector = RFE(estimator, n_features_to_select=n_features)
        selector.fit(X, y)

        selected = X.columns[selector.get_support()].tolist()

        logger.info(f"Selected {len(selected)} out of {len(X.columns)} features")

        return selected

    def model_based_selection(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        threshold: Optional[str] = None,
        estimator=None,
    ) -> List[str]:
        """
        Model-based feature selection.

        Args:
            X: Features
            y: Labels
            threshold: Threshold for selection
            estimator: Estimator with feature_importances_

        Returns:
            List[str]: Selected feature names
        """
        logger.info("Applying model-based selection")

        if estimator is None:
            estimator = RandomForestClassifier(
                n_estimators=50, random_state=self.random_state
            )

        selector = SelectFromModel(estimator, threshold=threshold)
        selector.fit(X, y)

        selected = X.columns[selector.get_support()].tolist()

        logger.info(f"Selected {len(selected)} out of {len(X.columns)} features")

        return selected

    def correlation_based_selection(
        self, X: pd.DataFrame, threshold: float = 0.9
    ) -> List[str]:
        """
        Remove features with high correlation.

        Args:
            X: Features
            threshold: Correlation threshold

        Returns:
            List[str]: Selected feature names
        """
        logger.info(f"Removing correlated features (threshold: {threshold})")

        # Calculate correlation matrix
        corr_matrix = X.corr().abs()

        # Select upper triangle
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation above threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        selected = [col for col in X.columns if col not in to_drop]

        logger.info(f"Selected {len(selected)} out of {len(X.columns)} features")
        logger.info(f"Dropped {len(to_drop)} features: {to_drop}")

        return selected

    def get_feature_scores(
        self, X: pd.DataFrame, y: np.ndarray, method: str = "mutual_info"
    ) -> pd.DataFrame:
        """
        Get feature importance scores.

        Args:
            X: Features
            y: Labels
            method: Scoring method

        Returns:
            pd.DataFrame: Feature scores
        """
        if method == "mutual_info":
            scores = mutual_info_classif(X, y)
        elif method == "f_classif":
            scores, _ = f_classif(X, y)
        else:
            scores = mutual_info_classif(X, y)

        df = pd.DataFrame({"feature": X.columns, "score": scores}).sort_values(
            "score", ascending=False
        )

        return df

    def select_features(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        methods: List[str] = None,
        n_features: Optional[int] = None,
    ) -> List[str]:
        """
        Combine multiple feature selection methods.

        Args:
            X: Features
            y: Labels
            methods: List of methods to apply
            n_features: Number of features to select

        Returns:
            List[str]: Selected feature names
        """
        if methods is None:
            methods = ["variance", "correlation", "univariate"]

        all_selected = set(X.columns.tolist())

        for method in methods:
            if method == "variance":
                selected = self.variance_threshold(X)
            elif method == "correlation":
                selected = self.correlation_based_selection(X)
            elif method == "univariate":
                selected = self.univariate_selection(X, y)
            else:
                continue

            all_selected = all_selected.intersection(set(selected))

            if n_features and len(all_selected) <= n_features:
                break

        selected = list(all_selected)

        if n_features and len(selected) > n_features:
            # Get top features by importance
            scores = self.get_feature_scores(X[selected], y)
            selected = scores.head(n_features)["feature"].tolist()

        logger.info(f"Final selection: {len(selected)} features")

        return selected
