"""Feature selection."""

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
    """Each method returns the kept column names; nothing is stateful except
    random_state."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.selected_features = None
        self.selector = None

    def variance_threshold(self, X: pd.DataFrame, threshold: float = 0.01) -> List[str]:
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
        """method is mutual_info, f_classif or chi2 (chi2 needs non-negative X).
        With neither n_features nor percentile, keeps the top 20."""
        logger.info(f"Applying univariate selection: {method}")

        if method == "mutual_info":
            score_func = mutual_info_classif
        elif method == "f_classif":
            score_func = f_classif
        elif method == "chi2":
            score_func = chi2
        else:
            raise ValueError(f"Unknown method: {method}")

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
        """Defaults to a 50-tree random forest and half the columns."""
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
        """estimator needs feature_importances_ or coef_. threshold=None lets
        SelectFromModel use the mean importance."""
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
        """Of each correlated pair the later column gets dropped - upper
        triangle only, so the first occurrence survives."""
        logger.info(f"Removing correlated features (threshold: {threshold})")

        corr_matrix = X.corr().abs()

        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        selected = [col for col in X.columns if col not in to_drop]

        logger.info(f"Selected {len(selected)} out of {len(X.columns)} features")
        logger.info(f"Dropped {len(to_drop)} features: {to_drop}")

        return selected

    def get_feature_scores(
        self, X: pd.DataFrame, y: np.ndarray, method: str = "mutual_info"
    ) -> pd.DataFrame:
        """Columns: feature, score. Sorted descending."""
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
        """Intersection of the methods, then trimmed to n_features by MI score
        if still too many."""
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
            scores = self.get_feature_scores(X[selected], y)
            selected = scores.head(n_features)["feature"].tolist()

        logger.info(f"Final selection: {len(selected)} features")

        return selected
