"""Feature importance by model attribute, permutation, or dropping columns."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from sklearn.inspection import permutation_importance
import logging

logger = logging.getLogger(__name__)


class FeatureImportance:
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names

    def get_model_importance(self) -> pd.DataFrame:
        """feature_importances_ for trees, |coef_| for linear models, empty frame otherwise."""
        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
            method = "tree_importance"
        elif hasattr(self.model, "coef_"):
            importance = np.abs(self.model.coef_[0])
            method = "coefficient_absolute"
        else:
            return pd.DataFrame()

        df = pd.DataFrame(
            {"feature": self.feature_names[: len(importance)], "importance": importance}
        ).sort_values("importance", ascending=False)

        return df

    def get_permutation_importance(
        self, X: np.ndarray, y: np.ndarray, n_repeats: int = 10, scoring: str = "f1"
    ) -> pd.DataFrame:
        logger.info("Calculating permutation importance...")

        result = permutation_importance(
            self.model, X, y, n_repeats=n_repeats, scoring=scoring, random_state=42
        )

        df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)

        return df

    def get_drop_column_importance(
        self, X: np.ndarray, y: np.ndarray, cv_scores: np.ndarray
    ) -> pd.DataFrame:
        """Refits once per column, so this is slow - cv_scores is the baseline to compare against."""
        from sklearn.base import clone
        from sklearn.model_selection import cross_val_score

        n_features = X.shape[1]
        importance = []

        baseline = np.mean(cv_scores)

        for i in range(n_features):
            X_dropped = np.delete(X, i, axis=1)
            model_clone = clone(self.model)
            scores = cross_val_score(model_clone, X_dropped, y, cv=5, scoring="f1")
            score_drop = np.mean(scores)

            importance.append(baseline - score_drop)

        df = pd.DataFrame(
            {"feature": self.feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

        return df

    def get_combined_importance(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """Each method normalised to sum to 1, then averaged, so the scales are comparable."""
        methods = {}

        model_imp = self.get_model_importance()
        if not model_imp.empty:
            methods["model"] = model_imp

        try:
            perm_imp = self.get_permutation_importance(X, y)
            methods["permutation"] = perm_imp
        except Exception as e:
            logger.warning(f"Permutation importance failed: {e}")

        combined = None

        for name, df in methods.items():
            df = df.set_index("feature")
            df["normalized"] = df.iloc[:, 0] / df.iloc[:, 0].sum()
            df = df[["normalized"]].rename(columns={"normalized": name})

            if combined is None:
                combined = df
            else:
                combined = combined.join(df, how="outer")

        if combined is not None:
            combined["combined"] = combined.mean(axis=1)
            combined = combined.sort_values("combined", ascending=False)

            return combined

        return pd.DataFrame()
