"""Cross-validation. metrics.py re-exports CrossValidator from here."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    GroupKFold,
    LeaveOneOut,
)
from sklearn.base import clone as sklearn_clone
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
import logging

logger = logging.getLogger(__name__)


def _clone_model(model):
    """Prefer the model's own .clone(); sklearn's doesn't work on our wrappers."""
    if hasattr(model, "clone") and callable(getattr(model, "clone")):
        return model.clone()
    return sklearn_clone(model)


class CrossValidator:
    """
    Stratified (default), plain, grouped, or leave-one-out CV.

    Use strategy='group' with species as groups to keep the same host out of
    both sides of a split. Each fold fits a fresh clone.
    """

    def __init__(
        self, n_folds: int = 5, random_state: int = 42, strategy: str = "stratified"
    ):
        self.n_folds = n_folds
        self.random_state = random_state
        self.strategy = strategy
        self.cv_splitter = None
        if strategy not in ("stratified", "standard", "loo", "group"):
            raise ValueError(f"Unknown strategy: {strategy}")
        if strategy != "group":
            self._setup_cv()  # group splitter needs the groups, built in split()

    def _setup_cv(self):
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

    def split(
        self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """List of (train_idx, test_idx) per fold."""
        if self.strategy == "group":
            if groups is None:
                raise ValueError("groups must be provided for strategy='group'")
            cv = GroupKFold(n_splits=self.n_folds)
            return list(cv.split(X, y, groups))
        else:
            return list(self.cv_splitter.split(X, y))

    def cross_validate(
        self,
        model,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        scoring: str = "f1",
        groups: Optional[np.ndarray] = None,
        alpha: float = 0.05,
    ) -> Dict:
        """
        Refits a clone per fold.

        y_true/predictions/probabilities come back concatenated in fold
        order, with fold_indices giving each fold's test rows.
        """
        valid_scorers = {"f1", "accuracy", "roc_auc", "precision", "recall"}
        if scoring not in valid_scorers:
            raise ValueError(f"scoring must be one of {valid_scorers}, got {scoring}")

        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)

        if len(X_arr) == 0:
            raise ValueError("Cannot cross-validate on empty data")

        logger.info(f"Performing {self.n_folds}-fold CV with {scoring}")

        splits = self.split(X_arr, y_arr, groups=groups)

        scores = []
        all_y_true = []
        all_predictions = []
        all_probabilities = []
        fold_indices = []

        for fold, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X_arr[train_idx], X_arr[test_idx]
            y_train, y_test = y_arr[train_idx], y_arr[test_idx]

            model_clone = _clone_model(model)
            model_clone.fit(X_train, y_train)

            y_pred = model_clone.predict(X_test)
            has_proba = hasattr(model_clone, "predict_proba")
            y_prob = model_clone.predict_proba(X_test)[:, 1] if has_proba else None

            if scoring == "f1":
                score = f1_score(y_test, y_pred, zero_division=0)
            elif scoring == "accuracy":
                score = accuracy_score(y_test, y_pred)
            elif scoring == "precision":
                score = precision_score(y_test, y_pred, zero_division=0)
            elif scoring == "recall":
                score = recall_score(y_test, y_pred, zero_division=0)
            elif scoring == "roc_auc":
                if not has_proba:
                    raise ValueError("roc_auc scoring requires predict_proba")
                score = roc_auc_score(y_test, y_prob)

            scores.append(score)
            all_y_true.append(y_test)
            all_predictions.append(y_pred)
            if y_prob is not None:
                all_probabilities.append(y_prob)
            fold_indices.append(test_idx)

            logger.info(f"  Fold {fold + 1}/{len(splits)}: {scoring}={score:.4f}")

        scores_arr = np.array(scores)
        mean_score = float(scores_arr.mean())
        std_score = float(scores_arr.std(ddof=1)) if len(scores_arr) > 1 else 0.0

        # t CI over fold scores - descriptive only, the folds aren't independent.
        # for model comparisons use StatisticalTests instead
        from scipy import stats as _stats

        if len(scores_arr) > 1:
            se = std_score / np.sqrt(len(scores_arr))
            t_crit = _stats.t.ppf(1 - alpha / 2, df=len(scores_arr) - 1)
            ci_lower = mean_score - t_crit * se
            ci_upper = mean_score + t_crit * se
        else:
            ci_lower = ci_upper = mean_score

        results = {
            "fold_scores": scores,
            "mean_score": mean_score,
            "std_score": std_score,
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "n_folds": len(splits),
            "scoring": scoring,
            "y_true": np.concatenate(all_y_true),
            "predictions": np.concatenate(all_predictions),
            "probabilities": (
                np.concatenate(all_probabilities) if all_probabilities else None
            ),
            "fold_indices": fold_indices,
        }

        logger.info(
            f"CV complete: {mean_score:.3f} ± {std_score:.3f} "
            f"(95% CI [{ci_lower:.3f}, {ci_upper:.3f}])"
        )

        return results

    def get_feature_importance_cv(
        self,
        model,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        feature_names: List[str],
    ) -> pd.DataFrame:
        """
        mean/std/min/max importance per feature across folds, for stability.

        Empty frame if the model has neither feature_importances_ nor coef_.
        """
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)

        splits = self.split(X_arr, y_arr)

        importance_list = []

        for train_idx, test_idx in splits:
            X_train, y_train = X_arr[train_idx], y_arr[train_idx]

            model_clone = _clone_model(model)
            model_clone.fit(X_train, y_train)

            raw = getattr(model_clone, "model", model_clone)
            if hasattr(raw, "feature_importances_"):
                importance = raw.feature_importances_
            elif hasattr(raw, "coef_"):
                importance = np.abs(raw.coef_[0])
            else:
                continue

            importance_list.append(importance)

        if not importance_list:
            return pd.DataFrame()

        importance_array = np.array(importance_list)

        results = pd.DataFrame(
            {
                "feature": feature_names[: importance_array.shape[1]],
                "mean": importance_array.mean(axis=0),
                "std": importance_array.std(axis=0),
                "min": importance_array.min(axis=0),
                "max": importance_array.max(axis=0),
            }
        )

        return results.sort_values("mean", ascending=False).reset_index(drop=True)
