"""Voting/stacking ensembles over the base classifiers."""

import numpy as np
from typing import Dict, List, Optional, Union
from sklearn.ensemble import VotingClassifier, StackingClassifier
import logging

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Voting or stacking over base_models.

    base_models can hold raw estimators or our BaseModel wrappers; wrappers
    get unwrapped to .model because sklearn clones base estimators during
    fit() and the wrappers aren't clone-able.
    """

    def __init__(
        self,
        base_models: Dict[str, object],
        ensemble_type: str = "voting",
        final_estimator: Optional[object] = None,
        voting: str = "soft",
    ):
        if not base_models:
            raise ValueError("base_models must contain at least one model")

        self.base_models = base_models
        self.ensemble_type = ensemble_type
        self.final_estimator = final_estimator
        self.voting = voting
        self.ensemble = None
        self._build_ensemble()

    @staticmethod
    def _unwrap(model):
        """Raw estimator behind a BaseModel wrapper, or the model itself."""
        return getattr(model, "model", model)

    def _build_ensemble(self):
        raw_estimators = [
            (name, self._unwrap(model)) for name, model in self.base_models.items()
        ]

        if self.ensemble_type == "voting":
            self.ensemble = VotingClassifier(
                estimators=raw_estimators, voting=self.voting
            )
        elif self.ensemble_type == "stacking":
            self.ensemble = StackingClassifier(
                estimators=raw_estimators,
                final_estimator=self._unwrap(self.final_estimator)
                if self.final_estimator is not None
                else None,
                cv=5,
            )
        else:
            raise ValueError(f"Unknown ensemble type: {self.ensemble_type}")

    def fit(self, X, y: np.ndarray) -> "EnsembleModel":
        # sklearn refits every base estimator here, fitted ones passed in are ignored
        X_arr = X.values if hasattr(X, "values") else np.asarray(X)
        y_arr = np.asarray(y)
        if len(X_arr) == 0:
            raise ValueError("Cannot fit ensemble on empty data")
        if len(X_arr) != len(y_arr):
            raise ValueError(f"X ({len(X_arr)}) and y ({len(y_arr)}) length mismatch")

        logger.info(
            f"Training {self.ensemble_type} ensemble with {len(self.base_models)} models"
        )
        self.ensemble.fit(X_arr, y_arr)
        return self

    def predict(self, X) -> np.ndarray:
        X_arr = X.values if hasattr(X, "values") else np.asarray(X)
        return self.ensemble.predict(X_arr)

    def predict_proba(self, X) -> np.ndarray:
        X_arr = X.values if hasattr(X, "values") else np.asarray(X)
        return self.ensemble.predict_proba(X_arr)

    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """{model name: importances}; models exposing neither attribute are skipped."""
        importance = {}

        for name, model in self.base_models.items():
            raw = self._unwrap(model)
            if hasattr(raw, "feature_importances_"):
                importance[name] = raw.feature_importances_
            elif hasattr(raw, "coef_"):
                importance[name] = np.abs(raw.coef_[0])

        return importance

    def get_voting_weights(self) -> Dict[str, float]:
        """Falls back to equal weights when the ensemble has none."""
        if hasattr(self.ensemble, "weights_"):
            return {
                name: weight
                for name, weight in zip(self.base_models.keys(), self.ensemble.weights_)
            }
        else:
            return {name: 1.0 for name in self.base_models.keys()}
