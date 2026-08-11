"""
Ensemble models combining multiple classifiers.
"""

import numpy as np
from typing import Dict, List, Optional, Union
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.base import clone
import logging

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Create ensemble models with voting or stacking.
    """

    def __init__(
        self,
        base_models: Dict[str, object],
        ensemble_type: str = "voting",
        final_estimator: Optional[object] = None,
        voting: str = "soft",
    ):
        """
        Initialize ensemble model.

        Args:
            base_models: Dictionary of {name: model}
            ensemble_type: 'voting' or 'stacking'
            final_estimator: Estimator for stacking
            voting: 'soft' or 'hard' for voting
        """
        self.base_models = base_models
        self.ensemble_type = ensemble_type
        self.final_estimator = final_estimator
        self.voting = voting
        self.ensemble = None
        self._build_ensemble()

    def _build_ensemble(self):
        """Build the ensemble model."""
        if self.ensemble_type == "voting":
            self.ensemble = VotingClassifier(
                estimators=list(self.base_models.items()), voting=self.voting
            )
        elif self.ensemble_type == "stacking":
            self.ensemble = StackingClassifier(
                estimators=list(self.base_models.items()),
                final_estimator=self.final_estimator,
                cv=5,
            )
        else:
            raise ValueError(f"Unknown ensemble type: {self.ensemble_type}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleModel":
        """Train the ensemble."""
        logger.info(
            f"Training {self.ensemble_type} ensemble with {len(self.base_models)} models"
        )
        self.ensemble.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.ensemble.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        return self.ensemble.predict_proba(X)

    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """
        Get feature importance from each model.

        Returns:
            Dict[str, np.ndarray]: Feature importance per model
        """
        importance = {}

        for name, model in self.base_models.items():
            if hasattr(model, "feature_importances_"):
                importance[name] = model.feature_importances_
            elif hasattr(model, "coef_"):
                importance[name] = np.abs(model.coef_[0])

        return importance

    def get_voting_weights(self) -> Dict[str, float]:
        """
        Get voting weights from ensemble.

        Returns:
            Dict[str, float]: Model weights
        """
        if hasattr(self.ensemble, "weights_"):
            return {
                name: weight
                for name, weight in zip(self.base_models.keys(), self.ensemble.weights_)
            }
        else:
            return {name: 1.0 for name in self.base_models.keys()}
