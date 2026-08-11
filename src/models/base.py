"""
Base model class with common functionality.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all models.

    Provides common functionality:
    - Training
    - Prediction
    - Saving/Loading
    - Feature importance
    """

    def __init__(self, name: str, config: Optional[Dict] = None):
        self.name = name
        self.config = config or {}
        self.model = None
        self.feature_names = None
        self._is_fitted = False
        self.training_history = {}

    @abstractmethod
    def _build_model(self):
        """Build the underlying model"""
        pass

    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray):
        """Train the model"""
        pass

    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions"""
        pass

    @abstractmethod
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Get prediction probabilities"""
        pass

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance if available.

        Returns:
            pd.DataFrame: Feature importance
        """
        if not self._is_fitted:
            logger.warning("Model not fitted")
            return None

        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importance = np.abs(self.model.coef_[0])
        else:
            logger.warning(f"Model {self.name} doesn't support feature importance")
            return None

        if self.feature_names is not None:
            return pd.DataFrame(
                {"feature": self.feature_names, "importance": importance}
            ).sort_values("importance", ascending=False)
        else:
            return pd.DataFrame(
                {
                    "feature": [f"feature_{i}" for i in range(len(importance))],
                    "importance": importance,
                }
            ).sort_values("importance", ascending=False)

    def save(self, path: Path):
        """Save model to disk"""
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "config": self.config,
            "name": self.name,
            "is_fitted": self._is_fitted,
            "training_history": self.training_history,
        }

        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: Path):
        """Load model from disk"""
        model_data = joblib.load(path)

        self.model = model_data["model"]
        self.feature_names = model_data.get("feature_names")
        self.config = model_data.get("config", {})
        self.name = model_data.get("name", self.name)
        self._is_fitted = model_data.get("is_fitted", True)
        self.training_history = model_data.get("training_history", {})

        logger.info(f"Model loaded from {path}")

    def get_params(self) -> Dict:
        """Get model parameters"""
        if hasattr(self.model, "get_params"):
            return self.model.get_params()
        return {}

    def set_params(self, **params):
        """Set model parameters"""
        if hasattr(self.model, "set_params"):
            self.model.set_params(**params)

    def __repr__(self) -> str:
        return f"{self.name}(fitted={self._is_fitted})"
