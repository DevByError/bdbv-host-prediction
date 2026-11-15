"""Shared base class for the model wrappers."""

from abc import ABC, abstractmethod
import copy
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Wraps a raw sklearn/xgboost estimator behind a config dict.

    Because the constructor takes config=... rather than the individual
    hyperparameters, sklearn.base.clone() does not work on these - use
    .clone() here instead. Anything needing real sklearn interop (search,
    voting) should reach through .model for the raw estimator.
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
        pass

    def _validate_input(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None, stage: str = "fit") -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Returns (X, y) as arrays; y is None when stage != "fit"."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values

        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be np.ndarray or pd.DataFrame, got {type(X)}")

        if X.size == 0:
            raise ValueError("X cannot be empty")

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {X.ndim}D with shape {X.shape}")

        if np.any(np.isnan(X)):
            n_nan = np.sum(np.isnan(X))
            raise ValueError(f"X contains {n_nan} NaN values. Handle missing data before training.")

        if np.any(np.isinf(X)):
            n_inf = np.sum(np.isinf(X))
            raise ValueError(f"X contains {n_inf} infinite values. Check for division by zero or overflow.")

        if stage == "fit":
            if y is None:
                raise ValueError("y is required for model fitting")

            if not isinstance(y, (np.ndarray, list)):
                raise TypeError(f"y must be np.ndarray or list, got {type(y)}")

            y = np.array(y)

            if y.ndim != 1:
                raise ValueError(f"y must be 1D, got {y.ndim}D")

            if len(y) != X.shape[0]:
                raise ValueError(
                    f"X and y have mismatched lengths: "
                    f"X has {X.shape[0]} samples, y has {len(y)}"
                )

            if np.any(np.isnan(y)):
                n_nan_y = np.sum(np.isnan(y))
                raise ValueError(f"y contains {n_nan_y} NaN values")

            unique_classes = np.unique(y)
            if len(unique_classes) < 2:
                raise ValueError(
                    f"y must have at least 2 classes for classification, "
                    f"got {len(unique_classes)}: {unique_classes}"
                )

            logger.debug(
                f"Input validation passed: X.shape={X.shape}, "
                f"y.shape={y.shape}, classes={unique_classes}"
            )
            return X, y

        else:
            if self.feature_names is not None and len(self.feature_names) != X.shape[1]:
                logger.warning(
                    f"Prediction input has {X.shape[1]} features but model expects "
                    f"{len(self.feature_names)} (trained on {self.feature_names})"
                )

            logger.debug(f"Input validation passed for prediction: X.shape={X.shape}")
            return X, None

    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray):
        pass

    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        pass

    @staticmethod
    def _validate_fit_inputs(
        X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> None:
        if X is None or y is None:
            raise ValueError("X and y must not be None")

        n_x = len(X)
        n_y = len(y)
        if n_x == 0 or n_y == 0:
            raise ValueError(f"Cannot fit on empty data (len(X)={n_x}, len(y)={n_y})")
        if n_x != n_y:
            raise ValueError(f"X ({n_x} rows) and y ({n_y} rows) length mismatch")

        X_values = X.values if isinstance(X, (pd.DataFrame, pd.Series)) else X
        try:
            X_arr = np.asarray(X_values, dtype=float)
        except (TypeError, ValueError) as e:
            raise TypeError(f"X must be numeric array-like: {e}")

        if np.isnan(X_arr).any():
            raise ValueError(
                "X contains NaN values. Impute missing values before fitting "
                "(e.g. via FeatureEngineer, which imputes automatically)."
            )
        if np.isinf(X_arr).any():
            raise ValueError("X contains infinite values")

        y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)
        if pd.isnull(y_arr).any():
            raise ValueError("y contains missing/NaN labels")

        n_classes = len(np.unique(y_arr))
        if n_classes < 2:
            raise ValueError(
                f"y must contain at least 2 classes for classification, found "
                f"{n_classes}. This usually indicates a bad train/test split "
                f"or a data-generation bug."
            )

    @staticmethod
    def _validate_predict_inputs(X: Union[np.ndarray, pd.DataFrame]) -> None:
        if X is None or len(X) == 0:
            raise ValueError("Cannot predict on empty data")

        X_values = X.values if isinstance(X, (pd.DataFrame, pd.Series)) else X
        X_arr = np.asarray(X_values, dtype=float)
        if np.isnan(X_arr).any():
            raise ValueError(
                "X contains NaN values at prediction time. Ensure the same "
                "FeatureEngineer used at training time is applied via "
                ".transform() (not .fit_transform())."
            )
        if np.isinf(X_arr).any():
            raise ValueError("X contains infinite values")

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Importances (trees) or |coef| (linear), sorted descending; None if neither."""
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
        """joblib dump of the estimator plus the wrapper state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "config": self.config,
            "name": self.name,
            "class_name": type(self).__name__,
            "is_fitted": self._is_fitted,
            "training_history": self.training_history,
        }

        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: Path):
        """Loads in place and returns self, so RandomForestModel().load(p) works."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        model_data = joblib.load(path)

        self.model = model_data["model"]
        self.feature_names = model_data.get("feature_names")
        self.config = model_data.get("config", {})
        self.name = model_data.get("name", self.name)
        self._is_fitted = model_data.get("is_fitted", True)
        self.training_history = model_data.get("training_history", {})

        logger.info(f"Model loaded from {path}")
        return self

    def clone(self) -> "BaseModel":
        """Fresh unfitted instance with a deep-copied config. Use this, not sklearn.clone."""
        return type(self)(config=copy.deepcopy(self.config))

    def get_params(self) -> Dict:
        # for logging only - these can't be fed back into the constructor
        if hasattr(self.model, "get_params"):
            return self.model.get_params()
        return {}

    def set_params(self, **params):
        if hasattr(self.model, "set_params"):
            self.model.set_params(**params)

    def __repr__(self) -> str:
        return f"{self.name}(fitted={self._is_fitted})"
