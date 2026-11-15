"""Classical classifiers used in the comparison."""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from src.models.base import BaseModel
import logging

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """Random Forest, balanced class weights."""

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        default_config = {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }
        default_config.update(config)
        super().__init__("RandomForest", default_config)
        self._build_model()

    def _build_model(self):
        self.model = RandomForestClassifier(**self.config)

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray):
        self._validate_fit_inputs(X, y)
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values

        self.model.fit(X, y)
        self._is_fitted = True
        logger.info(f"RandomForest trained on {X.shape[0]} samples")

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        self._validate_predict_inputs(X)

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        self._validate_predict_inputs(X)

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.predict_proba(X)


class XGBoostModel(BaseModel):
    """XGBoost. Optional dependency - raises on construction if not installed."""

    def __init__(self, config: Optional[Dict] = None):
        if not XGB_AVAILABLE:
            raise ImportError(
                "XGBoost is not installed. Install with: pip install xgboost"
            )

        config = config or {}
        default_config = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 1.0,
            "random_state": 42,
            "eval_metric": "logloss",
        }
        default_config.update(config)
        super().__init__("XGBoost", default_config)
        self._build_model()

    def _build_model(self):
        self.model = xgb.XGBClassifier(**self.config)

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray):
        self._validate_fit_inputs(X, y)
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values

        self.model.fit(X, y)
        self._is_fitted = True
        logger.info(f"XGBoost trained on {X.shape[0]} samples")

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        self._validate_predict_inputs(X)

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        self._validate_predict_inputs(X)

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.predict_proba(X)


class LogisticRegressionModel(BaseModel):
    """Linear baseline. Coefficients are log-odds per SD on the scaled features."""

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        default_config = {
            "max_iter": 1000,
            "random_state": 42,
            "class_weight": "balanced",
            "C": 1.0,
            "solver": "lbfgs",
        }
        default_config.update(config)
        super().__init__("LogisticRegression", default_config)
        self.scaler = StandardScaler()
        self._build_model()

    def _build_model(self):
        self.model = LogisticRegression(**self.config)

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray):
        # FeatureEngineer already scales upstream; this scaler is just so the
        # class works standalone on raw input
        self._validate_fit_inputs(X, y)
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values

        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled, y)
        self._is_fitted = True
        logger.info(f"LogisticRegression trained on {X.shape[0]} samples")

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        self._validate_predict_inputs(X)

        if isinstance(X, pd.DataFrame):
            X = X.values

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        self._validate_predict_inputs(X)

        if isinstance(X, pd.DataFrame):
            X = X.values

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class SVMModel(BaseModel):
    """RBF SVM. probability=True by default since we need predict_proba."""

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        default_config = {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "probability": True,
            "random_state": 42,
            "class_weight": "balanced",
        }
        default_config.update(config)
        super().__init__("SVM", default_config)
        self.scaler = StandardScaler()
        self._build_model()

    def _build_model(self):
        self.model = SVC(**self.config)

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray):
        # same standalone-scaler story as LogisticRegressionModel
        self._validate_fit_inputs(X, y)
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values

        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled, y)
        self._is_fitted = True
        logger.info(f"SVM trained on {X.shape[0]} samples")

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        self._validate_predict_inputs(X)

        if isinstance(X, pd.DataFrame):
            X = X.values

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        self._validate_predict_inputs(X)

        if isinstance(X, pd.DataFrame):
            X = X.values

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
