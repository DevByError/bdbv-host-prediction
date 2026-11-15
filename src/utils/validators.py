"""Shape and sanity checks on features, labels and predictions."""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class Validator:
    @staticmethod
    def validate_features(
        X: Union[np.ndarray, pd.DataFrame],
        required_features: Optional[List[str]] = None,
    ) -> bool:
        """NaNs only warn here; a missing required column is what makes this return False."""
        if X is None:
            logger.error("Features are None")
            return False

        if isinstance(X, pd.DataFrame):
            if required_features:
                missing = [f for f in required_features if f not in X.columns]
                if missing:
                    logger.error(f"Missing required features: {missing}")
                    return False

            if X.isnull().any().any():
                logger.warning("Features contain NaN values")

        if isinstance(X, np.ndarray):
            if np.isnan(X).any():
                logger.warning("Features contain NaN values")

        return True

    @staticmethod
    def validate_labels(
        y: Union[np.ndarray, pd.Series], classes: Optional[List] = None
    ) -> bool:
        if y is None:
            logger.error("Labels are None")
            return False

        if isinstance(y, pd.Series):
            if y.isnull().any():
                logger.error("Labels contain NaN")
                return False
        elif isinstance(y, np.ndarray):
            if np.isnan(y).any():
                logger.error("Labels contain NaN")
                return False

        if classes is not None:
            unique_classes = np.unique(y)
            if not all(c in unique_classes for c in classes):
                logger.error(
                    f"Invalid classes. Expected: {classes}, got: {unique_classes}"
                )
                return False

        return True

    @staticmethod
    def validate_input_shape(
        X: Union[np.ndarray, pd.DataFrame], expected_features: Optional[int] = None
    ) -> bool:
        if X is None:
            logger.error("Input is None")
            return False

        if isinstance(X, pd.DataFrame):
            n_samples, n_features = X.shape
        else:
            n_samples, n_features = X.shape[0], X.shape[1] if len(X.shape) > 1 else 1

        if n_samples == 0:
            logger.error("No samples in input")
            return False

        if expected_features is not None and n_features != expected_features:
            logger.error(f"Expected {expected_features} features, got {n_features}")
            return False

        return True

    @staticmethod
    def validate_data_split(
        X_train: Union[np.ndarray, pd.DataFrame],
        X_test: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        y_test: Union[np.ndarray, pd.Series],
    ) -> bool:
        if len(X_train) != len(y_train):
            logger.error(
                f"X_train ({len(X_train)}) and y_train ({len(y_train)}) length mismatch"
            )
            return False

        if len(X_test) != len(y_test):
            logger.error(
                f"X_test ({len(X_test)}) and y_test ({len(y_test)}) length mismatch"
            )
            return False

        if len(X_train) == 0:
            logger.error("Training set is empty")
            return False

        if len(X_test) == 0:
            logger.error("Test set is empty")
            return False

        return True

    @staticmethod
    def validate_predictions(
        y_pred: Union[np.ndarray, List], expected_shape: Optional[tuple] = None
    ) -> bool:
        if y_pred is None:
            logger.error("Predictions are None")
            return False

        if isinstance(y_pred, list):
            y_pred = np.array(y_pred)

        if expected_shape is not None and y_pred.shape != expected_shape:
            logger.error(f"Expected shape {expected_shape}, got {y_pred.shape}")
            return False

        return True

    @staticmethod
    def sanitize_input(data: Dict) -> Dict:
        """Trims strings and turns numpy scalars into plain ints/floats so they serialise."""
        sanitized = {}

        for key, value in data.items():
            if isinstance(value, str):
                value = value.strip()

            if isinstance(value, (np.integer, np.floating)):
                value = float(value) if isinstance(value, np.floating) else int(value)

            sanitized[key] = value

        return sanitized
