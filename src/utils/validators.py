"""
Input validation utilities.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class Validator:
    """
    Data validation utilities.
    """

    @staticmethod
    def validate_features(
        X: Union[np.ndarray, pd.DataFrame],
        required_features: Optional[List[str]] = None,
    ) -> bool:
        """
        Validate features.

        Args:
            X: Features
            required_features: List of required feature names

        Returns:
            bool: Whether valid
        """
        if X is None:
            logger.error("Features are None")
            return False

        if isinstance(X, pd.DataFrame):
            if required_features:
                missing = [f for f in required_features if f not in X.columns]
                if missing:
                    logger.error(f"Missing required features: {missing}")
                    return False

            # Check for NaN
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
        """
        Validate labels.

        Args:
            y: Labels
            classes: Expected classes

        Returns:
            bool: Whether valid
        """
        if y is None:
            logger.error("Labels are None")
            return False

        # Check for NaN
        if isinstance(y, pd.Series):
            if y.isnull().any():
                logger.error("Labels contain NaN")
                return False
        elif isinstance(y, np.ndarray):
            if np.isnan(y).any():
                logger.error("Labels contain NaN")
                return False

        # Check classes
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
        """
        Validate input shape.

        Args:
            X: Input data
            expected_features: Expected number of features

        Returns:
            bool: Whether valid
        """
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
        """
        Validate train/test split.

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels

        Returns:
            bool: Whether valid
        """
        # Check sizes
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

        # Check both have samples
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
        """
        Validate predictions.

        Args:
            y_pred: Predictions
            expected_shape: Expected shape

        Returns:
            bool: Whether valid
        """
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
        """
        Sanitize input data.

        Args:
            data: Input data

        Returns:
            Dict: Sanitized data
        """
        sanitized = {}

        for key, value in data.items():
            # Remove whitespace
            if isinstance(value, str):
                value = value.strip()

            # Convert to standard types
            if isinstance(value, (np.integer, np.floating)):
                value = float(value) if isinstance(value, np.floating) else int(value)

            sanitized[key] = value

        return sanitized
