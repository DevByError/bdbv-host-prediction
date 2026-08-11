"""
Feature preprocessing: scaling, imputation, encoding.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging

logger = logging.getLogger(__name__)


class FeaturePreprocessor:
    """
    Comprehensive feature preprocessing pipeline.

    Supports:
    - Multiple scaling methods
    - Different imputation strategies
    - Categorical encoding
    - Feature selection
    """

    def __init__(
        self,
        numeric_scaler: str = "standard",
        imputation_strategy: str = "median",
        categorical_encoding: str = "onehot",
        handle_unknown: str = "ignore",
    ):
        """
        Initialize preprocessor.

        Args:
            numeric_scaler: 'standard', 'minmax', 'robust'
            imputation_strategy: 'mean', 'median', 'most_frequent', 'knn'
            categorical_encoding: 'onehot', 'label', 'target'
            handle_unknown: 'ignore' or 'error'
        """
        self.numeric_scaler = numeric_scaler
        self.imputation_strategy = imputation_strategy
        self.categorical_encoding = categorical_encoding
        self.handle_unknown = handle_unknown

        self.preprocessor = None
        self.feature_names = None
        self._is_fitted = False

    def build_pipeline(
        self, numeric_cols: List[str], categorical_cols: List[str]
    ) -> Pipeline:
        """
        Build preprocessing pipeline.

        Args:
            numeric_cols: List of numeric column names
            categorical_cols: List of categorical column names

        Returns:
            Pipeline: Preprocessing pipeline
        """
        # Numeric transformer
        numeric_transformer = Pipeline(
            [("imputer", self._get_imputer()), ("scaler", self._get_scaler())]
        )

        # Categorical transformer
        categorical_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", self._get_encoder()),
            ]
        )

        # Combine
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        self.preprocessor = preprocessor

        return preprocessor

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit preprocessor and transform data.

        Args:
            X: Input DataFrame

        Returns:
            pd.DataFrame: Preprocessed data
        """
        # Identify column types
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # Remove non-feature columns
        exclude = ["infection_status", "notes", "id", "sample_id"]
        numeric_cols = [c for c in numeric_cols if c not in exclude]
        categorical_cols = [c for c in categorical_cols if c not in exclude]

        logger.info(f"Numeric columns: {len(numeric_cols)}")
        logger.info(f"Categorical columns: {len(categorical_cols)}")

        # Build pipeline
        self.build_pipeline(numeric_cols, categorical_cols)

        # Fit and transform
        X_transformed = self.preprocessor.fit_transform(X)

        # Get feature names
        self.feature_names = self._get_feature_names()
        self._is_fitted = True

        # Return as DataFrame
        return pd.DataFrame(X_transformed, columns=self.feature_names)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor.

        Args:
            X: Input DataFrame

        Returns:
            pd.DataFrame: Preprocessed data
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor must be fitted first")

        X_transformed = self.preprocessor.transform(X)

        return pd.DataFrame(X_transformed, columns=self.feature_names)

    def _get_scaler(self):
        """Get scaler based on configuration."""
        if self.numeric_scaler == "standard":
            return StandardScaler()
        elif self.numeric_scaler == "minmax":
            return MinMaxScaler()
        elif self.numeric_scaler == "robust":
            return RobustScaler()
        else:
            return StandardScaler()

    def _get_imputer(self):
        """Get imputer based on configuration."""
        if self.imputation_strategy == "knn":
            return KNNImputer(n_neighbors=3)
        else:
            return SimpleImputer(strategy=self.imputation_strategy)

    def _get_encoder(self):
        """Get encoder based on configuration."""
        from sklearn.preprocessing import OneHotEncoder, LabelEncoder

        if self.categorical_encoding == "onehot":
            return OneHotEncoder(
                handle_unknown=self.handle_unknown, sparse_output=False
            )
        elif self.categorical_encoding == "label":
            return LabelEncoder()
        else:
            return OneHotEncoder(
                handle_unknown=self.handle_unknown, sparse_output=False
            )

    def _get_feature_names(self) -> List[str]:
        """Get feature names from fitted preprocessor."""
        try:
            return self.preprocessor.get_feature_names_out().tolist()
        except (AttributeError, NotImplementedError):
            # Manual fallback
            names = []
            for name, transformer, columns in self.preprocessor.transformers_:
                if hasattr(transformer, "get_feature_names_out"):
                    names.extend(transformer.get_feature_names_out())
                else:
                    names.extend(columns)
            return names
