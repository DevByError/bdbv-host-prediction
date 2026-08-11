"""
Tests for machine learning models.
"""

import pytest
import numpy as np
from src.models.classical import RandomForestModel, LogisticRegressionModel, SVMModel


class TestModels:
    """Test model classes."""

    def test_random_forest(self):
        """Test Random Forest model."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        model = RandomForestModel({"n_estimators": 10, "random_state": 42})
        model.fit(X, y)

        assert model._is_fitted is True

        preds = model.predict(X)
        assert len(preds) == 100
        assert set(preds) <= {0, 1}

        proba = model.predict_proba(X)
        assert proba.shape == (100, 2)

    def test_logistic_regression(self):
        """Test Logistic Regression model."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        model = LogisticRegressionModel({"max_iter": 100, "random_state": 42})
        model.fit(X, y)

        assert model._is_fitted is True

        preds = model.predict(X)
        assert len(preds) == 100
        assert set(preds) <= {0, 1}

    def test_svm(self):
        """Test SVM model."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        model = SVMModel({"probability": True, "random_state": 42})
        model.fit(X, y)

        assert model._is_fitted is True

        preds = model.predict(X)
        assert len(preds) == 100
        assert set(preds) <= {0, 1}

        proba = model.predict_proba(X)
        assert proba.shape == (100, 2)

    def test_feature_importance(self):
        """Test feature importance extraction."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        model = RandomForestModel({"n_estimators": 10, "random_state": 42})
        model.fit(X, y)

        importance = model.get_feature_importance()

        assert importance is not None
        assert len(importance) == 5
        assert "feature" in importance.columns
        assert "importance" in importance.columns

    def test_save_load(self, tmp_path):
        """Test model saving and loading."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        model = RandomForestModel({"n_estimators": 10, "random_state": 42})
        model.fit(X, y)

        save_path = tmp_path / "test_model.pkl"
        model.save(save_path)

        # Load model
        new_model = RandomForestModel()
        new_model.load(save_path)

        assert new_model._is_fitted is True
        assert new_model.name == model.name

    def test_model_with_string_features(self):
        """Test model with string feature names."""
        import pandas as pd

        X = pd.DataFrame(
            np.random.randn(100, 5),
            columns=["feat1", "feat2", "feat3", "feat4", "feat5"],
        )
        y = np.random.randint(0, 2, 100)

        model = RandomForestModel({"n_estimators": 10, "random_state": 42})
        model.fit(X, y)

        assert model.feature_names == ["feat1", "feat2", "feat3", "feat4", "feat5"]
