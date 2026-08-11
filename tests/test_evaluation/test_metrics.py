"""
Tests for evaluation metrics.
"""

import pytest
import numpy as np
from src.evaluation.metrics import ModelEvaluator
from sklearn.ensemble import RandomForestClassifier


class TestModelEvaluator:
    """Test ModelEvaluator class."""

    def test_evaluate_metrics(self):
        """Test metric calculation."""
        # Create data
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Evaluate
        evaluator = ModelEvaluator(model)
        metrics = evaluator.evaluate(X, y)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1"] <= 1

    def test_confidence_intervals(self):
        """Test confidence interval calculation."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        evaluator = ModelEvaluator(model)
        evaluator.evaluate(X, y)

        assert "accuracy" in evaluator.confidence_intervals
        ci = evaluator.confidence_intervals["accuracy"]
        assert "ci_lower" in ci
        assert "ci_upper" in ci
        assert ci["ci_lower"] <= ci["ci_upper"]

    def test_confusion_matrix(self):
        """Test confusion matrix."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        evaluator = ModelEvaluator(model)
        cm = evaluator.get_confusion_matrix(X, y)

        assert cm.shape == (2, 2)
        assert cm.sum() == 100

    def test_compare_models(self):
        """Test model comparison."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        models = {
            "model1": RandomForestClassifier(n_estimators=10, random_state=42),
            "model2": RandomForestClassifier(n_estimators=20, random_state=42),
        }

        for name, model in models.items():
            model.fit(X, y)

        evaluator = ModelEvaluator(models["model1"])
        comparison = evaluator.compare_models(models, X, y)

        assert "model1" in comparison.index
        assert "model2" in comparison.index
        assert "accuracy" in comparison.columns

    def test_statistical_significance(self):
        """Test statistical significance test."""
        y_true = np.random.randint(0, 2, 100)
        y_pred1 = np.random.randint(0, 2, 100)
        y_pred2 = np.random.randint(0, 2, 100)

        evaluator = ModelEvaluator(None)
        result = evaluator.statistical_significance_test(y_pred1, y_pred2, y_true)

        assert "p_value" in result
        assert "significant" in result
        assert isinstance(result["significant"], bool)
