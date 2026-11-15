"""
Comprehensive tests for explainability and statistical analysis.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.evaluation.statistical_tests import StatisticalTests
from src.evaluation.uncertainty import UncertaintyEstimator
from src.evaluation.ablation import run_ablation_study
from src.models.classical import RandomForestModel


class TestStatisticalTests:
    """Comprehensive tests for statistical testing."""

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions for testing."""
        np.random.seed(42)
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        y_pred1 = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0, 1])
        y_pred2 = np.array([0, 0, 1, 0, 1, 0, 1, 1, 0, 0])
        return y_true, y_pred1, y_pred2

    def test_mcnemar_test(self, sample_predictions):
        """Test McNemar's test implementation."""
        y_true, y_pred1, y_pred2 = sample_predictions

        stat_test = StatisticalTests()
        result = stat_test.mcnemar_test(y_true, y_pred1, y_pred2)

        assert "b" in result
        assert "c" in result
        assert "chi2" in result
        assert "p_value" in result
        assert "significant" in result
        assert isinstance(result["significant"], (bool, np.bool_))

    def test_mcnemar_identical_predictions(self):
        """Test McNemar when predictions are identical."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])

        stat_test = StatisticalTests()
        result = stat_test.mcnemar_test(y_true, y_pred, y_pred)

        # When predictions are identical, b=c=0, so chi2=0 and p_value=1
        assert result["chi2"] == 0
        assert result["p_value"] == 1.0
        assert result["significant"] is False

    def test_paired_t_test(self, sample_predictions):
        """Test paired t-test implementation."""
        y_true, y_pred1, y_pred2 = sample_predictions

        stat_test = StatisticalTests()
        result = stat_test.paired_t_test(y_true, y_pred1, y_pred2, metric="f1")

        assert "mean_diff" in result
        assert "std_diff" in result
        assert "t_statistic" in result
        assert "p_value" in result
        assert "model1_mean" in result
        assert "model2_mean" in result

    def test_cohen_kappa_confidence(self, sample_predictions):
        """Test Cohen's Kappa with confidence intervals."""
        y_true, y_pred1, _ = sample_predictions

        stat_test = StatisticalTests()
        result = stat_test.cohen_kappa_confidence(y_true, y_pred1)

        assert "kappa" in result
        assert "ci_lower" in result or "lower" in result
        assert "ci_upper" in result or "upper" in result
        assert -1 <= result["kappa"] <= 1

    def test_anova_comparison(self, sample_predictions):
        """Test ANOVA for comparing multiple models."""
        y_true, y_pred1, y_pred2 = sample_predictions

        y_preds = {
            "model1": y_pred1,
            "model2": y_pred2,
        }

        stat_test = StatisticalTests()
        result = stat_test.compare_models_anova(y_true, y_preds)

        assert "f_statistic" in result
        assert "p_value" in result
        assert "significant" in result
        assert "model_scores" in result


class TestUncertaintyEstimation:
    """Comprehensive tests for uncertainty quantification."""

    @pytest.fixture
    def sample_model_and_data(self):
        """Create sample model and data."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)

        rf = RandomForestClassifier(n_estimators=20, random_state=42)
        rf.fit(X, y)

        return rf, X, y

    def test_ensemble_uncertainty(self, sample_model_and_data):
        """Test ensemble uncertainty estimation."""
        model, X, y = sample_model_and_data

        estimator = UncertaintyEstimator()
        uncertainty = estimator.ensemble_uncertainty(model, X)

        assert "mean" in uncertainty
        assert "variance" in uncertainty
        assert "std" in uncertainty
        assert "entropy" in uncertainty
        assert "confidence" in uncertainty

        assert len(uncertainty["mean"]) == len(X)
        assert len(uncertainty["variance"]) == len(X)

    def test_uncertainty_non_ensemble_model(self):
        """Test uncertainty with non-ensemble models."""
        from sklearn.linear_model import LogisticRegression

        X = np.random.randn(20, 5)
        y = np.random.randint(0, 2, 20)

        model = LogisticRegression()
        model.fit(X, y)

        estimator = UncertaintyEstimator()
        uncertainty = estimator.ensemble_uncertainty(model, X)

        # Should return empty dict for models without estimators_
        assert uncertainty == {} or "variance" not in uncertainty

    def test_conformal_prediction(self, sample_model_and_data):
        """Test conformal prediction for uncertainty quantification."""
        model, X, y = sample_model_and_data

        # Split into calibration and test
        X_cal, X_test = X[:20], X[20:]
        y_cal, y_test = y[:20], y[20:]

        estimator = UncertaintyEstimator()
        result = estimator.conformal_prediction(
            model, X_cal, y_cal, X_test, y_test, alpha=0.1
        )

        assert "prediction_sets" in result
        assert "coverage" in result
        assert len(result["prediction_sets"]) == len(X_test)

    def test_calibration_curve(self):
        """Test calibration curve generation."""
        from sklearn.linear_model import LogisticRegression

        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        model = LogisticRegression()
        model.fit(X, y)

        y_pred_proba = model.predict_proba(X)[:, 1]

        estimator = UncertaintyEstimator()
        cal_curve = estimator.calibration_curve(
            y, y_pred_proba, n_bins=5
        )

        assert "bin_centers" in cal_curve
        assert "bin_true_frac" in cal_curve or "bin_fractions" in cal_curve
        assert len(cal_curve["bin_centers"]) > 0


class TestAblationStudy:
    """Tests for ablation study framework."""

    @pytest.fixture
    def sample_data_with_features(self):
        """Create sample data with feature groups."""
        np.random.seed(42)
        X = pd.DataFrame({
            "gc_content": np.random.uniform(30, 70, 100),
            "cpg_frequency": np.random.uniform(0.01, 0.1, 100),
            "codon_usage": np.random.uniform(0.5, 1.0, 100),
            "body_mass": np.random.uniform(0.01, 100, 100),
            "lifespan": np.random.uniform(1, 80, 100),
            "hydrophobicity": np.random.uniform(-5, 5, 100),
        })

        y = np.random.randint(0, 2, 100)

        return X, y

    def test_ablation_study_execution(self, sample_data_with_features):
        """Test running an ablation study."""
        X, y = sample_data_with_features

        # Define feature groups
        feature_groups = {
            "genomic": ["gc_content", "cpg_frequency", "codon_usage"],
            "host": ["body_mass", "lifespan"],
            "protein": ["hydrophobicity"],
        }

        # Train baseline model
        model = RandomForestModel({"n_estimators": 10, "random_state": 42})
        model.fit(X.values, y)

        baseline_pred = model.predict(X.values)

        # Ablate each group
        ablation_results = {}

        for group_name, features in feature_groups.items():
            # Drop group
            X_ablated = X.drop(columns=features)

            # Train model without this group
            model_ablated = RandomForestModel({"n_estimators": 10, "random_state": 42})
            model_ablated.fit(X_ablated.values, y)
            pred_ablated = model_ablated.predict(X_ablated.values)

            # Compare performance
            from sklearn.metrics import f1_score
            baseline_f1 = f1_score(y, baseline_pred)
            ablated_f1 = f1_score(y, pred_ablated)

            ablation_results[group_name] = {
                "baseline_f1": baseline_f1,
                "ablated_f1": ablated_f1,
                "delta_f1": baseline_f1 - ablated_f1,
            }

        assert len(ablation_results) == 3
        assert all("delta_f1" in v for v in ablation_results.values())


class TestStatisticalRobustness:
    """Tests for statistical robustness of results."""

    def test_bootstrap_stability(self):
        """Test that bootstrap results are consistent."""
        from sklearn.ensemble import RandomForestClassifier
        from src.evaluation.metrics import ModelEvaluator

        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        evaluator = ModelEvaluator(model)

        # Evaluate twice with same random state
        metrics1 = evaluator.evaluate(X, y)
        metrics2 = evaluator.evaluate(X, y)

        # Confidence intervals should be consistent (same data, same model)
        assert "accuracy" in metrics1
        assert "accuracy" in metrics2
        assert metrics1["accuracy"] == metrics2["accuracy"]

    def test_multiple_comparison_correction(self):
        """Test Bonferroni correction for multiple comparisons."""
        p_values = [0.01, 0.05, 0.1, 0.02]
        n_tests = len(p_values)

        # Bonferroni correction
        alpha = 0.05
        corrected_alpha = alpha / n_tests

        significant = [p < corrected_alpha for p in p_values]

        assert sum(significant) == 1  # Only one should remain significant
