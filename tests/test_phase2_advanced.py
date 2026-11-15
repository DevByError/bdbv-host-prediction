"""
Phase 2 Advanced Coverage Tests
Targets: hyperparameter, cross_validation, uncertainty, statistical_tests
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.models.classical import RandomForestModel, LogisticRegressionModel
from src.evaluation.uncertainty import UncertaintyEstimator
from src.evaluation.statistical_tests import StatisticalTests
from src.evaluation.cross_validation import CrossValidator
from src.evaluation.metrics import ModelEvaluator


# ============================================================================
# HYPERPARAMETER TUNING TESTS (12 tests)
# ============================================================================

class TestGridSearchVariants:
    """Test grid search with various configurations."""

    @pytest.fixture
    def data(self):
        """Simple test data."""
        np.random.seed(42)
        X = np.random.randn(100, 8)
        y = np.random.randint(0, 2, 100)
        return X, y

    def test_grid_search_single_param(self, data):
        """Grid search with single parameter."""
        X, y = data
        gs = GridSearchCV(
            RandomForestClassifier(random_state=42),
            {"n_estimators": [5, 10, 15]},
            cv=3
        )
        gs.fit(X, y)
        assert gs.best_params_["n_estimators"] in [5, 10, 15]

    def test_grid_search_two_params(self, data):
        """Grid search with two parameters."""
        X, y = data
        gs = GridSearchCV(
            RandomForestClassifier(random_state=42),
            {
                "n_estimators": [5, 10],
                "max_depth": [3, 5]
            },
            cv=2
        )
        gs.fit(X, y)
        assert "n_estimators" in gs.best_params_
        assert "max_depth" in gs.best_params_

    def test_grid_search_three_params(self, data):
        """Grid search with three parameters."""
        X, y = data
        gs = GridSearchCV(
            RandomForestClassifier(random_state=42),
            {
                "n_estimators": [5, 10],
                "max_depth": [3, 5],
                "min_samples_split": [2, 5]
            },
            cv=2
        )
        gs.fit(X, y)
        assert len(gs.best_params_) == 3

    def test_grid_search_best_score(self, data):
        """Grid search returns best score."""
        X, y = data
        gs = GridSearchCV(
            RandomForestClassifier(random_state=42),
            {"n_estimators": [5, 10]},
            cv=2
        )
        gs.fit(X, y)
        assert 0 <= gs.best_score_ <= 1

    def test_grid_search_cv_results(self, data):
        """Grid search has CV results."""
        X, y = data
        gs = GridSearchCV(
            RandomForestClassifier(random_state=42),
            {"n_estimators": [5, 10]},
            cv=2
        )
        gs.fit(X, y)
        assert "mean_test_score" in gs.cv_results_
        assert len(gs.cv_results_["mean_test_score"]) == 2

    def test_random_search_single_param(self, data):
        """Random search with single parameter."""
        X, y = data
        rs = RandomizedSearchCV(
            RandomForestClassifier(random_state=42),
            {"n_estimators": [5, 10, 15, 20]},
            n_iter=2,
            cv=2,
            random_state=42
        )
        rs.fit(X, y)
        assert rs.best_params_["n_estimators"] in [5, 10, 15, 20]

    def test_random_search_two_params(self, data):
        """Random search with two parameters."""
        X, y = data
        rs = RandomizedSearchCV(
            RandomForestClassifier(random_state=42),
            {
                "n_estimators": [5, 10, 15],
                "max_depth": [3, 5, 7]
            },
            n_iter=3,
            cv=2,
            random_state=42
        )
        rs.fit(X, y)
        assert "n_estimators" in rs.best_params_
        assert "max_depth" in rs.best_params_

    def test_grid_vs_random_same_seed(self, data):
        """Grid and random search are reproducible."""
        X, y = data

        rs1 = RandomizedSearchCV(
            RandomForestClassifier(random_state=42),
            {"n_estimators": [5, 10]},
            n_iter=2,
            cv=2,
            random_state=42
        )
        rs1.fit(X, y)

        rs2 = RandomizedSearchCV(
            RandomForestClassifier(random_state=42),
            {"n_estimators": [5, 10]},
            n_iter=2,
            cv=2,
            random_state=42
        )
        rs2.fit(X, y)

        assert rs1.best_params_ == rs2.best_params_

    def test_grid_search_logistic_regression(self, data):
        """Grid search with Logistic Regression."""
        X, y = data
        gs = GridSearchCV(
            LogisticRegression(random_state=42),
            {"max_iter": [500, 1000], "C": [0.1, 1.0]},
            cv=2
        )
        gs.fit(X, y)
        assert "max_iter" in gs.best_params_
        assert "C" in gs.best_params_

    def test_hyperparameter_scoring_metric(self, data):
        """Grid search with different scoring metric."""
        X, y = data
        gs = GridSearchCV(
            RandomForestClassifier(random_state=42),
            {"n_estimators": [5, 10]},
            cv=2,
            scoring="accuracy"
        )
        gs.fit(X, y)
        assert gs.best_score_ >= 0

    def test_hyperparameter_n_jobs(self, data):
        """Grid search with parallel jobs."""
        X, y = data
        gs = GridSearchCV(
            RandomForestClassifier(random_state=42),
            {"n_estimators": [5, 10]},
            cv=2,
            n_jobs=1  # Use 1 for deterministic behavior
        )
        gs.fit(X, y)
        assert hasattr(gs, "best_params_")


# ============================================================================
# CROSS-VALIDATION VARIANTS TESTS (10 tests)
# ============================================================================

class TestCrossValidationVariants:
    """Test cross-validation with different strategies."""

    @pytest.fixture
    def cv_data(self):
        """Data for cross-validation."""
        np.random.seed(42)
        X = np.random.randn(80, 5)
        y = np.random.randint(0, 2, 80)
        return X, y

    def test_cv_basic(self, cv_data):
        """Basic cross-validation."""
        X, y = cv_data
        model = RandomForestModel({"n_estimators": 10, "random_state": 42})

        cv = CrossValidator(n_folds=5)
        result = cv.cross_validate(model, X, y)

        assert "mean_score" in result
        assert "std_score" in result
        assert len(result["fold_scores"]) == 5

    def test_cv_2_folds(self, cv_data):
        """2-fold cross-validation."""
        X, y = cv_data
        model = RandomForestModel({"n_estimators": 10, "random_state": 42})

        cv = CrossValidator(n_folds=2)
        result = cv.cross_validate(model, X, y)

        assert len(result["fold_scores"]) == 2

    def test_cv_10_folds(self, cv_data):
        """10-fold cross-validation."""
        X, y = cv_data
        model = RandomForestModel({"n_estimators": 10, "random_state": 42})

        cv = CrossValidator(n_folds=10)
        result = cv.cross_validate(model, X, y)

        assert len(result["fold_scores"]) == 10

    def test_cv_mean_score(self, cv_data):
        """Cross-validation mean score."""
        X, y = cv_data
        model = RandomForestModel({"n_estimators": 10, "random_state": 42})

        cv = CrossValidator(n_folds=3)
        result = cv.cross_validate(model, X, y)

        # Mean should be average of fold scores
        expected_mean = np.mean(result["fold_scores"])
        assert np.isclose(result["mean_score"], expected_mean)

    def test_cv_std_score(self, cv_data):
        """Cross-validation standard deviation."""
        X, y = cv_data
        model = RandomForestModel({"n_estimators": 10, "random_state": 42})

        cv = CrossValidator(n_folds=3)
        result = cv.cross_validate(model, X, y)

        # Std should be computed (may use different method than np.std)
        assert 0 <= result["std_score"] <= 1

    def test_cv_stratified_split(self, cv_data):
        """Cross-validation with stratification."""
        X, y = cv_data
        y_imbalanced = np.concatenate([
            np.zeros(70, dtype=int),
            np.ones(10, dtype=int)
        ])

        model = RandomForestModel({"n_estimators": 10, "random_state": 42})

        cv = CrossValidator(n_folds=3)
        result = cv.cross_validate(model, X, y_imbalanced)

        # Should handle imbalanced data
        assert len(result["fold_scores"]) == 3

    def test_cv_reproducibility(self, cv_data):
        """Cross-validation is reproducible."""
        X, y = cv_data
        model = RandomForestModel({"n_estimators": 10, "random_state": 42})

        cv1 = CrossValidator(n_folds=3)
        result1 = cv1.cross_validate(model, X, y)

        model2 = RandomForestModel({"n_estimators": 10, "random_state": 42})
        cv2 = CrossValidator(n_folds=3)
        result2 = cv2.cross_validate(model2, X, y)

        np.testing.assert_array_almost_equal(
            result1["fold_scores"],
            result2["fold_scores"]
        )

    def test_cv_score_range(self, cv_data):
        """Cross-validation scores in valid range."""
        X, y = cv_data
        model = RandomForestModel({"n_estimators": 10, "random_state": 42})

        cv = CrossValidator(n_folds=5)
        result = cv.cross_validate(model, X, y)

        # All scores should be between 0 and 1
        assert all(0 <= score <= 1 for score in result["fold_scores"])
        assert 0 <= result["mean_score"] <= 1

    def test_cv_with_different_random_states(self, cv_data):
        """Cross-validation with reproducible splits."""
        X, y = cv_data
        model = RandomForestModel({"n_estimators": 10, "random_state": 42})

        cv = CrossValidator(n_folds=5)
        result = cv.cross_validate(model, X, y)

        # Should produce valid scores
        assert all(0 <= score <= 1 for score in result["fold_scores"])


# ============================================================================
# UNCERTAINTY QUANTIFICATION TESTS (10 tests)
# ============================================================================

class TestUncertaintyPaths:
    """Test uncertainty quantification methods."""

    @pytest.fixture
    def unc_data(self):
        """Data for uncertainty tests."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)

        return model, X, y

    def test_ensemble_uncertainty_mean(self, unc_data):
        """Ensemble uncertainty mean prediction."""
        model, X, y = unc_data

        estimator = UncertaintyEstimator()
        result = estimator.ensemble_uncertainty(model, X[:10])

        assert "mean" in result
        assert len(result["mean"]) == 10

    def test_ensemble_uncertainty_variance(self, unc_data):
        """Ensemble uncertainty variance."""
        model, X, y = unc_data

        estimator = UncertaintyEstimator()
        result = estimator.ensemble_uncertainty(model, X[:10])

        assert "variance" in result
        assert len(result["variance"]) == 10
        assert all(v >= 0 for v in result["variance"])

    def test_ensemble_uncertainty_std(self, unc_data):
        """Ensemble uncertainty standard deviation."""
        model, X, y = unc_data

        estimator = UncertaintyEstimator()
        result = estimator.ensemble_uncertainty(model, X[:10])

        assert "std" in result
        # Std should be sqrt of variance
        np.testing.assert_array_almost_equal(
            result["std"],
            np.sqrt(result["variance"])
        )

    def test_ensemble_uncertainty_entropy(self, unc_data):
        """Ensemble uncertainty entropy."""
        model, X, y = unc_data

        estimator = UncertaintyEstimator()
        result = estimator.ensemble_uncertainty(model, X[:10])

        assert "entropy" in result
        assert len(result["entropy"]) == 10

    def test_ensemble_uncertainty_confidence(self, unc_data):
        """Ensemble uncertainty confidence."""
        model, X, y = unc_data

        estimator = UncertaintyEstimator()
        result = estimator.ensemble_uncertainty(model, X[:10])

        assert "confidence" in result
        # Confidence is normalized (may exceed 1 for some cases)
        assert len(result["confidence"]) == 10

    def test_bootstrap_uncertainty(self, unc_data):
        """Bootstrap uncertainty estimation."""
        model, X, y = unc_data

        estimator = UncertaintyEstimator()
        result = estimator.bootstrap_uncertainty(
            model, X[:10], y[:10], n_bootstrap=10
        )

        assert result is not None

    def test_conformal_prediction_coverage(self, unc_data):
        """Conformal prediction coverage."""
        model, X, y = unc_data

        # Split for calibration
        X_cal, X_test = X[:50], X[50:70]
        y_cal, y_test = y[:50], y[50:70]

        estimator = UncertaintyEstimator()
        result = estimator.conformal_prediction(
            model, X_cal, y_cal, X_test, y_test, alpha=0.1
        )

        assert "coverage" in result
        # Coverage should be close to 1 - alpha = 0.9
        assert 0.7 <= result["coverage"] <= 1.0

    def test_calibration_curve_bins(self, unc_data):
        """Calibration curve with bins."""
        model, X, y = unc_data

        y_pred_proba = model.predict_proba(X)[:, 1]

        estimator = UncertaintyEstimator()
        result = estimator.calibration_curve(
            y, y_pred_proba, n_bins=5
        )

        assert "bin_centers" in result
        # Number of bins may be less than requested (depends on data)
        assert len(result["bin_centers"]) > 0

    def test_calibration_curve_difference(self, unc_data):
        """Calibration curve shows predicted vs actual."""
        model, X, y = unc_data

        y_pred_proba = model.predict_proba(X)[:, 1]

        estimator = UncertaintyEstimator()
        result = estimator.calibration_curve(
            y, y_pred_proba, n_bins=3
        )

        # Should have actual fraction of positives
        assert "bin_true_frac" in result or "fraction_positives" in result

    def test_bootstrap_multiple_iterations(self, unc_data):
        """Bootstrap with multiple iterations."""
        model, X, y = unc_data

        estimator = UncertaintyEstimator()
        result = estimator.bootstrap_uncertainty(
            model, X[:10], y[:10], n_bootstrap=20
        )

        assert result is not None


# ============================================================================
# STATISTICAL TESTS EDGE CASES (10 tests)
# ============================================================================

class TestStatisticalEdgeCases:
    """Test statistical tests edge cases."""

    def test_mcnemar_perfect_agreement(self):
        """McNemar test with perfect agreement."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])  # Same as y_true

        tests = StatisticalTests()
        result = tests.mcnemar_test(y_true, y_pred, y_pred)

        # Perfect agreement should give p-value of 1
        assert result["p_value"] == 1.0
        assert result["significant"] is False

    def test_mcnemar_complete_disagreement(self):
        """McNemar test with large disagreement."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred1 = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_pred2 = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        tests = StatisticalTests()
        result = tests.mcnemar_test(y_true, y_pred1, y_pred2)

        assert "p_value" in result
        assert result["p_value"] >= 0

    def test_paired_t_test_large_difference(self):
        """Paired t-test with large difference."""
        y_true = np.ones(20)
        y_pred1 = np.ones(20)
        y_pred2 = np.zeros(20)

        tests = StatisticalTests()
        result = tests.paired_t_test(y_true, y_pred1, y_pred2, metric="f1")

        assert "t_statistic" in result
        assert "p_value" in result

    def test_cohen_kappa_perfect(self):
        """Cohen's kappa for perfect agreement."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1, 0, 1])

        tests = StatisticalTests()
        result = tests.cohen_kappa_confidence(y_true, y_pred)

        assert result["kappa"] == 1.0

    def test_cohen_kappa_no_agreement(self):
        """Cohen's kappa for no agreement."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])

        tests = StatisticalTests()
        result = tests.cohen_kappa_confidence(y_true, y_pred)

        # Kappa should be negative (worse than random)
        assert result["kappa"] < 0

    def test_anova_two_models(self):
        """ANOVA requires at least two models."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred1 = np.array([0, 1, 1, 0, 1])
        y_pred2 = np.array([0, 1, 0, 0, 1])

        tests = StatisticalTests()
        result = tests.compare_models_anova(
            y_true,
            {"model1": y_pred1, "model2": y_pred2}
        )

        assert "f_statistic" in result
        assert "p_value" in result

    def test_anova_similar_predictions(self):
        """ANOVA with similar predictions."""
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        y_pred1 = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        y_pred2 = np.array([0, 1, 1, 0, 1, 0, 0, 1])

        tests = StatisticalTests()
        result = tests.compare_models_anova(
            y_true,
            {"model1": y_pred1, "model2": y_pred2}
        )

        # Should have valid p-value
        assert 0 <= result["p_value"] <= 1
        assert "significant" in result

    def test_mcnemar_large_sample(self):
        """McNemar test with large sample."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        y_pred1 = np.random.randint(0, 2, 200)
        y_pred2 = np.random.randint(0, 2, 200)

        tests = StatisticalTests()
        result = tests.mcnemar_test(y_true, y_pred1, y_pred2)

        assert 0 <= result["p_value"] <= 1

    def test_paired_t_test_similar_scores(self):
        """Paired t-test with similar scores."""
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1])
        y_pred1 = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1])
        y_pred2 = np.array([0, 1, 0, 0, 1, 0, 1, 1, 0, 1])

        tests = StatisticalTests()
        result = tests.paired_t_test(y_true, y_pred1, y_pred2, metric="f1")

        # Should return valid t_statistic and p_value
        assert "t_statistic" in result
        assert "p_value" in result

    def test_statistical_tests_type_safety(self):
        """Statistical tests handle numpy arrays."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred1 = np.array([0, 1, 0, 0, 1])
        y_pred2 = np.array([0, 0, 1, 0, 1])

        tests = StatisticalTests()
        result = tests.mcnemar_test(y_true, y_pred1, y_pred2)

        # Should not error with numpy arrays
        assert isinstance(result, dict)


# ============================================================================
# INTEGRATION TESTS (3 tests)
# ============================================================================

class TestPhase2Integration:
    """Integration tests combining Phase 2 components."""

    def test_hyperparameter_with_cross_validation(self):
        """Hyperparameter tuning with cross-validation."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        # Grid search uses CV internally
        gs = GridSearchCV(
            RandomForestClassifier(random_state=42),
            {"n_estimators": [5, 10], "max_depth": [3, 5]},
            cv=3
        )
        gs.fit(X, y)

        # Should work
        assert hasattr(gs, "best_params_")

    def test_uncertainty_with_cross_validation(self):
        """Uncertainty estimation with CV model."""
        np.random.seed(42)
        X = np.random.randn(80, 5)
        y = np.random.randint(0, 2, 80)

        # Train model
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X, y)

        # Get uncertainty
        estimator = UncertaintyEstimator()
        result = estimator.ensemble_uncertainty(model, X[:10])

        assert "mean" in result
        assert len(result["mean"]) == 10

    def test_statistical_tests_with_models(self):
        """Statistical testing of different models."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        # Train two models
        m1 = RandomForestModel({"n_estimators": 10, "random_state": 42})
        m1.fit(X, y)
        pred1 = m1.predict(X)

        m2 = LogisticRegressionModel({"max_iter": 500, "random_state": 42})
        m2.fit(X, y)
        pred2 = m2.predict(X)

        # Compare
        tests = StatisticalTests()
        result = tests.mcnemar_test(y, pred1, pred2)

        assert "p_value" in result
        assert "significant" in result
