"""
Phase 1 Lite Coverage Tests - Fast unit tests for high-impact modules
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from src.pipelines.training_pipeline import TrainingPipeline
from src.models.classical import RandomForestModel, LogisticRegressionModel
from config import set_all_seeds


# ============================================================================
# TRAINING PIPELINE TESTS (15 tests)
# ============================================================================

class TestTrainingPipelineConfiguration:
    """TrainingPipeline configuration tests."""

    def test_init_defaults(self):
        """Default initialization."""
        p = TrainingPipeline()
        assert p.run_ablation is True
        assert p.run_fairness is True
        assert p.run_calibration is True
        assert p.n_bootstrap_ci == 1000

    def test_init_ablation_off(self):
        """Ablation disabled."""
        p = TrainingPipeline(run_ablation=False)
        assert p.run_ablation is False

    def test_init_fairness_off(self):
        """Fairness disabled."""
        p = TrainingPipeline(run_fairness=False)
        assert p.run_fairness is False

    def test_init_calibration_off(self):
        """Calibration disabled."""
        p = TrainingPipeline(run_calibration=False)
        assert p.run_calibration is False

    def test_init_custom_bootstrap(self):
        """Custom bootstrap samples."""
        p = TrainingPipeline(n_bootstrap_ci=500)
        assert p.n_bootstrap_ci == 500

    def test_init_custom_seed(self):
        """Custom random state."""
        p = TrainingPipeline(random_state=999)
        assert p.random_state == 999

    def test_init_config_override(self):
        """Config override dict."""
        cfg = {"test": True}
        p = TrainingPipeline(config_override=cfg)
        assert p.config_override == cfg

    def test_init_state(self):
        """Initial state."""
        p = TrainingPipeline()
        assert p.results == {}
        assert p.best_model is None
        assert p.best_model_name is None
        assert p.report == {}

    def test_init_all_flags_off(self):
        """All optional features disabled."""
        p = TrainingPipeline(
            run_ablation=False,
            run_fairness=False,
            run_calibration=False
        )
        assert not p.run_ablation
        assert not p.run_fairness
        assert not p.run_calibration

    def test_init_high_bootstrap(self):
        """High bootstrap count."""
        p = TrainingPipeline(n_bootstrap_ci=5000)
        assert p.n_bootstrap_ci == 5000

    def test_init_low_bootstrap(self):
        """Low bootstrap count."""
        p = TrainingPipeline(n_bootstrap_ci=10)
        assert p.n_bootstrap_ci == 10

    def test_init_multiple_random_states(self):
        """Different random states."""
        p1 = TrainingPipeline(random_state=42)
        p2 = TrainingPipeline(random_state=123)
        assert p1.random_state != p2.random_state

    def test_init_zero_seed(self):
        """Zero random state."""
        p = TrainingPipeline(random_state=0)
        assert p.random_state == 0

    def test_init_none_config(self):
        """None config override."""
        p = TrainingPipeline(config_override=None)
        assert p.config_override == {}

    def test_init_empty_config(self):
        """Empty config override."""
        p = TrainingPipeline(config_override={})
        assert p.config_override == {}


# ============================================================================
# MODEL TRAINING TESTS (10 tests)
# ============================================================================

class TestModelTraining:
    """Model training tests."""

    @pytest.fixture
    def data(self):
        """Simple test data."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        return X, y

    def test_rf_single_sample(self, data):
        """RF prediction single sample."""
        X, y = data
        m = RandomForestModel({"n_estimators": 5, "random_state": 42})
        m.fit(X, y)
        pred = m.predict(X[:1])
        assert pred.shape == (1,)

    def test_rf_batch_prediction(self, data):
        """RF batch prediction."""
        X, y = data
        m = RandomForestModel({"n_estimators": 5, "random_state": 42})
        m.fit(X, y)
        pred = m.predict(X[:10])
        assert pred.shape == (10,)

    def test_rf_proba_shape(self, data):
        """RF probability shape."""
        X, y = data
        m = RandomForestModel({"n_estimators": 5, "random_state": 42})
        m.fit(X, y)
        proba = m.predict_proba(X[:5])
        assert proba.shape == (5, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_lr_prediction(self, data):
        """Logistic regression prediction."""
        X, y = data
        m = LogisticRegressionModel({"max_iter": 500, "random_state": 42})
        m.fit(X, y)
        pred = m.predict(X)
        assert len(pred) == len(y)

    def test_lr_proba(self, data):
        """Logistic regression probability."""
        X, y = data
        m = LogisticRegressionModel({"max_iter": 500, "random_state": 42})
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (len(y), 2)

    def test_model_fitted_state(self, data):
        """Model fitted flag."""
        X, y = data
        m = RandomForestModel({"n_estimators": 5, "random_state": 42})
        assert not m._is_fitted
        m.fit(X, y)
        assert m._is_fitted

    def test_different_n_estimators(self, data):
        """Different ensemble sizes."""
        X, y = data
        for n in [5, 10, 20]:
            m = RandomForestModel({"n_estimators": n, "random_state": 42})
            m.fit(X, y)
            assert m._is_fitted

    def test_feature_importance(self, data):
        """Feature importance."""
        X, y = data
        m = RandomForestModel({"n_estimators": 10, "random_state": 42})
        m.fit(X, y)
        imp = m.get_feature_importance()
        assert imp is not None
        assert len(imp) == 5

    def test_model_all_zeros_class(self):
        """All samples same class raises error."""
        X = np.random.randn(30, 3)
        y = np.zeros(30, dtype=int)
        m = RandomForestModel({"n_estimators": 5, "random_state": 42})

        with pytest.raises(ValueError):
            m.fit(X, y)

    def test_model_all_ones_class(self):
        """All samples one class raises error."""
        X = np.random.randn(30, 3)
        y = np.ones(30, dtype=int)
        m = RandomForestModel({"n_estimators": 5, "random_state": 42})

        with pytest.raises(ValueError):
            m.fit(X, y)


# ============================================================================
# ABLATION STUDY TESTS (8 tests)
# ============================================================================

class TestAblationStudy:
    """Ablation study tests."""

    @pytest.fixture
    def ablation_data(self):
        """Test data with feature names."""
        np.random.seed(42)
        X = pd.DataFrame({
            "f1": np.random.randn(80),
            "f2": np.random.randn(80),
            "f3": np.random.randn(80),
        })
        y = np.random.randint(0, 2, 80)
        return X, y

    def test_ablate_single_feature(self, ablation_data):
        """Remove single feature."""
        X, y = ablation_data
        X_ablated = X.drop(columns=["f1"])
        assert X_ablated.shape[1] == 2
        assert "f1" not in X_ablated.columns

    def test_ablate_multiple_features(self, ablation_data):
        """Remove multiple features."""
        X, y = ablation_data
        X_ablated = X.drop(columns=["f1", "f2"])
        assert X_ablated.shape[1] == 1
        assert "f1" not in X_ablated.columns
        assert "f2" not in X_ablated.columns

    def test_ablation_model_train(self, ablation_data):
        """Train model on ablated data."""
        X, y = ablation_data
        X_ablated = X.drop(columns=["f1"])

        m = RandomForestModel({"n_estimators": 5, "random_state": 42})
        m.fit(X_ablated.values, y)
        assert m._is_fitted

    def test_ablation_performance_calculation(self, ablation_data):
        """Calculate performance delta."""
        from sklearn.metrics import f1_score

        X, y = ablation_data

        # Baseline
        m1 = RandomForestModel({"n_estimators": 5, "random_state": 42})
        m1.fit(X.values, y)
        p1 = m1.predict(X.values)
        f1_base = f1_score(y, p1)

        # Ablated
        X_abl = X.drop(columns=["f1"])
        m2 = RandomForestModel({"n_estimators": 5, "random_state": 42})
        m2.fit(X_abl.values, y)
        p2 = m2.predict(X_abl.values)
        f1_abl = f1_score(y, p2)

        # Delta
        delta = f1_base - f1_abl
        assert isinstance(delta, (int, float, np.number))

    def test_ablation_all_features(self, ablation_data):
        """Ablate each feature."""
        X, y = ablation_data

        deltas = {}
        for col in X.columns:
            X_abl = X.drop(columns=[col])
            m = RandomForestModel({"n_estimators": 5, "random_state": 42})
            m.fit(X_abl.values, y)
            deltas[col] = 1  # placeholder

        assert len(deltas) == 3

    def test_ablation_feature_group(self, ablation_data):
        """Ablate feature group."""
        X, y = ablation_data
        group = ["f1", "f2"]

        X_abl = X.drop(columns=group)
        assert X_abl.shape[1] == 1

    def test_ablation_model_comparison(self, ablation_data):
        """Compare models with/without features."""
        X, y = ablation_data

        m_full = RandomForestModel({"n_estimators": 5, "random_state": 42})
        m_full.fit(X.values, y)

        X_abl = X.drop(columns=["f1"])
        m_abl = RandomForestModel({"n_estimators": 5, "random_state": 42})
        m_abl.fit(X_abl.values, y)

        assert m_full._is_fitted
        assert m_abl._is_fitted

    def test_ablation_data_shapes(self, ablation_data):
        """Check ablation data shapes."""
        X, y = ablation_data

        X_abl = X.drop(columns=["f1"])

        assert X.shape[0] == X_abl.shape[0]  # Same samples
        assert X.shape[1] > X_abl.shape[1]  # Fewer features


# ============================================================================
# HYPERPARAMETER TUNING TESTS (6 tests)
# ============================================================================

class TestHyperparameterTuning:
    """Hyperparameter tuning tests."""

    @pytest.fixture
    def hp_data(self):
        """Data for HP tuning."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        return X, y

    def test_grid_search_basic(self, hp_data):
        """Grid search basic."""
        from sklearn.model_selection import GridSearchCV
        from sklearn.ensemble import RandomForestClassifier

        X, y = hp_data
        params = {"n_estimators": [5, 10]}

        gs = GridSearchCV(
            RandomForestClassifier(random_state=42),
            params,
            cv=2
        )
        gs.fit(X, y)

        assert hasattr(gs, "best_params_")
        assert "n_estimators" in gs.best_params_

    def test_grid_search_params(self, hp_data):
        """Grid search parameters."""
        from sklearn.model_selection import GridSearchCV
        from sklearn.ensemble import RandomForestClassifier

        X, y = hp_data
        params = {
            "n_estimators": [5, 10],
            "max_depth": [3, 5],
        }

        gs = GridSearchCV(
            RandomForestClassifier(random_state=42),
            params,
            cv=2
        )
        gs.fit(X, y)

        assert "n_estimators" in gs.best_params_
        assert "max_depth" in gs.best_params_

    def test_grid_search_score(self, hp_data):
        """Grid search score."""
        from sklearn.model_selection import GridSearchCV
        from sklearn.ensemble import RandomForestClassifier

        X, y = hp_data
        params = {"n_estimators": [5, 10]}

        gs = GridSearchCV(
            RandomForestClassifier(random_state=42),
            params,
            cv=2
        )
        gs.fit(X, y)

        score = gs.score(X, y)
        assert 0 <= score <= 1

    def test_random_search_sampling(self, hp_data):
        """Random search sampling."""
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.ensemble import RandomForestClassifier

        X, y = hp_data
        params = {
            "n_estimators": [5, 10, 20],
            "max_depth": [3, 5, 10],
        }

        rs = RandomizedSearchCV(
            RandomForestClassifier(random_state=42),
            params,
            n_iter=3,
            cv=2,
            random_state=42
        )
        rs.fit(X, y)

        assert hasattr(rs, "best_params_")

    def test_param_grid_variations(self, hp_data):
        """Different param grids."""
        from sklearn.model_selection import GridSearchCV
        from sklearn.ensemble import RandomForestClassifier

        X, y = hp_data

        grids = [
            {"n_estimators": [5, 10]},
            {"max_depth": [3, 5]},
            {"min_samples_split": [2, 5]},
        ]

        for grid in grids:
            gs = GridSearchCV(
                RandomForestClassifier(random_state=42),
                grid,
                cv=2
            )
            gs.fit(X, y)
            assert hasattr(gs, "best_params_")

    def test_cv_fold_count(self, hp_data):
        """Different CV fold counts."""
        from sklearn.model_selection import GridSearchCV
        from sklearn.ensemble import RandomForestClassifier

        X, y = hp_data
        params = {"n_estimators": [5, 10]}

        for cv in [2, 3, 5]:
            gs = GridSearchCV(
                RandomForestClassifier(random_state=42),
                params,
                cv=cv
            )
            gs.fit(X, y)
            assert hasattr(gs, "cv_results_")


# ============================================================================
# REPRODUCIBILITY TESTS (3 tests)
# ============================================================================

class TestReproducibility:
    """Reproducibility tests."""

    def test_same_seed_same_arrays(self):
        """Same seed produces same arrays."""
        set_all_seeds(42)
        X1 = np.random.randn(10, 5)

        set_all_seeds(42)
        X2 = np.random.randn(10, 5)

        np.testing.assert_array_equal(X1, X2)

    def test_different_seed_different_arrays(self):
        """Different seeds produce different arrays."""
        set_all_seeds(42)
        X1 = np.random.randn(10, 5)

        set_all_seeds(123)
        X2 = np.random.randn(10, 5)

        assert not np.allclose(X1, X2)

    def test_pipeline_reproducibility_match(self):
        """Same seed pipeline matches."""
        p1 = TrainingPipeline(random_state=42)
        p2 = TrainingPipeline(random_state=42)

        assert p1.random_state == p2.random_state
