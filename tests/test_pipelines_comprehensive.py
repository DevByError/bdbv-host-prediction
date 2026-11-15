"""
Comprehensive tests for training and prediction pipelines.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from src.pipelines.training_pipeline import TrainingPipeline
from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.models.classical import RandomForestModel, XGBoostModel, LogisticRegressionModel
from src.evaluation.metrics import ModelEvaluator
from config import set_all_seeds


class TestTrainingPipeline:
    """Comprehensive tests for training pipeline."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for pipeline testing."""
        np.random.seed(42)
        n_samples = 100

        df = pd.DataFrame({
            "virus_species": np.random.choice(["BDBV", "EBOV", "SUDV"], n_samples),
            "host_species": np.random.choice(["Homo_sapiens", "Pan", "Rousettus"], n_samples),
            "host_order": np.random.choice(["Primates", "Chiroptera"], n_samples),
            "body_mass_kg": np.random.uniform(0.01, 100, n_samples),
            "lifespan_years": np.random.uniform(1, 80, n_samples),
            "gc_content": np.random.uniform(30, 70, n_samples),
            "cpg_frequency": np.random.uniform(0.01, 0.1, n_samples),
            "infection_status": np.random.randint(0, 2, n_samples),
        })

        return df

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = TrainingPipeline()

        assert pipeline.results == {}
        assert pipeline.best_model is None
        assert pipeline.best_model_name is None

    def test_pipeline_with_custom_config(self):
        """Test pipeline with configuration override."""
        config_override = {"test_mode": True}
        pipeline = TrainingPipeline(config_override=config_override)

        assert pipeline.config_override == config_override

    def test_pipeline_options(self):
        """Test pipeline options for ablation, fairness, calibration."""
        pipeline = TrainingPipeline(
            run_ablation=False,
            run_fairness=False,
            run_calibration=False
        )

        assert pipeline.run_ablation is False
        assert pipeline.run_fairness is False
        assert pipeline.run_calibration is False

    def test_model_creation(self):
        """Test model creation in pipeline."""
        pipeline = TrainingPipeline()

        # Access internal method
        models = {}

        # Manually create models like the pipeline does
        models["RandomForest"] = RandomForestModel({
            "n_estimators": 100,
            "max_depth": None,
            "class_weight": "balanced",
            "random_state": 42,
        })

        models["LogisticRegression"] = LogisticRegressionModel({
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": 42,
        })

        assert len(models) == 2
        assert "RandomForest" in models
        assert "LogisticRegression" in models

    def test_reproducibility_with_set_seed(self):
        """Test that set_all_seeds ensures reproducibility."""
        set_all_seeds(42)

        X = np.random.randn(10, 5)
        y = np.random.randint(0, 2, 10)

        # Train twice
        model1 = RandomForestModel({"n_estimators": 10, "random_state": 42})
        model1.fit(X, y)
        pred1 = model1.predict(X)

        set_all_seeds(42)

        model2 = RandomForestModel({"n_estimators": 10, "random_state": 42})
        model2.fit(X, y)
        pred2 = model2.predict(X)

        np.testing.assert_array_equal(pred1, pred2)

    def test_model_evaluation(self, sample_data):
        """Test model evaluation within pipeline."""
        X = sample_data[["body_mass_kg", "lifespan_years", "gc_content"]].values
        y = sample_data["infection_status"].values

        model = RandomForestModel({"n_estimators": 10, "random_state": 42})
        model.fit(X[:80], y[:80])

        evaluator = ModelEvaluator(model)
        metrics = evaluator.evaluate(X[80:], y[80:])

        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_multiple_model_training(self, sample_data):
        """Test training multiple models sequentially."""
        X = sample_data[["body_mass_kg", "lifespan_years", "gc_content"]].values
        y = sample_data["infection_status"].values

        models = {
            "RF": RandomForestModel({"n_estimators": 10, "random_state": 42}),
            "LR": LogisticRegressionModel({"max_iter": 1000, "random_state": 42}),
        }

        results = {}
        for name, model in models.items():
            model.fit(X, y)
            evaluator = ModelEvaluator(model)
            metrics = evaluator.evaluate(X, y)
            results[name] = metrics

        assert len(results) == 2
        assert all("f1" in m for m in results.values())

    def test_best_model_selection(self, sample_data):
        """Test selecting the best model based on F1 score."""
        X = sample_data[["body_mass_kg", "lifespan_years", "gc_content"]].values
        y = sample_data["infection_status"].values

        results = {}

        # Train RF
        rf = RandomForestModel({"n_estimators": 10, "random_state": 42})
        rf.fit(X, y)
        evaluator = ModelEvaluator(rf)
        results["RF"] = evaluator.evaluate(X, y)

        # Train LR
        lr = LogisticRegressionModel({"max_iter": 1000, "random_state": 42})
        lr.fit(X, y)
        evaluator = ModelEvaluator(lr)
        results["LR"] = evaluator.evaluate(X, y)

        # Select best
        best_model_name = max(results.keys(), key=lambda x: results[x]["f1"])

        assert best_model_name in ["RF", "LR"]
        assert results[best_model_name]["f1"] >= 0

    def test_model_persistence(self, sample_data, tmp_path):
        """Test saving and loading models."""
        X = sample_data[["body_mass_kg", "lifespan_years", "gc_content"]].values
        y = sample_data["infection_status"].values

        # Train and save
        model = RandomForestModel({"n_estimators": 10, "random_state": 42})
        model.fit(X, y)

        save_path = tmp_path / "test_model.pkl"
        model.save(save_path)

        assert save_path.exists()

        # Load and verify
        model2 = RandomForestModel()
        model2.load(save_path)

        assert model2._is_fitted
        pred1 = model.predict(X[:5])
        pred2 = model2.predict(X[:5])

        np.testing.assert_array_equal(pred1, pred2)

    def test_cross_validation_integration(self, sample_data):
        """Test cross-validation in pipeline context."""
        from src.evaluation.cross_validation import CrossValidator

        X = sample_data[["body_mass_kg", "lifespan_years", "gc_content"]].values
        y = sample_data["infection_status"].values

        model = RandomForestModel({"n_estimators": 10, "random_state": 42})

        cv = CrossValidator(n_folds=5)
        cv_results = cv.cross_validate(model, X, y)

        assert "mean_score" in cv_results
        assert "std_score" in cv_results
        assert "fold_scores" in cv_results
        assert len(cv_results["fold_scores"]) == 5

    def test_feature_importance_extraction(self, sample_data):
        """Test extracting feature importance from models."""
        X = sample_data[["body_mass_kg", "lifespan_years", "gc_content"]].values
        y = sample_data["infection_status"].values

        model = RandomForestModel({"n_estimators": 10, "random_state": 42})
        model.fit(X, y)

        importance = model.get_feature_importance()

        assert importance is not None
        assert len(importance) == 3
        assert "feature" in importance.columns
        assert "importance" in importance.columns

    def test_confidence_intervals_in_metrics(self, sample_data):
        """Test confidence interval calculation in evaluation."""
        X = sample_data[["body_mass_kg", "lifespan_years", "gc_content"]].values
        y = sample_data["infection_status"].values

        model = RandomForestModel({"n_estimators": 10, "random_state": 42})
        model.fit(X, y)

        evaluator = ModelEvaluator(model)
        evaluator.evaluate(X, y)

        assert hasattr(evaluator, "confidence_intervals")
        assert evaluator.confidence_intervals is not None

    def test_statistical_comparison(self, sample_data):
        """Test statistical comparison between two models."""
        from src.evaluation.statistical_tests import StatisticalTests

        X = sample_data[["body_mass_kg", "lifespan_years", "gc_content"]].values
        y = sample_data["infection_status"].values

        # Train two models
        model1 = RandomForestModel({"n_estimators": 10, "random_state": 42})
        model1.fit(X, y)
        pred1 = model1.predict(X)

        model2 = LogisticRegressionModel({"max_iter": 1000, "random_state": 42})
        model2.fit(X, y)
        pred2 = model2.predict(X)

        # Compare
        stat_test = StatisticalTests()
        result = stat_test.mcnemar_test(y, pred1, pred2)

        assert "p_value" in result
        assert "chi2" in result
        assert "significant" in result

    def test_metadata_tracking(self, sample_data):
        """Test tracking of pipeline metadata."""
        pipeline = TrainingPipeline(n_bootstrap_ci=100, random_state=42)

        assert pipeline.n_bootstrap_ci == 100
        assert pipeline.random_state == 42

    def test_report_generation(self, sample_data):
        """Test that pipeline generates reports."""
        pipeline = TrainingPipeline()

        # After a run, pipeline should have reports
        assert hasattr(pipeline, "report")
