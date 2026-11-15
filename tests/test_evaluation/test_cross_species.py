"""
Tests for cross-species generalization analysis.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.evaluation.cross_species import CrossSpeciesValidator


class TestCrossSpeciesValidator:
    """Test cross-species validation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with multiple species."""
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 5)
        y = np.random.randint(0, 2, n_samples)
        species = np.repeat(["Species_A", "Species_B", "Species_C", "Species_D"], 25)
        return X, y, species

    def test_leave_one_species_out(self, sample_data):
        """Test LOSO cross-validation."""
        X, y, species = sample_data

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        validator = CrossSpeciesValidator()

        results = validator.leave_one_species_out(model, X, y, species)

        assert "per_species" in results
        assert "overall" in results
        assert "loso_predictions" in results
        assert "loso_probabilities" in results

        # Check per-species dataframe
        per_species_df = results["per_species"]
        assert len(per_species_df) == 4  # 4 unique species
        assert "species" in per_species_df.columns
        assert "accuracy" in per_species_df.columns
        assert "f1" in per_species_df.columns

    def test_per_species_performance(self, sample_data):
        """Test per-species performance calculation."""
        X, y, species = sample_data
        y_pred = np.random.randint(0, 2, len(y))

        result_df = CrossSpeciesValidator.per_species_performance(
            species, y, y_pred
        )

        assert len(result_df) == 4
        assert "species" in result_df.columns
        assert "n_samples" in result_df.columns
        assert "accuracy" in result_df.columns
        assert all(result_df["n_samples"] == 25)

    def test_with_probabilities(self, sample_data):
        """Test per-species performance with probability predictions."""
        X, y, species = sample_data
        y_pred = np.random.randint(0, 2, len(y))
        y_proba = np.random.rand(len(y), 2)
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)  # Normalize

        result_df = CrossSpeciesValidator.per_species_performance(
            species, y, y_pred, y_proba
        )

        assert "roc_auc" in result_df.columns
