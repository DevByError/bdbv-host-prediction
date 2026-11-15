"""
Tests for baseline models.
"""

import pytest
import numpy as np
import pandas as pd
from src.models.baselines import (
    MajorityClassBaseline,
    RandomBaseline,
    PhylogeneticDistanceBaseline,
    EcologicalSimilarityBaseline,
)


class TestMajorityClassBaseline:
    """Test majority class baseline."""

    def test_fit_and_predict(self):
        """Test fitting and prediction."""
        X = np.random.randn(100, 5)
        y = np.array([0] * 70 + [1] * 30)

        model = MajorityClassBaseline()
        model.fit(X, y)

        assert model._is_fitted
        assert model.majority_class == 0

        preds = model.predict(X)
        assert len(preds) == 100
        assert np.all(preds == 0)

    def test_predict_proba(self):
        """Test probability predictions."""
        X = np.random.randn(50, 3)
        y = np.array([0] * 40 + [1] * 10)

        model = MajorityClassBaseline()
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert proba.shape == (50, 2)
        assert np.allclose(proba[:, 0], 1.0)
        assert np.allclose(proba[:, 1], 0.0)


class TestRandomBaseline:
    """Test random baseline."""

    def test_fit_and_predict(self):
        """Test fitting and prediction."""
        X = np.random.randn(100, 5)
        y = np.array([0] * 70 + [1] * 30)

        model = RandomBaseline(random_state=42)
        model.fit(X, y)

        assert model._is_fitted
        assert model.class_proportions is not None

        preds = model.predict(X)
        assert len(preds) == 100
        assert set(preds) <= {0, 1}

    def test_predict_proba_shape(self):
        """Test probability prediction shape."""
        X = np.random.randn(50, 3)
        y = np.array([0] * 30 + [1] * 20)

        model = RandomBaseline(random_state=42)
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert proba.shape == (50, 2)
        assert np.all((proba >= 0) & (proba <= 1))


class TestPhylogeneticDistanceBaseline:
    """Test phylogenetic distance baseline."""

    def test_fit_and_predict(self):
        """Test fitting with sequences."""
        # Create toy sequence data
        sequences = [
            "ATGCGATCG",  # Similar to positive
            "ATGCGATCG",  # Positive
            "GCTAGCTAG",  # Dissimilar
            "ATGCGATCG",  # Similar to positive
        ]
        X = pd.DataFrame({"virus_sequence": sequences})
        y = np.array([1, 1, 0, 1])

        model = PhylogeneticDistanceBaseline()
        model.fit(X, y)

        assert model._is_fitted
        assert len(model.positive_sequences) == 3

    def test_predict_similarity(self):
        """Test similarity calculation."""
        seq1 = "ATGC"
        seq2 = "ATGC"
        sim = PhylogeneticDistanceBaseline._hamming_similarity(seq1, seq2)
        assert sim == 1.0

        seq3 = "GCTA"
        sim = PhylogeneticDistanceBaseline._hamming_similarity(seq1, seq3)
        assert 0 <= sim < 1

    def test_length_mismatch(self):
        """Test sequences of different lengths."""
        sim = PhylogeneticDistanceBaseline._hamming_similarity("ATGC", "AT")
        assert sim == 0.0


class TestEcologicalSimilarityBaseline:
    """Test ecological similarity baseline."""

    def test_fit_with_ecological_features(self):
        """Test fitting with ecological data."""
        X = pd.DataFrame({
            "body_mass": [100.0, 200.0, 150.0, 120.0],
            "lifespan": [10.0, 20.0, 15.0, 12.0],
            "geographic_range": [1000.0, 500.0, 800.0, 600.0],
        })
        y = np.array([1, 1, 0, 0])

        model = EcologicalSimilarityBaseline(ecological_features=["body_mass", "lifespan", "geographic_range"])
        model.fit(X, y)

        assert model._is_fitted
        assert model.positive_ecology is not None
        assert len(model.positive_ecology) > 0

    def test_predict_shape(self):
        """Test prediction shape."""
        X = pd.DataFrame({
            "body_mass": [100, 200, 150],
            "lifespan": [10, 20, 15],
        })
        y = np.array([1, 1, 0])

        model = EcologicalSimilarityBaseline(ecological_features=["body_mass", "lifespan"])
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert proba.shape == (3, 2)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_missing_features(self):
        """Test error on missing features."""
        X = pd.DataFrame({"some_feature": [1, 2, 3]})
        y = np.array([0, 1, 0])

        model = EcologicalSimilarityBaseline(ecological_features=["body_mass"])

        with pytest.raises(ValueError):
            model.fit(X, y)
