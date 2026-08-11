"""
Tests for data validator.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.validator import DataValidator


class TestDataValidator:
    """Test DataValidator class."""

    def test_validate_valid_data(self):
        """Test validation of valid data."""
        df = pd.DataFrame(
            {
                "virus_species": ["BDBV", "EBOV", "SUDV"],
                "host_species": [
                    "Homo_sapiens",
                    "Pan_troglodytes",
                    "Rousettus_aegyptiacus",
                ],
                "infection_status": [1, 0, 1],
            }
        )

        validator = DataValidator()
        results = validator.validate(df)

        assert results["valid"] is True
        assert "n_samples" in results["statistics"]
        assert results["statistics"]["n_samples"] == 3

    def test_validate_missing_columns(self):
        """Test validation with missing columns."""
        df = pd.DataFrame(
            {
                "virus_species": ["BDBV", "EBOV"],
                "host_species": ["Homo_sapiens", "Pan_troglodytes"],
            }
        )

        validator = DataValidator()
        results = validator.validate(df)

        assert results["valid"] is False
        assert len(results["errors"]) > 0

    def test_validate_class_imbalance(self):
        """Test class imbalance warning."""
        df = pd.DataFrame(
            {
                "virus_species": ["BDBV"] * 10 + ["EBOV"] * 90,
                "host_species": ["Homo_sapiens"] * 100,
                "infection_status": [1] * 5 + [0] * 95,
            }
        )

        validator = DataValidator()
        results = validator.validate(df)

        assert len(results["warnings"]) > 0

    def test_validate_missing_values(self):
        """Test missing values warning."""
        df = pd.DataFrame(
            {
                "virus_species": ["BDBV", "EBOV", None],
                "host_species": ["Homo_sapiens", None, "Rousettus"],
                "infection_status": [1, 0, 1],
            }
        )

        validator = DataValidator()
        results = validator.validate(df)

        # Should have warnings about missing values
        assert len(results["warnings"]) > 0

    def test_validate_duplicates(self):
        """Test duplicate rows detection."""
        df = pd.DataFrame(
            {
                "virus_species": ["BDBV", "BDBV", "EBOV"],
                "host_species": ["Homo_sapiens", "Homo_sapiens", "Pan"],
                "infection_status": [1, 1, 0],
            }
        )

        validator = DataValidator()
        results = validator.validate(df)

        assert results["statistics"]["duplicates"] == 1

    def test_validate_outliers(self):
        """Test outlier detection."""
        df = pd.DataFrame(
            {
                "virus_species": ["BDBV"] * 10,
                "host_species": ["Homo"] * 10,
                "infection_status": [1] * 10,
                "body_mass_kg": [70] * 9 + [10000],  # Outlier
                "lifespan_years": [80] * 10,
            }
        )

        validator = DataValidator()
        results = validator.validate(df)

        # Should have warnings about outliers
        assert len(results["warnings"]) > 0

    def test_validate_bdbv_specific(self):
        """Test BDBV-specific validation."""
        df = pd.DataFrame(
            {
                "virus_species": ["BDBV", "BDBV"],
                "host_species": ["Homo_sapiens", "Rousettus"],
                "infection_status": [1, 0],
                "bdbv_gp_binding_affinity": [-8.2, -7.5],
                "bdbv_npc1_residue_match": [0.92, 0.85],
            }
        )

        validator = DataValidator()
        valid, errors = validator.validate_for_bdbv(df)

        assert valid is True
        assert len(errors) == 0

    def test_validate_bdbv_missing_features(self):
        """Test BDBV validation with missing features."""
        df = pd.DataFrame(
            {
                "virus_species": ["BDBV", "BDBV"],
                "host_species": ["Homo_sapiens", "Rousettus"],
                "infection_status": [1, 0],
            }
        )

        validator = DataValidator()
        valid, errors = validator.validate_for_bdbv(df)

        assert valid is False
        assert len(errors) > 0
