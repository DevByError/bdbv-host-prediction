"""
Tests for feature engineering.
"""

import pytest
import pandas as pd
import numpy as np
from src.features.engineering import FeatureEngineer


class TestFeatureEngineer:
    """Test FeatureEngineer class."""

    def test_engineer_features(self):
        """Test feature engineering pipeline."""
        df = pd.DataFrame(
            {
                "virus_species": ["BDBV", "EBOV", "SUDV"],
                "host_species": ["Homo_sapiens", "Pan_troglodytes", "Rousettus"],
                "host_order": ["Primates", "Primates", "Chiroptera"],
                "infection_status": [1, 0, 1],
            }
        )

        engineer = FeatureEngineer()
        df_engineered = engineer.engineer_features(df)

        # Should add features
        assert "gc_content" in df_engineered.columns
        assert "au_content" in df_engineered.columns
        assert "cpg_frequency" in df_engineered.columns
        assert "body_mass_kg" in df_engineered.columns
        assert "lifespan_years" in df_engineered.columns

    def test_fit_transform(self):
        """Test fit_transform method."""
        df = pd.DataFrame(
            {
                "virus_species": ["BDBV", "EBOV", "SUDV", "MARV"],
                "host_species": ["Homo_sapiens", "Pan", "Rousettus", "Myotis"],
                "host_order": ["Primates", "Primates", "Chiroptera", "Chiroptera"],
                "body_mass_kg": [70, 50, 0.3, 0.02],
                "lifespan_years": [80, 50, 22, 8],
                "infection_status": [1, 1, 1, 0],
            }
        )

        engineer = FeatureEngineer()
        df_transformed = engineer.fit_transform(df)

        assert df_transformed is not None
        assert len(df_transformed) == 4
        assert "infection_status" in df_transformed.columns

    def test_transform(self):
        """Test transform method after fitting."""
        # First fit
        df_train = pd.DataFrame(
            {
                "virus_species": ["BDBV", "EBOV"],
                "host_species": ["Homo_sapiens", "Pan"],
                "host_order": ["Primates", "Primates"],
                "body_mass_kg": [70, 50],
                "lifespan_years": [80, 50],
                "infection_status": [1, 0],
            }
        )

        engineer = FeatureEngineer()
        engineer.fit_transform(df_train)

        # Now transform new data
        df_test = pd.DataFrame(
            {
                "virus_species": ["SUDV"],
                "host_species": ["Rousettus"],
                "host_order": ["Chiroptera"],
                "body_mass_kg": [0.3],
                "lifespan_years": [22],
            }
        )

        df_transformed = engineer.transform(df_test)

        assert df_transformed is not None
        assert len(df_transformed) == 1

    def test_get_feature_names(self):
        """Test getting feature names."""
        df = pd.DataFrame(
            {
                "virus_species": ["BDBV"],
                "host_species": ["Homo_sapiens"],
                "host_order": ["Primates"],
                "body_mass_kg": [70],
                "lifespan_years": [80],
                "infection_status": [1],
            }
        )

        engineer = FeatureEngineer()
        engineer.fit_transform(df)

        feature_names = engineer.get_feature_names()

        assert feature_names is not None
        assert len(feature_names) > 0

    def test_missing_values_handling(self):
        """Test handling of missing values."""
        df = pd.DataFrame(
            {
                "virus_species": ["BDBV", "EBOV"],
                "host_species": ["Homo_sapiens", None],
                "host_order": ["Primates", "Primates"],
                "body_mass_kg": [70, np.nan],
                "lifespan_years": [80, 50],
                "infection_status": [1, 0],
            }
        )

        engineer = FeatureEngineer()
        df_transformed = engineer.fit_transform(df)

        # Should handle missing values without error
        assert df_transformed is not None
        assert not df_transformed["body_mass_kg"].isnull().any()
