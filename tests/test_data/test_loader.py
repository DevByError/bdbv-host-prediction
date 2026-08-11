"""
Tests for data loader.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.loader import DataLoader
from src.data.generator import DataGenerator


class TestDataLoader:
    """Test DataLoader class."""

    def test_load_data(self, tmp_path):
        """Test loading data."""
        # Create test data
        df = pd.DataFrame(
            {
                "virus_species": ["BDBV", "EBOV"],
                "host_species": ["Homo_sapiens", "Pan_troglodytes"],
                "infection_status": [1, 0],
            }
        )

        test_path = tmp_path / "test_data.csv"
        df.to_csv(test_path, index=False)

        # Load data
        loader = DataLoader(test_path)
        loaded = loader.load_data()

        assert len(loaded) == 2
        assert "virus_species" in loaded.columns

    def test_generate_test_data(self):
        """Test synthetic data generation."""
        generator = DataGenerator()
        df = generator.generate_dataset(n_samples=10)

        assert len(df) == 10
        assert "virus_species" in df.columns
        assert "infection_status" in df.columns
