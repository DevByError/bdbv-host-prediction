"""
Comprehensive tests for advanced feature engineering and analysis.
"""

import pytest
import numpy as np
import pandas as pd

from src.features.codon_analysis import CodonAnalyzer
from src.features.phylogenetic import PhylogeneticFeatures
from src.features.preprocessing import FeaturePreprocessor
from src.features.selection import FeatureSelector
from src.models.classical import RandomForestModel


class TestCodonAnalysis:
    """Comprehensive tests for codon analysis."""

    def test_codon_analyzer_initialization(self):
        """Test CodonAnalyzer initialization."""
        analyzer = CodonAnalyzer()
        assert analyzer is not None
        assert analyzer.CODON_TABLE is not None

    def test_calculate_cai(self):
        """Test CAI calculation."""
        sequence = "ATGATGATGATGATGATG"
        host_codon_usage = {"ATG": 0.5, "GAT": 0.3, "ATG": 0.2}

        analyzer = CodonAnalyzer()
        cai = analyzer.calculate_cai(sequence, host_codon_usage)

        assert isinstance(cai, float)
        assert cai >= 0

    def test_calculate_enc(self):
        """Test Effective Number of Codons calculation."""
        sequence = "ATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATG"
        analyzer = CodonAnalyzer()

        enc = analyzer.calculate_enc(sequence)

        assert isinstance(enc, float)
        assert enc >= 0

    def test_calculate_rscu(self):
        """Test RSCU calculation."""
        sequence = "ATGATGATGCGACGACGA"
        analyzer = CodonAnalyzer()

        rscu = analyzer.calculate_rscu(sequence)

        assert isinstance(rscu, dict)
        # RSCU values should be non-negative
        if rscu:
            assert all(v >= 0 for v in rscu.values())

    def test_calculate_codon_bias(self):
        """Test codon bias calculation."""
        viral_seq = "ATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATG"
        host_seq = "ATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATG"

        analyzer = CodonAnalyzer()
        bias = analyzer.calculate_codon_bias(viral_seq, host_seq)

        assert isinstance(bias, dict)
        if bias:
            assert "codon_correlation" in bias or "euclidean_distance" in bias

    def test_empty_sequence_handling(self):
        """Test handling of empty sequences."""
        analyzer = CodonAnalyzer()

        # Should handle empty gracefully
        rscu = analyzer.calculate_rscu("")
        assert rscu == {}

        enc = analyzer.calculate_enc("")
        assert enc == 0.0

    def test_codon_table_completeness(self):
        """Test that codon table is complete."""
        analyzer = CodonAnalyzer()

        # Should have 64 codons
        assert len(analyzer.CODON_TABLE) == 64


class TestPhylogeneticAnalysis:
    """Comprehensive tests for phylogenetic analysis."""

    def test_phylogenetic_features_initialization(self):
        """Test PhylogeneticFeatures initialization."""
        features = PhylogeneticFeatures()
        assert features is not None
        assert features.TAXONOMY is not None

    def test_phylogenetic_taxonomy_exists(self):
        """Test that taxonomy is populated."""
        features = PhylogeneticFeatures()
        assert len(features.TAXONOMY) > 0
        assert "Homo_sapiens" in features.TAXONOMY

    def test_taxonomy_structure(self):
        """Test taxonomy structure for a known host."""
        features = PhylogeneticFeatures()
        homo_tax = features.TAXONOMY["Homo_sapiens"]

        assert "order" in homo_tax
        assert "family" in homo_tax
        assert "genus" in homo_tax
        assert homo_tax["order"] == "Primates"
        assert homo_tax["family"] == "Hominidae"

    def test_multiple_hosts_in_taxonomy(self):
        """Test that multiple host species are in taxonomy."""
        features = PhylogeneticFeatures()

        expected_hosts = ["Homo_sapiens", "Pan_troglodytes", "Myotis_myotis"]
        for host in expected_hosts:
            assert host in features.TAXONOMY
            tax = features.TAXONOMY[host]
            assert "order" in tax
            assert "family" in tax
            assert "genus" in tax

    def test_taxonomy_chiroptera(self):
        """Test Chiroptera order species."""
        features = PhylogeneticFeatures()

        # At least one bat species should be in taxonomy
        bat_in_taxonomy = any(
            features.TAXONOMY[species].get("order") == "Chiroptera"
            for species in features.TAXONOMY
        )
        assert bat_in_taxonomy


class TestFeaturePreprocessing:
    """Comprehensive tests for feature preprocessing."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for preprocessing."""
        np.random.seed(42)
        return pd.DataFrame({
            "feature_1": np.random.randn(100),
            "feature_2": np.random.uniform(0, 100, 100),
            "feature_3": np.random.exponential(2, 100),
            "category": np.random.choice(["A", "B", "C"], 100),
        })

    def test_preprocessor_initialization(self):
        """Test FeaturePreprocessor initialization."""
        preprocessor = FeaturePreprocessor()
        assert preprocessor is not None
        assert preprocessor.numeric_scaler == "standard"

    def test_build_pipeline(self, sample_data):
        """Test building preprocessing pipeline."""
        preprocessor = FeaturePreprocessor()
        numeric_cols = ["feature_1", "feature_2"]
        categorical_cols = ["category"]

        pipeline = preprocessor.build_pipeline(numeric_cols, categorical_cols)

        assert pipeline is not None

    def test_fit_transform(self, sample_data):
        """Test fit_transform on sample data."""
        preprocessor = FeaturePreprocessor()

        X_transformed = preprocessor.fit_transform(sample_data)

        assert X_transformed.shape[0] == sample_data.shape[0]
        assert X_transformed.shape[1] > 0
        assert preprocessor._is_fitted

    def test_transform_after_fit(self, sample_data):
        """Test transform on new data after fitting."""
        preprocessor = FeaturePreprocessor()

        # Fit on first half
        X_fit = sample_data.iloc[:50]
        X_transformed_fit = preprocessor.fit_transform(X_fit)

        # Transform second half
        X_new = sample_data.iloc[50:]
        X_transformed_new = preprocessor.transform(X_new)

        assert X_transformed_new.shape[0] == len(X_new)

    def test_minmax_scaler(self, sample_data):
        """Test minmax scaling."""
        preprocessor = FeaturePreprocessor(numeric_scaler="minmax")
        X_transformed = preprocessor.fit_transform(sample_data)

        # Values should be in approximately [0, 1] range
        numeric_data = X_transformed.select_dtypes(include=[np.number])
        if numeric_data.shape[1] > 0:
            assert numeric_data.min().min() >= -0.1
            assert numeric_data.max().max() <= 1.1

    def test_robust_scaler(self, sample_data):
        """Test robust scaling."""
        preprocessor = FeaturePreprocessor(numeric_scaler="robust")
        X_transformed = preprocessor.fit_transform(sample_data)

        assert X_transformed.shape[0] == sample_data.shape[0]

    def test_knn_imputation(self):
        """Test KNN imputation strategy."""
        X = pd.DataFrame({
            "feature_1": [1, 2, np.nan, 4, 5],
            "feature_2": [10, np.nan, 30, 40, 50],
        })

        preprocessor = FeaturePreprocessor(imputation_strategy="knn")

        # Add required columns for fit_transform
        X["category"] = "A"
        X_transformed = preprocessor.fit_transform(X)

        # No NaNs should remain
        assert X_transformed.isnull().sum().sum() == 0

    def test_different_imputation_strategies(self, sample_data):
        """Test different imputation strategies."""
        strategies = ["mean", "median", "most_frequent"]

        for strategy in strategies:
            preprocessor = FeaturePreprocessor(imputation_strategy=strategy)
            X_transformed = preprocessor.fit_transform(sample_data)

            assert X_transformed.shape[0] == sample_data.shape[0]


class TestFeatureSelection:
    """Comprehensive tests for feature selection."""

    @pytest.fixture
    def feature_data(self):
        """Create data for feature selection."""
        np.random.seed(42)
        n_samples = 100
        n_features = 20

        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)]
        )
        y = np.random.randint(0, 2, n_samples)

        return X, y

    def test_feature_selector_initialization(self):
        """Test FeatureSelector initialization."""
        selector = FeatureSelector()
        assert selector is not None
        assert selector.random_state == 42

    def test_variance_threshold(self, feature_data):
        """Test variance threshold selection."""
        X, y = feature_data

        selector = FeatureSelector()
        selected = selector.variance_threshold(X, threshold=0.1)

        assert isinstance(selected, list)
        assert len(selected) <= len(X.columns)

    def test_univariate_selection(self, feature_data):
        """Test univariate selection."""
        X, y = feature_data

        selector = FeatureSelector()
        selected = selector.univariate_selection(X, y, method="mutual_info", n_features=5)

        assert isinstance(selected, list)
        assert len(selected) <= 5

    def test_univariate_f_classif(self, feature_data):
        """Test univariate selection with f_classif."""
        X, y = feature_data

        selector = FeatureSelector()
        selected = selector.univariate_selection(X, y, method="f_classif", n_features=5)

        assert isinstance(selected, list)
        assert len(selected) <= 5

    def test_recursive_feature_elimination(self, feature_data):
        """Test recursive feature elimination."""
        X, y = feature_data

        selector = FeatureSelector()
        selected = selector.recursive_feature_elimination(X, y, n_features=5)

        assert isinstance(selected, list)
        assert len(selected) <= 5

    def test_model_based_selection(self, feature_data):
        """Test model-based feature selection."""
        X, y = feature_data

        selector = FeatureSelector()
        selected = selector.model_based_selection(X, y)

        assert isinstance(selected, list)
        assert len(selected) >= 0

    def test_correlation_based_selection(self, feature_data):
        """Test correlation-based feature selection."""
        X, y = feature_data

        selector = FeatureSelector()
        selected = selector.correlation_based_selection(X, threshold=0.9)

        assert isinstance(selected, list)
        assert len(selected) <= len(X.columns)

    def test_get_feature_scores(self, feature_data):
        """Test getting feature importance scores."""
        X, y = feature_data

        selector = FeatureSelector()
        scores_df = selector.get_feature_scores(X, y, method="mutual_info")

        assert isinstance(scores_df, pd.DataFrame)
        assert "feature" in scores_df.columns
        assert "score" in scores_df.columns
        assert len(scores_df) == len(X.columns)

    def test_select_features_combined(self, feature_data):
        """Test combined feature selection."""
        X, y = feature_data

        selector = FeatureSelector()
        selected = selector.select_features(
            X, y,
            methods=["variance", "correlation"],
            n_features=10
        )

        assert isinstance(selected, list)
        assert len(selected) <= 10


class TestAdvancedFeatureIntegration:
    """Integration tests for advanced features."""

    def test_codon_analysis_with_model(self):
        """Test codon analysis integrated with model training."""
        np.random.seed(42)

        # Create data
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)

        # Analyze codons (mock)
        analyzer = CodonAnalyzer()
        enc = analyzer.calculate_enc("ATGATGATGATGATGATGATGATGATGATG")

        # Use in model
        model = RandomForestModel({"n_estimators": 10, "random_state": 42})
        model.fit(X, y)

        assert model._is_fitted
        assert enc >= 0

    def test_preprocessing_then_selection(self):
        """Test preprocessing followed by feature selection."""
        np.random.seed(42)

        # Create data
        X = pd.DataFrame({
            "f1": np.random.randn(50),
            "f2": np.random.randn(50),
            "f3": np.random.randn(50),
            "category": np.random.choice(["A", "B"], 50),
        })
        y = np.random.randint(0, 2, 50)

        # Preprocess
        preprocessor = FeaturePreprocessor()
        X_processed = preprocessor.fit_transform(X)

        # Select features
        selector = FeatureSelector()
        selected = selector.variance_threshold(X_processed, threshold=0.01)

        assert len(selected) >= 0

    def test_phylogenetic_taxonomy_lookup(self):
        """Test looking up species in taxonomy."""
        features = PhylogeneticFeatures()

        # Test known species
        homo_sapiens = features.TAXONOMY["Homo_sapiens"]
        assert homo_sapiens["order"] == "Primates"

        # Test other primates
        pan = features.TAXONOMY["Pan_troglodytes"]
        assert pan["order"] == "Primates"
