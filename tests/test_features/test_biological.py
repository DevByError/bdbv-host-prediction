"""
Tests for biological feature calculation.
"""

import pytest
import numpy as np
from src.features.biological import BiologicalFeatures


class TestBiologicalFeatures:
    """Test BiologicalFeatures class."""

    def test_gc_content(self):
        """Test GC content calculation."""
        sequence = "ATGCGATCG"
        gc = BiologicalFeatures.calculate_gc_content(sequence)

        # Count: G at positions 3,5,9 (3 Gs), C at positions 4,8 (2 Cs)
        expected = 5 / 9 * 100  # 5 G/C out of 9 bases
        assert abs(gc - expected) < 0.01

    def test_gc_content_empty(self):
        """Test GC content with empty sequence."""
        gc = BiologicalFeatures.calculate_gc_content("")
        assert gc == 0.0

    def test_cpg_frequency(self):
        """Test CpG frequency calculation."""
        sequence = "CGCGATCG"
        cpg = BiologicalFeatures.calculate_cpg_frequency(sequence)

        # Count "CG" occurrences
        expected = 3 / 7  # 3 CpG out of 7 dinucleotides
        assert abs(cpg - expected) < 0.01

    def test_cpg_frequency_short(self):
        """Test CpG frequency with short sequence."""
        cpg = BiologicalFeatures.calculate_cpg_frequency("CG")
        assert cpg == 1.0

    def test_codon_usage(self):
        """Test codon usage calculation."""
        sequence = "ATGCGATCG"
        host_table = {"ATG": 0.5, "CGA": 0.3, "TCG": 0.2}

        result = BiologicalFeatures.calculate_codon_usage(sequence, host_table)

        assert "rscu_mean" in result
        assert "enc_value" in result
        assert "cai_value" in result
        assert result["cai_value"] >= 0

    def test_protein_features(self):
        """Test protein feature calculation."""
        sequence = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"

        features = BiologicalFeatures.calculate_protein_features(sequence)

        assert "hydrophobicity" in features
        assert "aliphatic_index" in features
        assert "instability_index" in features
        assert "length" in features
        assert features["length"] == len(sequence)

    def test_phylogenetic_distance(self):
        """Test phylogenetic distance calculation."""
        seq1 = "ATGCGATCG"
        seq2 = "ATGCGATCA"

        distance = BiologicalFeatures.calculate_phylogenetic_distance(seq1, seq2)

        expected = 1 / 9  # 1 difference out of 9
        assert abs(distance - expected) < 0.01

    def test_phylogenetic_distance_different_lengths(self):
        """Test phylogenetic distance with different lengths."""
        seq1 = "ATGCGATCG"
        seq2 = "ATGCGATC"

        distance = BiologicalFeatures.calculate_phylogenetic_distance(seq1, seq2)

        # Should return 1.0 for different lengths
        assert distance == 1.0

    def test_codon_usage_empty_sequence(self):
        """Test codon usage with empty sequence."""
        result = BiologicalFeatures.calculate_codon_usage("", {})

        assert result["rscu_mean"] == 0
        assert result["enc_value"] == 0
        assert result["cai_value"] == 0
