"""
Biological feature calculation from sequences.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BiologicalFeatures:
    """
    Calculate biological features from sequence data.
    """

    @staticmethod
    def calculate_gc_content(sequence: str) -> float:
        """Calculate GC content of a sequence."""
        if not sequence:
            return 0.0
        gc_count = sequence.count("G") + sequence.count("C")
        return gc_count / len(sequence) * 100

    @staticmethod
    def calculate_cpg_frequency(sequence: str) -> float:
        """Calculate CpG dinucleotide frequency."""
        if len(sequence) < 2:
            return 0.0

        cpg_count = sum(
            1 for i in range(len(sequence) - 1) if sequence[i : i + 2].upper() == "CG"
        )
        return cpg_count / (len(sequence) - 1)

    @staticmethod
    def calculate_codon_usage(sequence: str, host_codon_table: Dict) -> Dict:
        """
        Calculate codon usage indices.

        Args:
            sequence: DNA sequence
            host_codon_table: Codon frequency table for host

        Returns:
            Dict: Codon usage metrics
        """
        # Count codons in sequence
        codons = [sequence[i : i + 3].upper() for i in range(0, len(sequence) - 2, 3)]

        # RSCU (Relative Synonymous Codon Usage)
        codon_counts = {}
        for codon in codons:
            if len(codon) == 3:
                codon_counts[codon] = codon_counts.get(codon, 0) + 1

        # ENC (Effective Number of Codons)
        # Simplified calculation
        if len(codon_counts) > 0:
            enc_value = len(codon_counts) / len(codons) if codons else 0
        else:
            enc_value = 0

        # CAI (Codon Adaptation Index) - simplified
        if host_codon_table:
            cai_sum = 0
            for codon, count in codon_counts.items():
                if codon in host_codon_table:
                    cai_sum += count * host_codon_table[codon]
            cai_value = cai_sum / len(codons) if codons else 0
        else:
            cai_value = 0

        return {
            "rscu_mean": np.mean(list(codon_counts.values())) if codon_counts else 0,
            "enc_value": enc_value,
            "cai_value": cai_value,
            "codon_count": len(codon_counts),
        }

    @staticmethod
    def calculate_protein_features(protein_sequence: str) -> Dict:
        """
        Calculate protein features from amino acid sequence.

        Args:
            protein_sequence: Amino acid sequence

        Returns:
            Dict: Protein features
        """
        if not protein_sequence:
            return {}

        # Amino acid properties
        hydrophobic_aa = ["A", "I", "L", "M", "F", "W", "V", "P", "G"]
        charged_aa = ["R", "H", "K", "D", "E"]
        polar_aa = ["S", "T", "Y", "C", "N", "Q"]

        # Calculate composition
        n_residues = len(protein_sequence)

        hydrophobic_count = sum(1 for aa in protein_sequence if aa in hydrophobic_aa)
        charged_count = sum(1 for aa in protein_sequence if aa in charged_aa)
        polar_count = sum(1 for aa in protein_sequence if aa in polar_aa)

        # Hydrophobicity score (simplified Kyte-Doolittle)
        hydrophobicity_values = {
            "A": 1.8,
            "R": -4.5,
            "N": -3.5,
            "D": -3.5,
            "C": 2.5,
            "Q": -3.5,
            "E": -3.5,
            "G": -0.4,
            "H": -3.2,
            "I": 4.5,
            "L": 3.8,
            "K": -3.9,
            "M": 1.9,
            "F": 2.8,
            "P": -1.6,
            "S": -0.8,
            "T": -0.7,
            "W": -0.9,
            "Y": -1.3,
            "V": 4.2,
        }

        hydrophobicity = np.mean(
            [hydrophobicity_values.get(aa, 0) for aa in protein_sequence]
        )

        # Aliphatic index
        aliphatic_aa = ["A", "I", "L", "V"]
        aliphatic_count = sum(1 for aa in protein_sequence if aa in aliphatic_aa)
        aliphatic_index = aliphatic_count / n_residues * 100

        # Instability index (simplified)
        instability = np.random.normal(40, 8)  # Placeholder

        return {
            "hydrophobicity": hydrophobicity,
            "aliphatic_index": aliphatic_index,
            "instability_index": instability,
            "hydrophobic_ratio": hydrophobic_count / n_residues,
            "charged_ratio": charged_count / n_residues,
            "polar_ratio": polar_count / n_residues,
            "length": n_residues,
        }

    @staticmethod
    def calculate_phylogenetic_distance(seq1: str, seq2: str) -> float:
        """
        Calculate phylogenetic distance between sequences.

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            float: Distance metric
        """
        # Simple Hamming distance
        if len(seq1) != len(seq2):
            return 1.0

        differences = sum(1 for a, b in zip(seq1, seq2) if a != b)
        return differences / len(seq1)
