"""
Protein sequence analysis for glycoprotein features.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ProteinAnalyzer:
    """
    Analyze protein sequences for structural and functional features.
    """

    # Amino acid properties
    AA_PROPERTIES = {
        "A": {
            "hydrophobic": True,
            "charge": 0,
            "polar": False,
            "mass": 89.09,
            "pI": 6.01,
        },
        "R": {
            "hydrophobic": False,
            "charge": 1,
            "polar": True,
            "mass": 174.20,
            "pI": 10.76,
        },
        "N": {
            "hydrophobic": False,
            "charge": 0,
            "polar": True,
            "mass": 132.12,
            "pI": 5.41,
        },
        "D": {
            "hydrophobic": False,
            "charge": -1,
            "polar": True,
            "mass": 133.10,
            "pI": 2.77,
        },
        "C": {
            "hydrophobic": True,
            "charge": 0,
            "polar": False,
            "mass": 121.15,
            "pI": 5.05,
        },
        "Q": {
            "hydrophobic": False,
            "charge": 0,
            "polar": True,
            "mass": 146.15,
            "pI": 5.65,
        },
        "E": {
            "hydrophobic": False,
            "charge": -1,
            "polar": True,
            "mass": 147.13,
            "pI": 3.22,
        },
        "G": {
            "hydrophobic": False,
            "charge": 0,
            "polar": False,
            "mass": 75.07,
            "pI": 6.06,
        },
        "H": {
            "hydrophobic": False,
            "charge": 0,
            "polar": True,
            "mass": 155.16,
            "pI": 7.59,
        },
        "I": {
            "hydrophobic": True,
            "charge": 0,
            "polar": False,
            "mass": 131.17,
            "pI": 6.02,
        },
        "L": {
            "hydrophobic": True,
            "charge": 0,
            "polar": False,
            "mass": 131.17,
            "pI": 6.02,
        },
        "K": {
            "hydrophobic": False,
            "charge": 1,
            "polar": True,
            "mass": 146.19,
            "pI": 9.74,
        },
        "M": {
            "hydrophobic": True,
            "charge": 0,
            "polar": False,
            "mass": 149.21,
            "pI": 5.74,
        },
        "F": {
            "hydrophobic": True,
            "charge": 0,
            "polar": False,
            "mass": 165.19,
            "pI": 5.48,
        },
        "P": {
            "hydrophobic": True,
            "charge": 0,
            "polar": False,
            "mass": 115.13,
            "pI": 6.30,
        },
        "S": {
            "hydrophobic": False,
            "charge": 0,
            "polar": True,
            "mass": 105.09,
            "pI": 5.68,
        },
        "T": {
            "hydrophobic": False,
            "charge": 0,
            "polar": True,
            "mass": 119.12,
            "pI": 5.60,
        },
        "W": {
            "hydrophobic": True,
            "charge": 0,
            "polar": False,
            "mass": 204.23,
            "pI": 5.89,
        },
        "Y": {
            "hydrophobic": False,
            "charge": 0,
            "polar": True,
            "mass": 181.19,
            "pI": 5.66,
        },
        "V": {
            "hydrophobic": True,
            "charge": 0,
            "polar": False,
            "mass": 117.15,
            "pI": 5.97,
        },
    }

    @staticmethod
    def analyze_sequence(sequence: str) -> Dict:
        """
        Comprehensive analysis of protein sequence.

        Args:
            sequence: Amino acid sequence

        Returns:
            Dict: Protein features
        """
        if not sequence:
            return {}

        n_residues = len(sequence)

        # Composition
        composition = {}
        for aa in sequence:
            composition[aa] = composition.get(aa, 0) + 1

        # Calculate properties
        hydrophobic_count = 0
        charged_count = 0
        polar_count = 0
        total_mass = 0
        total_pI = 0

        for aa, count in composition.items():
            props = ProteinAnalyzer.AA_PROPERTIES.get(aa, {})
            hydrophobic_count += count if props.get("hydrophobic", False) else 0
            charged_count += count if props.get("charge", 0) != 0 else 0
            polar_count += count if props.get("polar", False) else 0
            total_mass += count * props.get("mass", 100)
            total_pI += count * props.get("pI", 6)

        # Hydrophobicity (Kyte-Doolittle)
        hydrophobicity = ProteinAnalyzer._calculate_hydrophobicity(sequence)

        # Aliphatic index
        aliphatic_index = ProteinAnalyzer._calculate_aliphatic_index(sequence)

        # Instability index
        instability_index = ProteinAnalyzer._calculate_instability_index(sequence)

        # Molecular weight
        molecular_weight = total_mass / n_residues

        return {
            "length": n_residues,
            "composition": composition,
            "hydrophobicity": hydrophobicity,
            "aliphatic_index": aliphatic_index,
            "instability_index": instability_index,
            "molecular_weight": molecular_weight,
            "hydrophobic_ratio": hydrophobic_count / n_residues,
            "charged_ratio": charged_count / n_residues,
            "polar_ratio": polar_count / n_residues,
            "pI": total_pI / n_residues if n_residues > 0 else 0,
            "total_charge": sum(
                ProteinAnalyzer.AA_PROPERTIES.get(aa, {}).get("charge", 0)
                for aa in sequence
            ),
        }

    @staticmethod
    def _calculate_hydrophobicity(sequence: str) -> float:
        """Calculate average hydrophobicity (Kyte-Doolittle scale)."""
        hydrophobic_values = {
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

        values = [hydrophobic_values.get(aa, 0) for aa in sequence]
        return np.mean(values) if values else 0

    @staticmethod
    def _calculate_aliphatic_index(sequence: str) -> float:
        """Calculate aliphatic index."""
        aliphatic_values = {"A": 1, "I": 1, "L": 1, "V": 1}

        aliphatic_count = sum(1 for aa in sequence if aa in aliphatic_values)
        return (aliphatic_count / len(sequence)) * 100 if sequence else 0

    @staticmethod
    def _calculate_instability_index(sequence: str) -> float:
        """Calculate instability index (simplified)."""
        # Simplified instability calculation
        if not sequence:
            return 0

        # Dipeptide instability values (simplified)
        instability = {
            "DI": 1.0,
            "DV": 1.0,
            "CD": 0.0,
            "GD": 1.0,
            "DP": 1.0,
            "CP": 0.0,
            "CC": 0.0,
            "CG": 0.0,
        }

        score = 0
        for i in range(len(sequence) - 1):
            dipeptide = sequence[i : i + 2]
            score += instability.get(dipeptide, 0)

        return score / len(sequence) * 10

    @staticmethod
    def find_motifs(sequence: str, motifs: List[str]) -> Dict[str, int]:
        """
        Find motif occurrences in sequence.

        Args:
            sequence: Protein sequence
            motifs: List of motifs to find

        Returns:
            Dict[str, int]: Motif counts
        """
        results = {}
        for motif in motifs:
            count = sequence.upper().count(motif.upper())
            results[motif] = count
        return results

    @staticmethod
    def calculate_domain_scores(sequence: str) -> Dict[str, float]:
        """
        Calculate domain-related scores.

        Args:
            sequence: Protein sequence

        Returns:
            Dict[str, float]: Domain scores
        """
        # Simple domain score calculation
        # In practice, this would use HMMER or similar

        scores = {
            "signal_peptide": np.random.uniform(0, 0.8),
            "transmembrane": np.random.uniform(0, 0.6),
            "coiled_coil": np.random.uniform(0, 0.5),
            "disorder": np.random.uniform(0, 0.7),
            "nuclear_localization": np.random.uniform(0, 0.4),
        }

        # Adjust based on sequence properties
        if sequence:
            hydrophobicity = ProteinAnalyzer._calculate_hydrophobicity(sequence)
            if hydrophobicity > 0.3:
                scores["transmembrane"] = min(1.0, scores["transmembrane"] + 0.3)
            if "KR" in sequence:
                scores["nuclear_localization"] = min(
                    1.0, scores["nuclear_localization"] + 0.2
                )

        return scores
