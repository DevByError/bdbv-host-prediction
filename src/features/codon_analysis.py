"""
Codon usage analysis for viral-host adaptation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CodonAnalyzer:
    """
    Analyze codon usage patterns between virus and host.
    """

    # Standard genetic code
    CODON_TABLE = {
        "TTT": "F",
        "TTC": "F",
        "TTA": "L",
        "TTG": "L",
        "TCT": "S",
        "TCC": "S",
        "TCA": "S",
        "TCG": "S",
        "TAT": "Y",
        "TAC": "Y",
        "TAA": "*",
        "TAG": "*",
        "TGT": "C",
        "TGC": "C",
        "TGA": "*",
        "TGG": "W",
        "CTT": "L",
        "CTC": "L",
        "CTA": "L",
        "CTG": "L",
        "CCT": "P",
        "CCC": "P",
        "CCA": "P",
        "CCG": "P",
        "CAT": "H",
        "CAC": "H",
        "CAA": "Q",
        "CAG": "Q",
        "CGT": "R",
        "CGC": "R",
        "CGA": "R",
        "CGG": "R",
        "ATT": "I",
        "ATC": "I",
        "ATA": "I",
        "ATG": "M",
        "ACT": "T",
        "ACC": "T",
        "ACA": "T",
        "ACG": "T",
        "AAT": "N",
        "AAC": "N",
        "AAA": "K",
        "AAG": "K",
        "AGT": "S",
        "AGC": "S",
        "AGA": "R",
        "AGG": "R",
        "GTT": "V",
        "GTC": "V",
        "GTA": "V",
        "GTG": "V",
        "GCT": "A",
        "GCC": "A",
        "GCA": "A",
        "GCG": "A",
        "GAT": "D",
        "GAC": "D",
        "GAA": "E",
        "GAG": "E",
        "GGT": "G",
        "GGC": "G",
        "GGA": "G",
        "GGG": "G",
    }

    def __init__(self, species_codon_usage: Optional[Dict] = None):
        """
        Initialize codon analyzer.

        Args:
            species_codon_usage: Codon usage frequencies for host species
        """
        self.species_codon_usage = species_codon_usage or {}

    def calculate_cai(
        self, viral_sequence: str, host_codon_usage: Dict[str, float]
    ) -> float:
        """
        Calculate Codon Adaptation Index.

        Args:
            viral_sequence: Viral DNA sequence
            host_codon_usage: Host codon usage frequencies

        Returns:
            float: CAI value
        """
        # Get codon counts from viral sequence
        viral_codons = self._get_codon_counts(viral_sequence)

        if not viral_codons:
            return 0.0

        # Calculate CAI
        cai_values = []
        for codon, count in viral_codons.items():
            if codon in host_codon_usage:
                cai_values.append(count * host_codon_usage[codon])

        if not cai_values:
            return 0.0

        return sum(cai_values) / sum(viral_codons.values())

    def calculate_enc(self, sequence: str) -> float:
        """
        Calculate Effective Number of Codons.

        Args:
            sequence: DNA sequence

        Returns:
            float: ENC value
        """
        codon_counts = self._get_codon_counts(sequence)

        if not codon_counts:
            return 0.0

        # Calculate ENC
        n_codons = sum(codon_counts.values())

        if n_codons < 20:  # Too few codons for reliable ENC
            return 0.0

        # Group by amino acid
        aa_groups = {}
        for codon, count in codon_counts.items():
            if codon in self.CODON_TABLE:
                aa = self.CODON_TABLE[codon]
                if aa != "*":  # Skip stop codons
                    if aa not in aa_groups:
                        aa_groups[aa] = []
                    aa_groups[aa].append(count)

        # Calculate ENC
        total_enc = 0
        for aa, counts in aa_groups.items():
            if len(counts) >= 2:  # Amino acid has degenerate codons
                total_count = sum(counts)
                if total_count > 0:
                    # F-Statistic
                    f = sum((c / total_count) ** 2 for c in counts)
                    enc_aa = (len(counts) - 1) / (f - 1) if f > 1 else len(counts)
                    total_enc += enc_aa

        if aa_groups:
            return total_enc / len(aa_groups)
        else:
            return 0.0

    def calculate_rscu(self, sequence: str) -> Dict[str, float]:
        """
        Calculate Relative Synonymous Codon Usage.

        Args:
            sequence: DNA sequence

        Returns:
            Dict[str, float]: RSCU values per codon
        """
        codon_counts = self._get_codon_counts(sequence)

        if not codon_counts:
            return {}

        # Group by amino acid
        aa_codons = {}
        for codon, count in codon_counts.items():
            if codon in self.CODON_TABLE:
                aa = self.CODON_TABLE[codon]
                if aa != "*":
                    if aa not in aa_codons:
                        aa_codons[aa] = {}
                    aa_codons[aa][codon] = count

        # Calculate RSCU
        rscu = {}
        for aa, codons in aa_codons.items():
            total_count = sum(codons.values())
            if total_count > 0:
                for codon, count in codons.items():
                    rscu[codon] = (count / total_count) / (1 / len(codons))

        return rscu

    def calculate_codon_bias(self, viral_sequence: str, host_sequence: str) -> Dict:
        """
        Compare codon usage between virus and host.

        Args:
            viral_sequence: Viral DNA sequence
            host_sequence: Host DNA sequence

        Returns:
            Dict: Codon bias metrics
        """
        # Get codon usage for both
        viral_codons = self._get_codon_counts(viral_sequence)
        host_codons = self._get_codon_counts(host_sequence)

        if not viral_codons or not host_codons:
            return {}

        # Calculate codon adaptation
        total_viral = sum(viral_codons.values())
        total_host = sum(host_codons.values())

        # Convert to frequencies
        viral_freq = {c: count / total_viral for c, count in viral_codons.items()}
        host_freq = {c: count / total_host for c, count in host_codons.items()}

        # Calculate similarity (correlation between codon usage)
        common_codons = set(viral_freq.keys()) & set(host_freq.keys())

        if not common_codons:
            return {}

        viral_common = [viral_freq[c] for c in common_codons]
        host_common = [host_freq[c] for c in common_codons]

        # Pearson correlation
        correlation = np.corrcoef(viral_common, host_common)[0, 1]

        # Euclidean distance
        distance = np.sqrt(
            sum((viral_freq[c] - host_freq.get(c, 0)) ** 2 for c in viral_freq.keys())
        )

        return {
            "codon_correlation": correlation,
            "euclidean_distance": distance,
            "common_codons": len(common_codons),
            "viral_codons": len(viral_codons),
            "host_codons": len(host_codons),
        }

    def _get_codon_counts(self, sequence: str) -> Dict[str, int]:
        """Get codon counts from sequence."""
        codon_counts = {}
        for i in range(0, len(sequence) - 2, 3):
            codon = sequence[i : i + 3].upper()
            if len(codon) == 3:
                codon_counts[codon] = codon_counts.get(codon, 0) + 1
        return codon_counts
