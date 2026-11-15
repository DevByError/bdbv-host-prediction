"""Codon usage metrics for virus-host adaptation."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CodonAnalyzer:
    """CAI, ENC, RSCU and virus-vs-host codon bias."""

    # standard genetic code, '*' = stop
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
        self.species_codon_usage = species_codon_usage or {}

    def calculate_cai(
        self, viral_sequence: str, host_codon_usage: Dict[str, float]
    ) -> float:
        """Usage-weighted mean rather than the geometric-mean CAI of Sharp &
        Li - close enough for a feature, not comparable to published CAI."""
        viral_codons = self._get_codon_counts(viral_sequence)

        if not viral_codons:
            return 0.0

        cai_values = []
        for codon, count in viral_codons.items():
            if codon in host_codon_usage:
                cai_values.append(count * host_codon_usage[codon])

        if not cai_values:
            return 0.0

        return sum(cai_values) / sum(viral_codons.values())

    def calculate_enc(self, sequence: str) -> float:
        """Mean per-amino-acid effective codon number. Returns 0.0 for short
        sequences."""
        codon_counts = self._get_codon_counts(sequence)

        if not codon_counts:
            return 0.0

        n_codons = sum(codon_counts.values())

        if n_codons < 20:  # too few to mean anything
            return 0.0

        aa_groups = {}
        for codon, count in codon_counts.items():
            if codon in self.CODON_TABLE:
                aa = self.CODON_TABLE[codon]
                if aa != "*":
                    if aa not in aa_groups:
                        aa_groups[aa] = []
                    aa_groups[aa].append(count)

        total_enc = 0
        for aa, counts in aa_groups.items():
            if len(counts) >= 2:  # only degenerate families contribute
                total_count = sum(counts)
                if total_count > 0:
                    # homozygosity-style F
                    f = sum((c / total_count) ** 2 for c in counts)
                    enc_aa = (len(counts) - 1) / (f - 1) if f > 1 else len(counts)
                    total_enc += enc_aa

        if aa_groups:
            return total_enc / len(aa_groups)
        else:
            return 0.0

    def calculate_rscu(self, sequence: str) -> Dict[str, float]:
        """RSCU per codon. 1.0 means no preference within the family.

        Note the denominator uses codons actually seen in this sequence, not
        the full degeneracy of each family."""
        codon_counts = self._get_codon_counts(sequence)

        if not codon_counts:
            return {}

        aa_codons = {}
        for codon, count in codon_counts.items():
            if codon in self.CODON_TABLE:
                aa = self.CODON_TABLE[codon]
                if aa != "*":
                    if aa not in aa_codons:
                        aa_codons[aa] = {}
                    aa_codons[aa][codon] = count

        rscu = {}
        for aa, codons in aa_codons.items():
            total_count = sum(codons.values())
            if total_count > 0:
                for codon, count in codons.items():
                    rscu[codon] = (count / total_count) / (1 / len(codons))

        return rscu

    def calculate_codon_bias(self, viral_sequence: str, host_sequence: str) -> Dict:
        """Correlation and Euclidean distance between the two codon frequency
        vectors. Empty dict if either sequence has no codons."""
        viral_codons = self._get_codon_counts(viral_sequence)
        host_codons = self._get_codon_counts(host_sequence)

        if not viral_codons or not host_codons:
            return {}

        total_viral = sum(viral_codons.values())
        total_host = sum(host_codons.values())

        viral_freq = {c: count / total_viral for c, count in viral_codons.items()}
        host_freq = {c: count / total_host for c, count in host_codons.items()}

        # correlation only over codons present in both
        common_codons = set(viral_freq.keys()) & set(host_freq.keys())

        if not common_codons:
            return {}

        viral_common = [viral_freq[c] for c in common_codons]
        host_common = [host_freq[c] for c in common_codons]

        correlation = np.corrcoef(viral_common, host_common)[0, 1]

        # distance walks viral codons only, missing host codons count as 0
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
        """Reading frame 0 only, no frame detection."""
        codon_counts = {}
        for i in range(0, len(sequence) - 2, 3):
            codon = sequence[i : i + 3].upper()
            if len(codon) == 3:
                codon_counts[codon] = codon_counts.get(codon, 0) + 1
        return codon_counts
