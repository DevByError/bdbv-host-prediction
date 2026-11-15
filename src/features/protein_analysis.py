"""Protein sequence features, mostly for the glycoprotein."""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# dipeptide instability weights, Guruprasad et al. 1990, Protein Eng 4(2):155-161.
# all 400 entries, same table ProtParam and Biopython use.
DIWV = {
    "AA": 1.0, "AC": 44.94, "AD": -7.49, "AE": 1.0,
    "AF": 1.0, "AG": 1.0, "AH": -7.49, "AI": 1.0,
    "AK": 1.0, "AL": 1.0, "AM": 1.0, "AN": 1.0,
    "AP": 20.26, "AQ": 1.0, "AR": 1.0, "AS": 1.0,
    "AT": 1.0, "AV": 1.0, "AW": 1.0, "AY": 1.0,
    "CA": 1.0, "CC": 1.0, "CD": 20.26, "CE": 1.0,
    "CF": 1.0, "CG": 1.0, "CH": 33.6, "CI": 1.0,
    "CK": 1.0, "CL": 20.26, "CM": 33.6, "CN": 1.0,
    "CP": 20.26, "CQ": -6.54, "CR": 1.0, "CS": 1.0,
    "CT": 33.6, "CV": -6.54, "CW": 24.68, "CY": 1.0,
    "DA": 1.0, "DC": 1.0, "DD": 1.0, "DE": 1.0,
    "DF": -6.54, "DG": 1.0, "DH": 1.0, "DI": 1.0,
    "DK": -7.49, "DL": 1.0, "DM": 1.0, "DN": 1.0,
    "DP": 1.0, "DQ": 1.0, "DR": -6.54, "DS": 20.26,
    "DT": -14.03, "DV": 1.0, "DW": 1.0, "DY": 1.0,
    "EA": 1.0, "EC": 44.94, "ED": 20.26, "EE": 33.6,
    "EF": 1.0, "EG": 1.0, "EH": -6.54, "EI": 20.26,
    "EK": 1.0, "EL": 1.0, "EM": 1.0, "EN": 1.0,
    "EP": 20.26, "EQ": 20.26, "ER": 1.0, "ES": 20.26,
    "ET": 1.0, "EV": 1.0, "EW": -14.03, "EY": 1.0,
    "FA": 1.0, "FC": 1.0, "FD": 13.34, "FE": 1.0,
    "FF": 1.0, "FG": 1.0, "FH": 1.0, "FI": 1.0,
    "FK": -14.03, "FL": 1.0, "FM": 1.0, "FN": 1.0,
    "FP": 20.26, "FQ": 1.0, "FR": 1.0, "FS": 1.0,
    "FT": 1.0, "FV": 1.0, "FW": 1.0, "FY": 33.601,
    "GA": -7.49, "GC": 1.0, "GD": 1.0, "GE": -6.54,
    "GF": 1.0, "GG": 13.34, "GH": 1.0, "GI": -7.49,
    "GK": -7.49, "GL": 1.0, "GM": 1.0, "GN": -7.49,
    "GP": 1.0, "GQ": 1.0, "GR": 1.0, "GS": 1.0,
    "GT": -7.49, "GV": 1.0, "GW": 13.34, "GY": -7.49,
    "HA": 1.0, "HC": 1.0, "HD": 1.0, "HE": 1.0,
    "HF": -9.37, "HG": -9.37, "HH": 1.0, "HI": 44.94,
    "HK": 24.68, "HL": 1.0, "HM": 1.0, "HN": 24.68,
    "HP": -1.88, "HQ": 1.0, "HR": 1.0, "HS": 1.0,
    "HT": -6.54, "HV": 1.0, "HW": -1.88, "HY": 44.94,
    "IA": 1.0, "IC": 1.0, "ID": 1.0, "IE": 44.94,
    "IF": 1.0, "IG": 1.0, "IH": 13.34, "II": 1.0,
    "IK": -7.49, "IL": 20.26, "IM": 1.0, "IN": 1.0,
    "IP": -1.88, "IQ": 1.0, "IR": 1.0, "IS": 1.0,
    "IT": 1.0, "IV": -7.49, "IW": 1.0, "IY": 1.0,
    "KA": 1.0, "KC": 1.0, "KD": 1.0, "KE": 1.0,
    "KF": 1.0, "KG": -7.49, "KH": 1.0, "KI": -7.49,
    "KK": 1.0, "KL": -7.49, "KM": 33.6, "KN": 1.0,
    "KP": -6.54, "KQ": 24.64, "KR": 33.6, "KS": 1.0,
    "KT": 1.0, "KV": -7.49, "KW": 1.0, "KY": 1.0,
    "LA": 1.0, "LC": 1.0, "LD": 1.0, "LE": 1.0,
    "LF": 1.0, "LG": 1.0, "LH": 1.0, "LI": 1.0,
    "LK": -7.49, "LL": 1.0, "LM": 1.0, "LN": 1.0,
    "LP": 20.26, "LQ": 33.6, "LR": 20.26, "LS": 1.0,
    "LT": 1.0, "LV": 1.0, "LW": 24.68, "LY": 1.0,
    "MA": 13.34, "MC": 1.0, "MD": 1.0, "ME": 1.0,
    "MF": 1.0, "MG": 1.0, "MH": 58.28, "MI": 1.0,
    "MK": 1.0, "ML": 1.0, "MM": -1.88, "MN": 1.0,
    "MP": 44.94, "MQ": -6.54, "MR": -6.54, "MS": 44.94,
    "MT": -1.88, "MV": 1.0, "MW": 1.0, "MY": 24.68,
    "NA": 1.0, "NC": -1.88, "ND": 1.0, "NE": 1.0,
    "NF": -14.03, "NG": -14.03, "NH": 1.0, "NI": 44.94,
    "NK": 24.68, "NL": 1.0, "NM": 1.0, "NN": 1.0,
    "NP": -1.88, "NQ": -6.54, "NR": 1.0, "NS": 1.0,
    "NT": -7.49, "NV": 1.0, "NW": -9.37, "NY": 1.0,
    "PA": 20.26, "PC": -6.54, "PD": -6.54, "PE": 18.38,
    "PF": 20.26, "PG": 1.0, "PH": 1.0, "PI": 1.0,
    "PK": 1.0, "PL": 1.0, "PM": -6.54, "PN": 1.0,
    "PP": 20.26, "PQ": 20.26, "PR": -6.54, "PS": 20.26,
    "PT": 1.0, "PV": 20.26, "PW": -1.88, "PY": 1.0,
    "QA": 1.0, "QC": -6.54, "QD": 20.26, "QE": 20.26,
    "QF": -6.54, "QG": 1.0, "QH": 1.0, "QI": 1.0,
    "QK": 1.0, "QL": 1.0, "QM": 1.0, "QN": 1.0,
    "QP": 20.26, "QQ": 20.26, "QR": 1.0, "QS": 44.94,
    "QT": 1.0, "QV": -6.54, "QW": 1.0, "QY": -6.54,
    "RA": 1.0, "RC": 1.0, "RD": 1.0, "RE": 1.0,
    "RF": 1.0, "RG": -7.49, "RH": 20.26, "RI": 1.0,
    "RK": 1.0, "RL": 1.0, "RM": 1.0, "RN": 13.34,
    "RP": 20.26, "RQ": 20.26, "RR": 58.28, "RS": 44.94,
    "RT": 1.0, "RV": 1.0, "RW": 58.28, "RY": -6.54,
    "SA": 1.0, "SC": 33.6, "SD": 1.0, "SE": 20.26,
    "SF": 1.0, "SG": 1.0, "SH": 1.0, "SI": 1.0,
    "SK": 1.0, "SL": 1.0, "SM": 1.0, "SN": 1.0,
    "SP": 44.94, "SQ": 20.26, "SR": 20.26, "SS": 20.26,
    "ST": 1.0, "SV": 1.0, "SW": 1.0, "SY": 1.0,
    "TA": 1.0, "TC": 1.0, "TD": 1.0, "TE": 20.26,
    "TF": 13.34, "TG": -7.49, "TH": 1.0, "TI": 1.0,
    "TK": 1.0, "TL": 1.0, "TM": 1.0, "TN": -14.03,
    "TP": 1.0, "TQ": -6.54, "TR": 1.0, "TS": 1.0,
    "TT": 1.0, "TV": 1.0, "TW": -14.03, "TY": 1.0,
    "VA": 1.0, "VC": 1.0, "VD": -14.03, "VE": 1.0,
    "VF": 1.0, "VG": -7.49, "VH": 1.0, "VI": 1.0,
    "VK": -1.88, "VL": 1.0, "VM": 1.0, "VN": 1.0,
    "VP": 20.26, "VQ": 1.0, "VR": 1.0, "VS": 1.0,
    "VT": -7.49, "VV": 1.0, "VW": 1.0, "VY": -6.54,
    "WA": -14.03, "WC": 1.0, "WD": 1.0, "WE": 1.0,
    "WF": 1.0, "WG": -9.37, "WH": 24.68, "WI": 1.0,
    "WK": 1.0, "WL": 13.34, "WM": 24.68, "WN": 13.34,
    "WP": 1.0, "WQ": 1.0, "WR": 1.0, "WS": 1.0,
    "WT": -14.03, "WV": -7.49, "WW": 1.0, "WY": 1.0,
    "YA": 24.68, "YC": 1.0, "YD": 24.68, "YE": -6.54,
    "YF": 1.0, "YG": -7.49, "YH": 13.34, "YI": 1.0,
    "YK": 1.0, "YL": 1.0, "YM": 44.94, "YN": 1.0,
    "YP": 13.34, "YQ": 1.0, "YR": -15.91, "YS": 1.0,
    "YT": -7.49, "YV": 1.0, "YW": -9.37, "YY": 13.34,
}


class ProteinAnalyzer:
    # mass in Da (free amino acid, not residue), pI of the free amino acid
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
        """Composition, ratios and the usual indices. Note 'molecular_weight'
        and 'pI' here are per-residue averages, not whole-protein values."""
        if not sequence:
            return {}

        n_residues = len(sequence)

        composition = {}
        for aa in sequence:
            composition[aa] = composition.get(aa, 0) + 1

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

        hydrophobicity = ProteinAnalyzer._calculate_hydrophobicity(sequence)
        aliphatic_index = ProteinAnalyzer._calculate_aliphatic_index(sequence)
        instability_index = ProteinAnalyzer._calculate_instability_index(sequence)

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
        """Mean Kyte-Doolittle; unknown residues count as 0."""
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
        """Percent A/I/L/V - not the volume-weighted Ikai aliphatic index."""
        aliphatic_values = {"A": 1, "I": 1, "L": 1, "V": 1}

        aliphatic_count = sum(1 for aa in sequence if aa in aliphatic_values)
        return (aliphatic_count / len(sequence)) * 100 if sequence else 0

    @staticmethod
    def _calculate_instability_index(sequence: str) -> float:
        """ProtParam instability index: (10/N) * sum of dipeptide weights.
        >40 means predicted unstable. Ambiguous residues (X, B, Z, U) get
        weight 0; 0.0 for sequences under 2 residues."""
        sequence = sequence.upper()
        n = len(sequence)
        if n < 2:
            return 0.0

        total = sum(
            DIWV.get(sequence[i : i + 2], 0.0) for i in range(n - 1)
        )
        return (10.0 / n) * total

    @staticmethod
    def find_motifs(sequence: str, motifs: List[str]) -> Dict[str, int]:
        """Plain substring counts, non-overlapping, case-insensitive."""
        results = {}
        for motif in motifs:
            count = sequence.upper().count(motif.upper())
            results[motif] = count
        return results

    @staticmethod
    def is_stable(instability_index: float) -> bool:
        """II <= 40 is the usual stable/unstable cut."""
        return instability_index <= 40.0

    @staticmethod
    def calculate_domain_scores(sequence: str, window: int = 19) -> Dict[str, float]:
        """Crude domain-propensity scores in [0, 1] for signal_peptide,
        transmembrane, coiled_coil, disorder and nuclear_localization.

        Rough proxies for when TMHMM/SignalP/IUPred aren't available - don't
        report these as predictions. window is the TM scan width in residues.
        """
        if not sequence:
            return {
                "signal_peptide": 0.0,
                "transmembrane": 0.0,
                "coiled_coil": 0.0,
                "disorder": 0.0,
                "nuclear_localization": 0.0,
            }

        sequence = sequence.upper()
        hydro_values = {
            "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
            "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
            "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
            "Y": -1.3, "V": 4.2,
        }
        hydro = np.array([hydro_values.get(aa, 0.0) for aa in sequence])
        n = len(sequence)

        # signal peptides have a hydrophobic core near the N-terminus
        n_term = hydro[: min(30, n)]
        signal_peptide = float(np.clip((n_term.mean() + 4.5) / 9.0, 0, 1)) if len(n_term) else 0.0

        # best window, not the mean - a TM helix is a sustained stretch
        if n >= window:
            window_means = np.convolve(hydro, np.ones(window) / window, mode="valid")
            max_window_mean = float(window_means.max())
        else:
            max_window_mean = float(hydro.mean())
        transmembrane = float(np.clip((max_window_mean + 1.0) / 3.0, 0, 1))

        # aliphatic density standing in for heptad repeats; real detection
        # needs COILS-style position-specific scoring
        aliphatic = sum(1 for aa in sequence if aa in "LIVM") / n
        coiled_coil = float(np.clip(aliphatic * 1.5, 0, 1))

        # disordered regions run polar/charged and light on bulky hydrophobics
        charged_polar = sum(1 for aa in sequence if aa in "DEKRQNSTH") / n
        bulky_hydrophobic = sum(1 for aa in sequence if aa in "FWYILV") / n
        disorder = float(np.clip(charged_polar - 0.3 * bulky_hydrophobic + 0.3, 0, 1))

        # K/R-rich 4-mers, loosely after the SV40 PKKKRKV NLS
        basic_runs = sum(
            1 for i in range(n - 3)
            if sum(1 for aa in sequence[i : i + 4] if aa in "KR") >= 3
        )
        nuclear_localization = float(np.clip(basic_runs / max(1, n / 20), 0, 1))

        return {
            "signal_peptide": signal_peptide,
            "transmembrane": transmembrane,
            "coiled_coil": coiled_coil,
            "disorder": disorder,
            "nuclear_localization": nuclear_localization,
        }
