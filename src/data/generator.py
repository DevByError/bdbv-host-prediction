"""
Synthetic data generator for testing and demonstration.
Based on actual biological distributions.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class DataGenerator:
    """
    Generate realistic synthetic data for BDBV host prediction.

    Based on:
    - Actual filovirus genome properties
    - Known host species traits
    - Ecological and biological relationships
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)

        # Virus properties from literature
        self.viruses = {
            "BDBV": {
                "genome": 18950,
                "gc": 41.3,
                "cpg": 0.041,
                "gp_len": 676,
                "binding": -8.2,
                "risk_factor": 0.3,
            },
            "EBOV": {
                "genome": 18959,
                "gc": 41.2,
                "cpg": 0.042,
                "gp_len": 676,
                "binding": -7.9,
                "risk_factor": 0.25,
            },
            "SUDV": {
                "genome": 18875,
                "gc": 41.0,
                "cpg": 0.044,
                "gp_len": 672,
                "binding": -7.5,
                "risk_factor": 0.2,
            },
            "MARV": {
                "genome": 19100,
                "gc": 39.0,
                "cpg": 0.045,
                "gp_len": 680,
                "binding": -6.8,
                "risk_factor": 0.15,
            },
            "RESTV": {
                "genome": 18850,
                "gc": 41.1,
                "cpg": 0.043,
                "gp_len": 675,
                "binding": -6.5,
                "risk_factor": 0.1,
            },
        }

        # Host species with known susceptibility
        self.known_hosts = {
            # Positive hosts
            "Homo_sapiens": {
                "order": "Primates",
                "family": "Hominidae",
                "risk": 0.85,
                "group": "Positive",
            },
            "Pan_troglodytes": {
                "order": "Primates",
                "family": "Hominidae",
                "risk": 0.82,
                "group": "Positive",
            },
            "Gorilla_gorilla": {
                "order": "Primates",
                "family": "Hominidae",
                "risk": 0.80,
                "group": "Positive",
            },
            "Rousettus_aegyptiacus": {
                "order": "Chiroptera",
                "family": "Pteropodidae",
                "risk": 0.72,
                "group": "Positive",
            },
            "Eidolon_helvum": {
                "order": "Chiroptera",
                "family": "Pteropodidae",
                "risk": 0.68,
                "group": "Positive",
            },
            # Negative hosts
            "Mastomys_natalensis": {
                "order": "Rodentia",
                "family": "Muridae",
                "risk": 0.15,
                "group": "Negative",
            },
            "Sus_scrofa": {
                "order": "Artiodactyla",
                "family": "Suidae",
                "risk": 0.20,
                "group": "Negative",
            },
            "Capra_hircus": {
                "order": "Artiodactyla",
                "family": "Bovidae",
                "risk": 0.18,
                "group": "Negative",
            },
        }

    def generate_dataset(
        self, n_samples: int = 200, bdbv_ratio: float = 0.3
    ) -> pd.DataFrame:
        """
        Generate complete dataset.

        Args:
            n_samples: Total number of samples
            bdbv_ratio: Proportion of BDBV samples

        Returns:
            pd.DataFrame: Synthetic dataset
        """
        logger.info(f"Generating {n_samples} synthetic samples ({bdbv_ratio:.0%} BDBV)")

        n_bdbv = int(n_samples * bdbv_ratio)
        n_other = n_samples - n_bdbv

        # Select viruses
        other_viruses = list(self.viruses.keys())
        other_viruses.remove("BDBV")

        virus_list = ["BDBV"] * n_bdbv + list(
            np.random.choice(
                other_viruses,
                n_other,
                p=[0.3, 0.2, 0.15, 0.15, 0.2],  # EBOV, SUDV, MARV, RESTV, others
            )
        )
        np.random.shuffle(virus_list)

        # Generate data
        data = []
        for virus in virus_list:
            record = self._generate_sample(virus)
            data.append(record)

        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} samples with {len(df.columns)} features")

        return df

    def _generate_sample(self, virus: str) -> Dict:
        """Generate a single sample"""
        virus_props = self.viruses[virus]

        # Select host (weighted towards known hosts)
        if np.random.random() < 0.6:
            host = np.random.choice(list(self.known_hosts.keys()))
            host_info = self.known_hosts[host]
            host_risk = host_info["risk"]
            host_order = host_info["order"]
            host_family = host_info["family"]
            host_group = host_info["group"]
        else:
            # Unknown host - estimate based on phylogeny
            host = self._generate_unknown_host()
            host_risk = self._estimate_risk(host)
            host_order = "Chiroptera" if "bat" in host.lower() else "Other"
            host_family = "Unknown"
            host_group = "Unknown"

        # Calculate infection probability
        infection_score = self._calculate_infection_score(
            virus, virus_props, host_risk, host_order
        )

        # Determine infection status
        infection = 1 if infection_score > 0.55 else 0
        if np.random.random() < 0.05:  # Noise
            infection = 1 - infection

        # Generate features
        return {
            # Virus features
            "virus_species": virus,
            "genome_length": virus_props["genome"] + np.random.normal(0, 50),
            "gc_content": virus_props["gc"] + np.random.normal(0, 0.3),
            "cpg_frequency": virus_props["cpg"] + np.random.normal(0, 0.001),
            # Glycoprotein features
            "gp_length": virus_props["gp_len"] + np.random.normal(0, 3),
            "gp_hydrophobicity": np.random.normal(-0.45, 0.05),
            "gp_binding_energy": virus_props["binding"] + np.random.normal(0, 0.3),
            # Host features
            "host_species": host,
            "host_order": host_order,
            "host_family": host_family,
            "body_mass_kg": np.exp(np.random.normal(2, 1.5)),
            "lifespan_years": np.exp(np.random.normal(2.5, 0.8)),
            "geographic_range_km2": np.exp(np.random.normal(14, 2)),
            # BDBV-specific features (only for BDBV)
            "bdbv_gp_binding_affinity": (
                virus_props["binding"] + np.random.normal(0, 0.2)
                if virus == "BDBV"
                else np.nan
            ),
            "bdbv_npc1_residue_match": (
                np.random.uniform(0.7, 0.95) if virus == "BDBV" else np.nan
            ),
            # Target
            "infection_status": infection,
        }

    def _generate_unknown_host(self) -> str:
        """Generate a plausible unknown host species"""
        genera = [
            "Myotis",
            "Rhinolophus",
            "Hipposideros",
            "Pteropus",
            "Epomophorus",
            "Chaerephon",
            "Taphozous",
        ]
        species = ["sp.", "nov.", "crypticus", "africanus", "occidentalis"]

        genus = np.random.choice(genera)
        species_name = np.random.choice(species)

        return f"{genus}_{species_name}"

    def _estimate_risk(self, host: str) -> float:
        """Estimate risk for unknown host based on name patterns"""
        if "bat" in host.lower() or any(
            x in host.lower() for x in ["myotis", "rhinolophus"]
        ):
            return 0.45 + np.random.normal(0, 0.1)
        elif any(x in host.lower() for x in ["primate", "monkey"]):
            return 0.55 + np.random.normal(0, 0.1)
        else:
            return 0.25 + np.random.normal(0, 0.1)

    def _calculate_infection_score(
        self, virus: str, virus_props: Dict, host_risk: float, host_order: str
    ) -> float:
        """Calculate infection probability based on multiple factors"""

        # Base risk
        score = host_risk * 0.40

        # Virus effect
        virus_effect = virus_props["risk_factor"]
        score += virus_effect * 0.25

        # Genomic factors
        score += (virus_props["gc"] / 45) * 0.10
        score += (1 - virus_props["cpg"] / 0.05) * 0.10

        # Host order effect
        if host_order == "Primates":
            score += 0.10
        elif host_order == "Chiroptera":
            score += 0.05

        # Add noise
        score += np.random.normal(0, 0.1)

        # Clip to [0,1]
        return np.clip(score, 0, 1)
