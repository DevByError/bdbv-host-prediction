"""
Synthetic virus-host data for pipeline testing.

The associations here are made up (loosely literature-informed) - fine for
exercising the pipeline, not for any biological claim. Real work goes through
DataLoader(data_path=...). Column names in _generate_sample are the schema the
rest of the package expects.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class DataGenerator:
    """Synthetic BDBV host-prediction data."""

    # label -> relative risk weight; used both to draw values and to bias
    # the infection probability
    DIET_RISK = {
        "Frugivore": 0.20,
        "Omnivore": 0.15,
        "Herbivore": 0.10,
        "Insectivore": 0.10,
        "Carnivore": 0.15,
    }
    SOCIAL_RISK = {"Solitary": 1, "Pairs": 2, "Groups": 3, "Colonies": 4}
    HABITAT_OPTIONS = ["Forest", "Urban", "Caves", "Savanna", "Farm"]
    CONTACT_RISK = {"Low": 0.05, "Medium": 0.15, "High": 0.30}

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

        # approximate literature values (Kuhn et al. 2011 for genome/GP lengths)
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

        # covariates picked so the usual correlations show up, e.g. frugivorous
        # bats -> forest/cave -> high human contact
        self.known_hosts = {
            "Homo_sapiens": {
                "order": "Primates",
                "family": "Hominidae",
                "risk": 0.85,
                "group": "Positive",
                "diet": "Omnivore",
                "social": "Groups",
                "habitat": "Urban",
                "contact": "High",
            },
            "Pan_troglodytes": {
                "order": "Primates",
                "family": "Hominidae",
                "risk": 0.82,
                "group": "Positive",
                "diet": "Frugivore",
                "social": "Groups",
                "habitat": "Forest",
                "contact": "Medium",
            },
            "Gorilla_gorilla": {
                "order": "Primates",
                "family": "Hominidae",
                "risk": 0.80,
                "group": "Positive",
                "diet": "Herbivore",
                "social": "Groups",
                "habitat": "Forest",
                "contact": "Medium",
            },
            "Rousettus_aegyptiacus": {
                "order": "Chiroptera",
                "family": "Pteropodidae",
                "risk": 0.72,
                "group": "Positive",
                "diet": "Frugivore",
                "social": "Colonies",
                "habitat": "Caves",
                "contact": "Medium",
            },
            "Eidolon_helvum": {
                "order": "Chiroptera",
                "family": "Pteropodidae",
                "risk": 0.68,
                "group": "Positive",
                "diet": "Frugivore",
                "social": "Colonies",
                "habitat": "Forest",
                "contact": "Medium",
            },
            "Mastomys_natalensis": {
                "order": "Rodentia",
                "family": "Muridae",
                "risk": 0.15,
                "group": "Negative",
                "diet": "Omnivore",
                "social": "Colonies",
                "habitat": "Farm",
                "contact": "High",
            },
            "Sus_scrofa": {
                "order": "Artiodactyla",
                "family": "Suidae",
                "risk": 0.20,
                "group": "Negative",
                "diet": "Omnivore",
                "social": "Groups",
                "habitat": "Farm",
                "contact": "High",
            },
            "Capra_hircus": {
                "order": "Artiodactyla",
                "family": "Bovidae",
                "risk": 0.18,
                "group": "Negative",
                "diet": "Herbivore",
                "social": "Groups",
                "habitat": "Farm",
                "contact": "High",
            },
        }

    def generate_dataset(
        self, n_samples: int = 200, bdbv_ratio: float = 0.3
    ) -> pd.DataFrame:
        """One row per virus-host pair; infection_status is the binary target."""
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        if not 0.0 <= bdbv_ratio <= 1.0:
            raise ValueError(f"bdbv_ratio must be in [0, 1], got {bdbv_ratio}")

        logger.info(f"Generating {n_samples} synthetic samples ({bdbv_ratio:.0%} BDBV)")

        n_bdbv = int(round(n_samples * bdbv_ratio))
        n_other = n_samples - n_bdbv

        other_viruses = [v for v in self.viruses if v != "BDBV"]
        other_weights = np.array([0.35, 0.25, 0.20, 0.20])  # EBOV, SUDV, MARV, RESTV
        other_weights = other_weights / other_weights.sum()

        virus_list = ["BDBV"] * n_bdbv + list(
            self._rng.choice(other_viruses, size=n_other, p=other_weights)
        )
        self._rng.shuffle(virus_list)

        data = [self._generate_sample(virus) for virus in virus_list]

        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} samples with {len(df.columns)} features")
        logger.info(
            f"Class balance: {df['infection_status'].mean():.1%} positive "
            f"({int(df['infection_status'].sum())}/{len(df)})"
        )

        return df

    def _generate_sample(self, virus: str) -> Dict:
        virus_props = self.viruses[virus]

        if self._rng.random() < 0.6:
            host = self._rng.choice(list(self.known_hosts.keys()))
            info = self.known_hosts[host]
            host_risk = info["risk"]
            host_order = info["order"]
            host_family = info["family"]
            diet_type = info["diet"]
            social_behavior = info["social"]
            habitat_type = info["habitat"]
            human_contact_frequency = info["contact"]
        else:
            host = self._generate_unknown_host()
            host_risk = self._estimate_risk(host)
            host_order = "Chiroptera" if self._looks_like_bat(host) else "Other"
            host_family = "Unknown"
            diet_type = self._rng.choice(list(self.DIET_RISK.keys()))
            social_behavior = self._rng.choice(list(self.SOCIAL_RISK.keys()))
            habitat_type = self._rng.choice(self.HABITAT_OPTIONS)
            human_contact_frequency = self._rng.choice(list(self.CONTACT_RISK.keys()))

        infection_score = self._calculate_infection_score(
            virus_props,
            host_risk,
            host_order,
            diet_type,
            human_contact_frequency,
        )

        infection = 1 if infection_score > 0.55 else 0
        if self._rng.random() < 0.05:  # label noise, stands in for surveillance error
            infection = 1 - infection

        return {
            # Virus / genomic features
            "virus_species": virus,
            "genome_length": virus_props["genome"] + self._rng.normal(0, 50),
            "gc_content": virus_props["gc"] + self._rng.normal(0, 0.3),
            "cpg_frequency": virus_props["cpg"] + self._rng.normal(0, 0.001),
            # Glycoprotein features
            "gp_length": virus_props["gp_len"] + self._rng.normal(0, 3),
            "gp_hydrophobicity": self._rng.normal(-0.45, 0.05),
            "gp_binding_energy": virus_props["binding"] + self._rng.normal(0, 0.3),
            # Host identity / taxonomy
            "host_species": host,
            "host_genus": host.split("_")[0],
            "host_order": host_order,
            "host_family": host_family,
            # Host life-history traits
            "body_mass_kg": float(np.exp(self._rng.normal(2, 1.5))),
            "lifespan_years": float(np.exp(self._rng.normal(2.5, 0.8))),
            "geographic_range_km2": float(np.exp(self._rng.normal(14, 2))),
            # Host ecology / behavior
            "diet_type": diet_type,
            "social_behavior": social_behavior,
            "habitat_type": habitat_type,
            "human_contact_frequency": human_contact_frequency,
            # NaN for non-BDBV rows on purpose - those assays only get run for
            # the virus of interest
            "bdbv_gp_binding_affinity": (
                virus_props["binding"] + self._rng.normal(0, 0.2)
                if virus == "BDBV"
                else np.nan
            ),
            "bdbv_npc1_residue_match": (
                self._rng.uniform(0.7, 0.95) if virus == "BDBV" else np.nan
            ),
            # Target
            "infection_status": infection,
        }

    def _generate_unknown_host(self) -> str:
        """Made-up species name outside the known_hosts table."""
        genera = [
            "Myotis",
            "Rhinolophus",
            "Hipposideros",
            "Pteropus",
            "Epomophorus",
            "Chaerephon",
            "Taphozous",
        ]
        species = ["sp", "nov", "crypticus", "africanus", "occidentalis"]

        genus = self._rng.choice(genera)
        species_name = self._rng.choice(species)

        return f"{genus}_{species_name}"

    @staticmethod
    def _looks_like_bat(host: str) -> bool:
        """Crude genus-name check for bats."""
        bat_genera = ("myotis", "rhinolophus", "hipposideros", "pteropus",
                      "epomophorus", "chaerephon", "taphozous", "bat")
        return any(g in host.lower() for g in bat_genera)

    def _estimate_risk(self, host: str) -> float:
        """Baseline risk for an unknown host, guessed from the name."""
        if self._looks_like_bat(host):
            return float(np.clip(0.45 + self._rng.normal(0, 0.1), 0, 1))
        elif any(x in host.lower() for x in ["primate", "monkey", "pan", "gorilla"]):
            return float(np.clip(0.55 + self._rng.normal(0, 0.1), 0, 1))
        else:
            return float(np.clip(0.25 + self._rng.normal(0, 0.1), 0, 1))

    def _calculate_infection_score(
        self,
        virus_props: Dict,
        host_risk: float,
        host_order: str,
        diet_type: str,
        human_contact_frequency: str,
    ) -> float:
        """Hand-tuned score in [0, 1]; thresholded to get the label."""
        score = host_risk * 0.35
        score += virus_props["risk_factor"] * 0.20
        score += (virus_props["gc"] / 45) * 0.08
        score += (1 - virus_props["cpg"] / 0.05) * 0.08

        if host_order == "Primates":
            score += 0.10
        elif host_order == "Chiroptera":
            score += 0.06

        score += self.DIET_RISK.get(diet_type, 0.1) * 0.3
        score += self.CONTACT_RISK.get(human_contact_frequency, 0.1) * 0.5

        score += self._rng.normal(0, 0.08)

        return float(np.clip(score, 0, 1))
