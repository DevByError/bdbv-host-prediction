"""Taxonomy-based distance and risk features."""

import numpy as np
from typing import Dict, List, Optional, Tuple
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class PhylogeneticFeatures:
    """Taxonomic lookup tables and the features derived from them.

    Not a real phylogeny - order/family/genus rank distances standing in for
    branch lengths.
    """

    TAXONOMY = {
        "Homo_sapiens": {"order": "Primates", "family": "Hominidae", "genus": "Homo"},
        "Pan_troglodytes": {"order": "Primates", "family": "Hominidae", "genus": "Pan"},
        "Gorilla_gorilla": {
            "order": "Primates",
            "family": "Hominidae",
            "genus": "Gorilla",
        },
        "Rousettus_aegyptiacus": {
            "order": "Chiroptera",
            "family": "Pteropodidae",
            "genus": "Rousettus",
        },
        "Eidolon_helvum": {
            "order": "Chiroptera",
            "family": "Pteropodidae",
            "genus": "Eidolon",
        },
        "Myotis_myotis": {
            "order": "Chiroptera",
            "family": "Vespertilionidae",
            "genus": "Myotis",
        },
        "Rhinolophus_ferrumequinum": {
            "order": "Chiroptera",
            "family": "Rhinolophidae",
            "genus": "Rhinolophus",
        },
        "Mastomys_natalensis": {
            "order": "Rodentia",
            "family": "Muridae",
            "genus": "Mastomys",
        },
        "Sus_scrofa": {"order": "Artiodactyla", "family": "Suidae", "genus": "Sus"},
        "Capra_hircus": {
            "order": "Artiodactyla",
            "family": "Bovidae",
            "genus": "Capra",
        },
        "Ovis_aries": {"order": "Artiodactyla", "family": "Bovidae", "genus": "Ovis"},
    }

    # between-order distances, hand-set and symmetric
    TAXONOMIC_DISTANCE = {
        ("Primates", "Primates"): 0.0,
        ("Primates", "Chiroptera"): 0.7,
        ("Primates", "Rodentia"): 0.8,
        ("Primates", "Artiodactyla"): 0.9,
        ("Chiroptera", "Primates"): 0.7,
        ("Chiroptera", "Chiroptera"): 0.0,
        ("Chiroptera", "Rodentia"): 0.6,
        ("Chiroptera", "Artiodactyla"): 0.7,
        ("Rodentia", "Primates"): 0.8,
        ("Rodentia", "Chiroptera"): 0.6,
        ("Rodentia", "Rodentia"): 0.0,
        ("Rodentia", "Artiodactyla"): 0.5,
        ("Artiodactyla", "Primates"): 0.9,
        ("Artiodactyla", "Chiroptera"): 0.7,
        ("Artiodactyla", "Rodentia"): 0.5,
        ("Artiodactyla", "Artiodactyla"): 0.0,
    }

    @classmethod
    def get_taxonomic_distance(cls, species1: str, species2: str) -> float:
        """0 (same genus) to 1. Either species unknown -> 1.0."""
        tax1 = cls.TAXONOMY.get(species1, {})
        tax2 = cls.TAXONOMY.get(species2, {})

        if not tax1 or not tax2:
            return 1.0

        if tax1.get("order") == tax2.get("order"):
            if tax1.get("family") == tax2.get("family"):
                if tax1.get("genus") == tax2.get("genus"):
                    return 0.0
                return 0.3
            return 0.6

        return cls.TAXONOMIC_DISTANCE.get(
            (tax1.get("order", ""), tax2.get("order", "")), 1.0
        )

    @classmethod
    def get_phylogenetic_proximity(cls, species: str, known_hosts: List[str]) -> float:
        """1 - distance to the nearest known host. Higher = closer."""
        if not known_hosts:
            return 0.5

        distances = [cls.get_taxonomic_distance(species, host) for host in known_hosts]
        min_distance = min(distances) if distances else 1.0

        proximity = 1 - min_distance

        return proximity

    @classmethod
    def get_order_risk_factor(cls, order: str) -> float:
        """Prior from reported filovirus host range; 0.3 for anything unlisted."""
        risk_factors = {
            "Primates": 0.8,
            "Chiroptera": 0.6,
            "Rodentia": 0.3,
            "Artiodactyla": 0.2,
            "Carnivora": 0.2,
            "Perissodactyla": 0.1,
            "Unknown": 0.3,
        }

        return risk_factors.get(order, 0.3)

    @classmethod
    def calculate_phylogenetic_features(cls, species: str) -> Dict:
        taxonomy = cls.TAXONOMY.get(species, {})
        order = taxonomy.get("order", "Unknown")
        family = taxonomy.get("family", "Unknown")
        genus = taxonomy.get("genus", "Unknown")

        order_risk = cls.get_order_risk_factor(order)

        known_bdbv_hosts = [
            "Rousettus_aegyptiacus",
            "Eidolon_helvum",
            "Homo_sapiens",
            "Pan_troglodytes",
        ]
        proximity = cls.get_phylogenetic_proximity(species, known_bdbv_hosts)

        return {
            "order": order,
            "family": family,
            "genus": genus,
            "order_risk": order_risk,
            "proximity_to_known_hosts": proximity,
            "is_chiroptera": 1 if order == "Chiroptera" else 0,
            "is_primate": 1 if order == "Primates" else 0,
            "is_artiodactyla": 1 if order == "Artiodactyla" else 0,
            # name-length proxy, carries no real signal
            "taxonomic_rank_score": (len(order) + len(family)) / 20,
        }

    @classmethod
    def calculate_host_species_similarity(cls, species1: str, species2: str) -> float:
        """60% taxonomy, 40% ecology - but the ecology term is still a constant,
        so this is really just taxonomic similarity rescaled."""
        tax_dist = cls.get_taxonomic_distance(species1, species2)
        tax_similarity = 1 - tax_dist

        feat1 = cls.calculate_phylogenetic_features(species1)
        feat2 = cls.calculate_phylogenetic_features(species2)

        eco_similarity = 0.5  # placeholder until there's real ecological data

        similarity = tax_similarity * 0.6 + eco_similarity * 0.4

        return similarity
