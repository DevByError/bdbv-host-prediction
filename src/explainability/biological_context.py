"""Lookup table tying feature names to what they mean biologically."""

import pandas as pd
from typing import Dict, List, Optional


class BiologicalContext:
    BIOLOGICAL_MAP = {
        # Genomic features
        "gc_content": {
            "category": "Genomic",
            "description": "GC content of viral genome",
            "biological_meaning": "Affects genome stability, codon usage, and host adaptation",
            "pathway": "Viral evolution",
        },
        "cpg_frequency": {
            "category": "Genomic",
            "description": "CpG dinucleotide frequency",
            "biological_meaning": "Influences immune recognition via TLR9, host evasion",
            "pathway": "Immune evasion",
        },
        "genome_length": {
            "category": "Genomic",
            "description": "Viral genome length",
            "biological_meaning": "Affects replication efficiency and packaging",
            "pathway": "Viral replication",
        },
        # Codon usage
        "cai_value": {
            "category": "Codon Usage",
            "description": "Codon Adaptation Index",
            "biological_meaning": "Measures translational efficiency in host cells",
            "pathway": "Protein synthesis",
        },
        "rscu_mean": {
            "category": "Codon Usage",
            "description": "Relative Synonymous Codon Usage",
            "biological_meaning": "Indicates codon bias and host adaptation",
            "pathway": "Protein synthesis",
        },
        "enc_value": {
            "category": "Codon Usage",
            "description": "Effective Number of Codons",
            "biological_meaning": "Measures codon usage bias strength",
            "pathway": "Protein synthesis",
        },
        # Glycoprotein
        "gp_hydrophobicity": {
            "category": "Glycoprotein",
            "description": "GP hydrophobic index",
            "biological_meaning": "Affects protein folding and membrane interaction",
            "pathway": "Viral entry",
        },
        "gp_binding_energy": {
            "category": "Glycoprotein",
            "description": "NPC1 receptor binding energy",
            "biological_meaning": "Directly affects virus entry efficiency",
            "pathway": "Viral entry",
        },
        # Host traits
        "body_mass_kg": {
            "category": "Host Traits",
            "description": "Host body mass",
            "biological_meaning": "Correlates with metabolic rate and viral persistence",
            "pathway": "Host physiology",
        },
        "lifespan_years": {
            "category": "Host Traits",
            "description": "Host lifespan",
            "biological_meaning": "Longer lifespan species may maintain viruses longer",
            "pathway": "Host physiology",
        },
        # Ecological
        "habitat_type": {
            "category": "Ecological",
            "description": "Primary habitat",
            "biological_meaning": "Determines exposure to viruses and vectors",
            "pathway": "Ecology",
        },
        "diet_type": {
            "category": "Ecological",
            "description": "Dietary preference",
            "biological_meaning": "Affects exposure routes and immune status",
            "pathway": "Ecology",
        },
        # BDBV-specific
        "bdbv_npc1_residue_match": {
            "category": "BDBV-Specific",
            "description": "NPC1 receptor residue matching",
            "biological_meaning": "Direct molecular compatibility for BDBV entry",
            "pathway": "Viral entry",
        },
        "bdbv_gp_binding_affinity": {
            "category": "BDBV-Specific",
            "description": "BDBV GP binding affinity",
            "biological_meaning": "Species-specific binding efficiency",
            "pathway": "Viral entry",
        },
    }

    @classmethod
    def map_feature(cls, feature_name: str) -> Dict:
        """Substring match, so engineered variants like gc_content_log resolve to the base entry."""
        for key, context in cls.BIOLOGICAL_MAP.items():
            if key in feature_name.lower():
                return context

        return {
            "category": "Unknown",
            "description": feature_name,
            "biological_meaning": "Unknown biological relevance",
            "pathway": "Unknown",
        }

    @classmethod
    def annotate_importance(cls, importance_df: pd.DataFrame) -> pd.DataFrame:
        """Adds category, biological_meaning and pathway columns in place."""
        importance_df["category"] = importance_df["feature"].apply(
            lambda x: cls.map_feature(x)["category"]
        )
        importance_df["biological_meaning"] = importance_df["feature"].apply(
            lambda x: cls.map_feature(x)["biological_meaning"]
        )
        importance_df["pathway"] = importance_df["feature"].apply(
            lambda x: cls.map_feature(x)["pathway"]
        )

        return importance_df

    @classmethod
    def get_category_summary(cls, importance_df: pd.DataFrame) -> pd.DataFrame:
        summary = (
            importance_df.groupby("category")
            .agg({"importance": ["sum", "mean", "count"]})
            .reset_index()
        )

        summary.columns = [
            "category",
            "total_importance",
            "mean_importance",
            "n_features",
        ]
        summary = summary.sort_values("total_importance", ascending=False)

        return summary

    @classmethod
    def get_pathway_insights(cls, importance_df: pd.DataFrame) -> Dict:
        pathway_summary = (
            importance_df.groupby("pathway")
            .agg({"importance": "sum"})
            .sort_values("importance", ascending=False)
        )

        top_pathways = pathway_summary.head(5)

        insights = {
            "top_pathways": top_pathways.index.tolist(),
            "pathway_importance": pathway_summary.to_dict()["importance"],
            "dominant_category": importance_df.groupby("category")["importance"]
            .sum()
            .idxmax(),
            "key_features": importance_df.head(10).to_dict("records"),
        }

        # canned sentences, triggered only if the pathway lands in the top 3
        interpretations = []

        if "Viral entry" in top_pathways.index.tolist()[:3]:
            interpretations.append(
                "Virus entry mechanisms are strong predictors of susceptibility"
            )

        if "Immune evasion" in top_pathways.index.tolist()[:3]:
            interpretations.append(
                "Immune evasion strategies are important for host adaptation"
            )

        if "Host physiology" in top_pathways.index.tolist()[:3]:
            interpretations.append(
                "Host physiological traits correlate with susceptibility"
            )

        if "Protein synthesis" in top_pathways.index.tolist()[:3]:
            interpretations.append(
                "Codon adaptation to host translational machinery is significant"
            )

        insights["interpretations"] = interpretations

        return insights
