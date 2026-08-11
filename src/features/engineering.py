"""
Feature engineering for BDBV host prediction.
Creates biologically relevant features from raw data.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering pipeline for BDBV host prediction.
    """

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the data."""
        logger.info("Starting feature engineering...")

        df_engineered = self.engineer_features(df)
        df_engineered = self._encode_categorical(df_engineered)
        df_engineered = self._scale_numerical(df_engineered)

        self.feature_columns = df_engineered.columns.tolist()
        self.is_fitted = True

        logger.info(f"Engineered {len(self.feature_columns)} features")
        return df_engineered

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted encoders/scaler"""
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")

        df_engineered = self.engineer_features(df)
        df_engineered = self._encode_categorical(df_engineered, fit=False)
        df_engineered = self._scale_numerical(df_engineered, fit=False)

        for col in self.feature_columns:
            if col not in df_engineered.columns:
                df_engineered[col] = 0

        return df_engineered[self.feature_columns]

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all engineered features."""
        df_engineered = df.copy()

        # 1. Viral features
        df_engineered = self._add_viral_features(df_engineered)

        # 2. Host features
        df_engineered = self._add_host_features(df_engineered)

        # 3. Interaction features
        df_engineered = self._add_interaction_features(df_engineered)

        # 4. BDBV-specific features
        df_engineered = self._add_bdbv_features(df_engineered)

        # 5. Risk scores
        df_engineered = self._add_risk_scores(df_engineered)

        # ===== FIX: Drop problematic columns =====
        columns_to_drop = ["notes", "host_genus"]
        for col in columns_to_drop:
            if col in df_engineered.columns:
                df_engineered = df_engineered.drop(columns=[col])
                logger.info(f"Dropped '{col}' from features")

        return df_engineered

    def _add_viral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add viral genome features"""
        if "gc_content" in df.columns:
            df["gc_content_squared"] = df["gc_content"] ** 2
            df["gc_content_log"] = np.log1p(
                df["gc_content"] - df["gc_content"].min() + 1
            )

        if "cpg_frequency" in df.columns:
            df["cpg_density"] = df["cpg_frequency"] / df["gc_content"].clip(lower=0.01)
            df["cpg_log"] = np.log1p(df["cpg_frequency"])

        if "glycoprotein_length" in df.columns:
            df["gp_length_ratio"] = (
                df["glycoprotein_length"] / df["viral_genome_length"]
            )

        if "gp_hydrophobicity" in df.columns:
            df["gp_hydro_abs"] = np.abs(df["gp_hydrophobicity"])

        return df

    def _add_host_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add host trait features"""
        if "body_mass_kg" in df.columns:
            df["body_mass_log"] = np.log1p(df["body_mass_kg"])
            df["body_mass_sqrt"] = np.sqrt(df["body_mass_kg"] + 1)

        if "lifespan_years" in df.columns:
            df["lifespan_log"] = np.log1p(df["lifespan_years"])

        if "geographic_range_km2" in df.columns:
            df["range_log"] = np.log1p(df["geographic_range_km2"])

        return df

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add virus-host interaction features"""
        # Host taxonomy binary features
        if "host_order" in df.columns:
            df["is_primate"] = (df["host_order"] == "Primates").astype(int)
            df["is_chiroptera"] = (df["host_order"] == "Chiroptera").astype(int)
            df["is_rodentia"] = (df["host_order"] == "Rodentia").astype(int)
            df["is_artiodactyla"] = (df["host_order"] == "Artiodactyla").astype(int)

        if "host_family" in df.columns:
            df["is_hominidae"] = (df["host_family"] == "Hominidae").astype(int)
            df["is_pteropodidae"] = (df["host_family"] == "Pteropodidae").astype(int)
            df["is_vespertilionidae"] = (
                df["host_family"] == "Vespertilionidae"
            ).astype(int)
            df["is_rhinolophidae"] = (df["host_family"] == "Rhinolophidae").astype(int)
            df["is_muridae"] = (df["host_family"] == "Muridae").astype(int)

        # Human contact risk
        if "human_contact_frequency" in df.columns:
            df["human_contact_score"] = (
                df["human_contact_frequency"]
                .map({"Low": 1, "Medium": 2, "High": 3})
                .fillna(1)
            )

        # Social behavior risk
        if "social_behavior" in df.columns:
            df["social_group_score"] = (
                df["social_behavior"]
                .map({"Solitary": 1, "Pairs": 2, "Groups": 3, "Colonies": 4})
                .fillna(1)
            )

        # Habitat binary features
        if "habitat_type" in df.columns:
            df["urban_habitat"] = (df["habitat_type"] == "Urban").astype(int)
            df["forest_habitat"] = (df["habitat_type"] == "Forest").astype(int)
            df["cave_habitat"] = (df["habitat_type"] == "Caves").astype(int)
            df["farm_habitat"] = (df["habitat_type"] == "Farm").astype(int)

        # Diet binary features
        if "diet_type" in df.columns:
            df["is_frugivore"] = (df["diet_type"] == "Frugivore").astype(int)
            df["is_omnivore"] = (df["diet_type"] == "Omnivore").astype(int)
            df["is_herbivore"] = (df["diet_type"] == "Herbivore").astype(int)
            df["is_insectivore"] = (df["diet_type"] == "Insectivore").astype(int)

        return df

    def _add_bdbv_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add BDBV-specific features"""
        # Virus binary features
        if "virus" in df.columns:
            df["is_bdbv"] = (df["virus"] == "BDBV").astype(int)
            df["is_ebov"] = (df["virus"] == "EBOV").astype(int)
            df["is_marv"] = (df["virus"] == "MARV").astype(int)
            df["is_sudv"] = (df["virus"] == "SUDV").astype(int)
            df["is_restv"] = (df["virus"] == "RESTV").astype(int)

        # BDBV risk score
        df["bdbv_risk"] = 0
        if "virus" in df.columns and "host_order" in df.columns:
            bdbv_mask = df["virus"] == "BDBV"
            df.loc[bdbv_mask & (df["host_order"] == "Chiroptera"), "bdbv_risk"] += 3
            df.loc[bdbv_mask & (df["host_order"] == "Primates"), "bdbv_risk"] += 2

        if "host_family" in df.columns:
            df.loc[bdbv_mask & (df["host_family"] == "Pteropodidae"), "bdbv_risk"] += 2
            df.loc[bdbv_mask & (df["host_family"] == "Hominidae"), "bdbv_risk"] += 1

        return df

    def _add_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add combined risk scores"""
        df["baseline_risk"] = 0

        if "host_order" in df.columns:
            order_risk = {
                "Primates": 0.4,
                "Chiroptera": 0.3,
                "Rodentia": 0.2,
                "Artiodactyla": 0.15,
            }
            df["taxonomic_risk"] = df["host_order"].map(order_risk).fillna(0.1)
            df["baseline_risk"] += df["taxonomic_risk"]

        if "diet_type" in df.columns:
            diet_risk = {
                "Frugivore": 0.2,
                "Omnivore": 0.15,
                "Herbivore": 0.1,
                "Insectivore": 0.1,
                "Carnivore": 0.15,
            }
            df["ecological_risk"] = df["diet_type"].map(diet_risk).fillna(0.1)
            df["baseline_risk"] += df["ecological_risk"]

        if "human_contact_score" in df.columns:
            df["contact_risk"] = df["human_contact_score"] * 0.1
            df["baseline_risk"] += df["contact_risk"]

        df["combined_risk"] = df["baseline_risk"].clip(0, 1)
        df["high_risk"] = (df["combined_risk"] > 0.6).astype(int)

        return df

    def _encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features"""
        df_encoded = df.copy()

        # Get all object/categorical columns
        categorical_cols = df_encoded.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    df_encoded[col] = df_encoded[col].astype(str)
                    df_encoded[col] = df_encoded[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                else:
                    df_encoded = df_encoded.drop(columns=[col])

        return df_encoded

    def _scale_numerical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features"""
        df_scaled = df.copy()

        exclude_cols = ["infection_status", "high_risk"]
        numerical_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]

        if fit and numerical_cols:
            df_scaled[numerical_cols] = self.scaler.fit_transform(
                df_scaled[numerical_cols]
            )
        elif numerical_cols:
            available_cols = [col for col in numerical_cols if col in df_scaled.columns]
            if available_cols:
                df_scaled[available_cols] = self.scaler.transform(
                    df_scaled[available_cols]
                )

        return df_scaled

    def get_feature_columns(self) -> List[str]:
        return self.feature_columns

    def get_feature_importance(self, model) -> pd.DataFrame:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            return pd.DataFrame(
                {"feature": self.feature_columns, "importance": importances}
            ).sort_values("importance", ascending=False)
        return pd.DataFrame()


# Convenience function
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    engineer = FeatureEngineer()
    return engineer.fit_transform(df)
