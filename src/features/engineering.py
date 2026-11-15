"""Feature engineering for the host prediction models."""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_fitted = False
        self._impute_medians: Dict[str, float] = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Train split only - everything else goes through transform()."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df)}")
        if df.empty:
            raise ValueError("Cannot fit FeatureEngineer on an empty DataFrame")

        logger.info("Starting feature engineering...")

        df_engineered = self.engineer_features(df)
        df_engineered = self._encode_categorical(df_engineered)
        df_engineered = self._scale_numerical(df_engineered)

        self.feature_columns = df_engineered.columns.tolist()
        self.is_fitted = True

        logger.info(f"Engineered {len(self.feature_columns)} features")
        return df_engineered

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reuses the fitted encoders/scaler; returns the fit-time columns in
        the fit-time order, zero-filling any that are absent."""
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df)}")
        if df.empty:
            raise ValueError("Cannot transform an empty DataFrame")

        df_engineered = self.engineer_features(df)
        df_engineered = self._encode_categorical(df_engineered, fit=False)
        df_engineered = self._scale_numerical(df_engineered, fit=False)

        for col in self.feature_columns:
            if col not in df_engineered.columns:
                df_engineered[col] = 0

        return df_engineered[self.feature_columns]

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_engineered = df.copy()

        df_engineered = self._add_viral_features(df_engineered)
        df_engineered = self._add_host_features(df_engineered)
        df_engineered = self._add_interaction_features(df_engineered)
        df_engineered = self._add_bdbv_features(df_engineered)
        df_engineered = self._add_risk_scores(df_engineered)

        # free text and near-unique ids, no use as features
        columns_to_drop = ["notes", "host_genus"]
        for col in columns_to_drop:
            if col in df_engineered.columns:
                df_engineered = df_engineered.drop(columns=[col])
                logger.info(f"Dropped '{col}' from features")

        return df_engineered

    def _add_viral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "gc_content" in df.columns:
            df["gc_content_squared"] = df["gc_content"] ** 2
            df["gc_content_log"] = np.log1p(
                df["gc_content"] - df["gc_content"].min() + 1
            )

        if "cpg_frequency" in df.columns:
            df["cpg_density"] = df["cpg_frequency"] / df["gc_content"].clip(lower=0.01)
            df["cpg_log"] = np.log1p(df["cpg_frequency"])

        if "gp_length" in df.columns and "genome_length" in df.columns:
            df["gp_length_ratio"] = df["gp_length"] / df["genome_length"].replace(
                0, np.nan
            )

        if "gp_hydrophobicity" in df.columns:
            df["gp_hydro_abs"] = np.abs(df["gp_hydrophobicity"])

        return df

    def _add_host_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # body mass / lifespan / range are all heavily right-skewed, so log them
        if "body_mass_kg" in df.columns:
            df["body_mass_log"] = np.log1p(df["body_mass_kg"])
            df["body_mass_sqrt"] = np.sqrt(df["body_mass_kg"] + 1)

        if "lifespan_years" in df.columns:
            df["lifespan_log"] = np.log1p(df["lifespan_years"])

        if "geographic_range_km2" in df.columns:
            df["range_log"] = np.log1p(df["geographic_range_km2"])

        return df

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot-ish indicators plus ordinal scores for contact and sociality."""
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

        if "human_contact_frequency" in df.columns:
            df["human_contact_score"] = (
                df["human_contact_frequency"]
                .map({"Low": 1, "Medium": 2, "High": 3})
                .fillna(1)
            )

        if "social_behavior" in df.columns:
            df["social_group_score"] = (
                df["social_behavior"]
                .map({"Solitary": 1, "Pairs": 2, "Groups": 3, "Colonies": 4})
                .fillna(1)
            )

        if "habitat_type" in df.columns:
            df["urban_habitat"] = (df["habitat_type"] == "Urban").astype(int)
            df["forest_habitat"] = (df["habitat_type"] == "Forest").astype(int)
            df["cave_habitat"] = (df["habitat_type"] == "Caves").astype(int)
            df["farm_habitat"] = (df["habitat_type"] == "Farm").astype(int)

        if "diet_type" in df.columns:
            df["is_frugivore"] = (df["diet_type"] == "Frugivore").astype(int)
            df["is_omnivore"] = (df["diet_type"] == "Omnivore").astype(int)
            df["is_herbivore"] = (df["diet_type"] == "Herbivore").astype(int)
            df["is_insectivore"] = (df["diet_type"] == "Insectivore").astype(int)

        return df

    def _add_bdbv_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # mask up front, all-False when the column is missing, so the risk
        # bumps below don't each need their own guard
        has_virus_col = "virus_species" in df.columns
        bdbv_mask = (
            df["virus_species"] == "BDBV"
            if has_virus_col
            else pd.Series(False, index=df.index)
        )

        if has_virus_col:
            df["is_bdbv"] = (df["virus_species"] == "BDBV").astype(int)
            df["is_ebov"] = (df["virus_species"] == "EBOV").astype(int)
            df["is_marv"] = (df["virus_species"] == "MARV").astype(int)
            df["is_sudv"] = (df["virus_species"] == "SUDV").astype(int)
            df["is_restv"] = (df["virus_species"] == "RESTV").astype(int)

        df["bdbv_risk"] = 0
        if "host_order" in df.columns:
            df.loc[bdbv_mask & (df["host_order"] == "Chiroptera"), "bdbv_risk"] += 3
            df.loc[bdbv_mask & (df["host_order"] == "Primates"), "bdbv_risk"] += 2

        if "host_family" in df.columns:
            df.loc[bdbv_mask & (df["host_family"] == "Pteropodidae"), "bdbv_risk"] += 2
            df.loc[bdbv_mask & (df["host_family"] == "Hominidae"), "bdbv_risk"] += 1

        return df

    def _add_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
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
        """At transform time, labels unseen during fit become -1."""
        df_encoded = df.copy()

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
        """The bdbv_* columns are NaN for non-BDBV rows by design and
        StandardScaler won't take NaN, so: median fill, plus a _was_missing
        flag so the model can see it was missing. Medians come from the fit
        split only."""
        df_scaled = df.copy()

        exclude_cols = ["infection_status", "high_risk"]
        numerical_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]

        if fit:
            self._impute_medians = {}
            for col in numerical_cols:
                if df_scaled[col].isnull().any():
                    df_scaled[f"{col}_was_missing"] = df_scaled[col].isnull().astype(int)
                    median = df_scaled[col].median()
                    if pd.isna(median):
                        median = 0.0
                    self._impute_medians[col] = median
                    df_scaled[col] = df_scaled[col].fillna(median)
        else:
            for col, median in getattr(self, "_impute_medians", {}).items():
                if col in df_scaled.columns:
                    missing_flag_col = f"{col}_was_missing"
                    if missing_flag_col in self.feature_columns:
                        df_scaled[missing_flag_col] = df_scaled[col].isnull().astype(int)
                    df_scaled[col] = df_scaled[col].fillna(median)
            # anything still NaN wasn't seen at fit time; 0 or the scaler blows up
            remaining_numeric = df_scaled.select_dtypes(include=[np.number]).columns
            df_scaled[remaining_numeric] = df_scaled[remaining_numeric].fillna(0.0)

        # redo this, the *_was_missing columns are new
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


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    engineer = FeatureEngineer()
    return engineer.fit_transform(df)
