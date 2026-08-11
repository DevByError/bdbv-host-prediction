"""
Prediction pipeline for BDBV host prediction.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging

from config.config import config
from src.features.engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class PredictionPipeline:
    """
    End-to-end prediction pipeline for BDBV host susceptibility.
    """

    def __init__(
        self, model_name: str = "RandomForest", model_dir: Optional[Path] = None
    ):
        """
        Initialize prediction pipeline.

        Args:
            model_name: Name of the model to use ('RandomForest', 'XGBoost', etc.)
            model_dir: Directory containing the model files
        """
        self.model_name = model_name
        self.model_dir = model_dir or config.paths.MODELS_DIR

        # Ensure model_dir is a Path object
        if isinstance(self.model_dir, str):
            self.model_dir = Path(self.model_dir)

        # Set model path
        self.model_path = self.model_dir / f"{model_name}_model.pkl"

        self.model = None
        self.feature_columns = None
        self.scaler = None
        self.label_encoders = None

        self._load_model()
        self._load_preprocessors()
        self.feature_engineer = FeatureEngineer()

    def _load_model(self):
        """Load the trained model."""
        # Try both wrapped and raw model paths
        model_paths = [
            self.model_dir / f"{self.model_name}_model.pkl",
            self.model_dir / f"{self.model_name}_model_raw.pkl",
        ]

        # Also try alternative directories
        alt_dirs = [
            Path("results/models"),
            Path("../results/models"),
            Path("../../results/models"),
        ]

        for alt_dir in alt_dirs:
            for path in model_paths:
                alt_path = alt_dir / path.name
                if alt_path.exists():
                    model_paths.append(alt_path)

        loaded = False
        for path in model_paths:
            if path.exists():
                try:
                    with open(path, "rb") as f:
                        self.model = pickle.load(f)
                    self.model_path = path
                    logger.info(f"Loaded model from {path}")
                    loaded = True

                    # Check if it's a wrapper or raw model
                    if hasattr(self.model, "model"):
                        self._raw_model = self.model.model
                    else:
                        self._raw_model = self.model

                    break
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")
                    continue

        if not loaded:
            raise FileNotFoundError(
                f"Model not found. Please run training first:\n"
                f"  python scripts/run_training.py\n\n"
                f"Tried:\n" + "\n".join(f"  - {p}" for p in model_paths)
            )

    def _load_preprocessors(self):
        """Load preprocessors if they exist."""
        preprocessor_dir = config.paths.RESULTS_DIR / "preprocessors"

        # Try to load scaler
        scaler_path = preprocessor_dir / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            logger.info("Loaded scaler")

        # Try to load label encoders
        le_path = preprocessor_dir / "label_encoders.pkl"
        if le_path.exists():
            with open(le_path, "rb") as f:
                self.label_encoders = pickle.load(f)
            logger.info("Loaded label encoders")

        # Try to load feature columns
        fc_path = preprocessor_dir / "feature_columns.pkl"
        if fc_path.exists():
            with open(fc_path, "rb") as f:
                self.feature_columns = pickle.load(f)
            logger.info(f"Loaded {len(self.feature_columns)} feature columns")

    def predict(self, data: Union[pd.DataFrame, Dict]) -> pd.DataFrame:
        """
        Make predictions for a single host or multiple hosts.

        Args:
            data: DataFrame or dictionary with host data

        Returns:
            pd.DataFrame: Predictions with probabilities
        """
        # Convert to DataFrame if dictionary
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        logger.info(f"Making predictions for {len(df)} samples")

        # Engineer features
        try:
            df_engineered = self.feature_engineer.fit_transform(df)
        except Exception as e:
            logger.warning(f"Feature engineering failed: {e}, using raw data")
            df_engineered = df

        # Ensure all required columns exist
        if self.feature_columns is not None:
            for col in self.feature_columns:
                if col not in df_engineered.columns:
                    df_engineered[col] = 0
            df_engineered = df_engineered[self.feature_columns]
        elif hasattr(self._raw_model, "feature_names_in_"):
            required_cols = self._raw_model.feature_names_in_
            for col in required_cols:
                if col not in df_engineered.columns:
                    df_engineered[col] = 0
            df_engineered = df_engineered[required_cols]

        # Make predictions
        try:
            predictions = self.model.predict(df_engineered)
            probabilities = self.model.predict_proba(df_engineered)[:, 1]
        except Exception as e:
            logger.warning(f"Prediction failed with wrapper, trying raw model: {e}")
            predictions = self._raw_model.predict(df_engineered)
            probabilities = self._raw_model.predict_proba(df_engineered)[:, 1]

        # Create results DataFrame
        results = df.copy()
        results["prediction"] = predictions
        results["probability"] = probabilities
        results["prediction_label"] = results["prediction"].map(
            {0: "RESISTANT", 1: "SUSCEPTIBLE"}
        )
        results["risk_category"] = results["probability"].apply(self._get_risk_category)

        return results

    def predict_batch(
        self, species_list: List[str], virus: str = "BDBV"
    ) -> pd.DataFrame:
        """
        Predict for a list of species.

        Args:
            species_list: List of species names
            virus: Virus type (default: BDBV)

        Returns:
            pd.DataFrame: Predictions
        """
        # Create data dictionary for each species
        data_list = []
        for species in species_list:
            data = {
                "virus": virus,
                "host_species": species,
                "host_genus": species.split("_")[0] if "_" in species else species,
                "host_family": "Unknown",
                "host_order": "Unknown",
                "viral_genome_length": 18950,
                "gc_content": 41.0,
                "cpg_frequency": 0.043,
                "glycoprotein_length": 676,
                "gp_hydrophobicity": -0.45,
                "body_mass_kg": 1.0,
                "lifespan_years": 10,
                "geographic_range_km2": 1000000,
                "diet_type": "Omnivore",
                "social_behavior": "Groups",
                "habitat_type": "Forest",
                "human_contact_frequency": "Medium",
            }
            data_list.append(data)

        df = pd.DataFrame(data_list)
        return self.predict(df)

    def _get_risk_category(self, probability: float) -> str:
        """Get risk category based on probability."""
        if probability >= 0.8:
            return "High"
        elif probability >= 0.5:
            return "Medium"
        else:
            return "Low"

    def get_high_risk_species(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Filter for high risk species."""
        return predictions[predictions["risk_category"] == "High"]

    def save_predictions(self, predictions: pd.DataFrame, output_path: Path) -> None:
        """Save predictions to CSV."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_path, index=False)
        logger.info(f"Saved predictions to {output_path}")


# Convenience function
def predict_bdbv_host(
    data: Union[pd.DataFrame, Dict], model_name: str = "RandomForest"
) -> pd.DataFrame:
    """
    Quick prediction function.

    Args:
        data: DataFrame or dictionary with host data
        model_name: Model to use

    Returns:
        pd.DataFrame: Predictions
    """
    pipeline = PredictionPipeline(model_name=model_name)
    return pipeline.predict(data)
