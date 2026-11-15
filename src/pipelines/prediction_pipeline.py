"""Load a trained model and score host records."""

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
    def __init__(
        self, model_name: str = "RandomForest", model_dir: Optional[Path] = None
    ):
        self.model_name = model_name
        self.model_dir = model_dir or config.paths.MODELS_DIR

        if isinstance(self.model_dir, str):
            self.model_dir = Path(self.model_dir)

        self.model_path = self.model_dir / f"{model_name}_model.pkl"

        self.model = None
        self.feature_columns = None
        self.scaler = None
        self.label_encoders = None

        self._load_model()
        self._load_preprocessors()
        # has to be the engineer fitted during training, see _load_feature_engineer
        self.feature_engineer = self._load_feature_engineer()

    def _load_model(self):
        model_path = self.model_dir / f"{self.model_name}_model.pkl"

        loaded = False
        try:
            if model_path.exists():
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
                self.model_path = model_path
                logger.info(f"Loaded model from {model_path}")
                loaded = True

                # could be a BaseModel wrapper or a bare sklearn estimator
                if hasattr(self.model, "model"):
                    self._raw_model = self.model.model
                else:
                    self._raw_model = self.model
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")

        if not loaded:
            raise FileNotFoundError(
                f"Model not found at {model_path}.\n"
                f"Please run training first:\n"
                f"  python scripts/run_training.py"
            )

    def _load_preprocessors(self):
        preprocessor_dir = config.paths.RESULTS_DIR / "preprocessors"

        scaler_path = preprocessor_dir / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            logger.info("Loaded scaler")

        le_path = preprocessor_dir / "label_encoders.pkl"
        if le_path.exists():
            with open(le_path, "rb") as f:
                self.label_encoders = pickle.load(f)
            logger.info("Loaded label encoders")

        fc_path = preprocessor_dir / "feature_columns.pkl"
        if fc_path.exists():
            with open(fc_path, "rb") as f:
                self.feature_columns = pickle.load(f)
            logger.info(f"Loaded {len(self.feature_columns)} feature columns")

    def _load_feature_engineer(self):
        """Falls back to an unfitted engineer if the pickle is missing - predictions get shaky then."""
        fe_path = self.model_dir / "feature_engineer.pkl"

        if not fe_path.exists():
            logger.warning(f"Feature engineer not found at {fe_path}")
            logger.warning("Creating new FeatureEngineer (predictions may be incorrect)")
            from src.features.engineering import FeatureEngineer
            return FeatureEngineer()

        try:
            with open(fe_path, "rb") as f:
                engineer = pickle.load(f)
            logger.info(f"Loaded fitted feature engineer from {fe_path}")
            return engineer
        except Exception as e:
            logger.error(f"Failed to load feature engineer: {e}")
            logger.warning("Falling back to new FeatureEngineer")
            from src.features.engineering import FeatureEngineer
            return FeatureEngineer()

    def predict(self, data: Union[pd.DataFrame, Dict]) -> pd.DataFrame:
        """Input frame plus prediction, probability, prediction_label, risk_category."""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        logger.info(f"Making predictions for {len(df)} samples")

        try:
            # transform, not fit_transform - fit_transform would recompute the scaling from this one row
            df_engineered = self.feature_engineer.transform(df)
        except Exception as e:
            logger.warning(f"Feature engineering failed: {e}, using raw data")
            df_engineered = df

        # engineer emits the target column too; model never saw it
        if 'infection_status' in df_engineered.columns:
            df_engineered = df_engineered.drop(columns=['infection_status'])

        # pad any column the engineer did not produce, then put them in training order
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

        # models were fitted on bare arrays, so strip the column names
        X_pred = df_engineered.values if isinstance(df_engineered, pd.DataFrame) else df_engineered

        try:
            predictions = self.model.predict(X_pred)
            probabilities = self.model.predict_proba(X_pred)[:, 1]
        except Exception as e:
            logger.warning(f"Prediction failed with wrapper, trying raw model: {e}")
            predictions = self._raw_model.predict(X_pred)
            probabilities = self._raw_model.predict_proba(X_pred)[:, 1]

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
        """Scores names against generic placeholder traits - only useful for a rough screen."""
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


def predict_bdbv_host(
    data: Union[pd.DataFrame, Dict], model_name: str = "RandomForest"
) -> pd.DataFrame:
    pipeline = PredictionPipeline(model_name=model_name)
    return pipeline.predict(data)
