"""Prediction by species name: database lookup when we know the species, nearest-neighbour interpolation when we do not."""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import logging
from typing import Dict, List, Tuple, Optional

from config.config import config
from src.database.species_database import get_species_database
from src.features.engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class AdvancedPredictionPipeline:
    def __init__(self, model_name: str = "SVM", model_dir: Optional[Path] = None):
        self.model_name = model_name
        self.model_dir = model_dir or config.paths.MODELS_DIR
        self.model = None
        self._raw_model = None

        self.species_db = get_species_database()
        logger.info(f"Species database loaded: {self.species_db.get_statistics()}")

        # must be the engineer that was fitted during training
        self.feature_engineer = self._load_feature_engineer()

        self._load_model()

        logger.info(f"Advanced Prediction Pipeline initialized with {model_name} model")

    def _load_feature_engineer(self):
        """Unfitted fallback if the pickle is missing; predictions are unreliable in that case."""
        fe_path = self.model_dir / "feature_engineer.pkl"

        if not fe_path.exists():
            logger.warning(f"Feature engineer not found at {fe_path}")
            logger.warning("Creating new FeatureEngineer (predictions may be incorrect)")
            return FeatureEngineer()

        try:
            with open(fe_path, "rb") as f:
                engineer = pickle.load(f)
            logger.info(f"Loaded fitted feature engineer from {fe_path}")
            return engineer
        except Exception as e:
            logger.error(f"Failed to load feature engineer: {e}")
            logger.warning("Falling back to new FeatureEngineer")
            return FeatureEngineer()

    def _load_model(self):
        model_path = self.model_dir / f"{self.model_name}_model.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}.\n"
                f"Please run training first:\n"
                f"  python scripts/run_training.py"
            )

        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info(f"Loaded model from {model_path}")

            # may be a BaseModel wrapper or a bare estimator
            if hasattr(self.model, "model"):
                self._raw_model = self.model.model
            else:
                self._raw_model = self.model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict_for_species(
        self,
        species_name: str,
        virus: str = "BDBV"
    ) -> Dict:
        """Keys: prediction, probability, confidence (0-1), source, reference_species, data_quality, metadata."""
        logger.info(f"Predicting for species: {species_name}")

        if self.species_db.species_exists(species_name):
            logger.info(f"Species found in database: {species_name}")
            return self._predict_known_species(species_name, virus)
        else:
            logger.info(f"Species not in database, using similarity matching: {species_name}")
            return self._predict_unknown_species(species_name, virus)

    def _predict_known_species(self, species_name: str, virus: str) -> Dict:
        species_features = self.species_db.get_species_info(species_name)

        features_dict = self._species_features_to_prediction_input(species_features, species_name, virus)

        prediction, probability = self._make_prediction(features_dict)

        similar_species = self.species_db.find_similar_species(species_name, n_neighbors=3)

        return {
            'species': species_name,
            'prediction': prediction,
            'probability': probability,
            'confidence': 0.95,  # flat value for anything we hold real records for
            'source': 'database',
            'reference_species': [s[0] for s in similar_species],
            'data_quality': 'high',
            'sample_count': species_features.get('sample_count', 1),
            'virus': virus,
            'metadata': {
                'order': species_features.get('order', 'Unknown'),
                'family': species_features.get('family', 'Unknown'),
                'infection_rate_historical': species_features.get('infection_rate', 0.5),
            }
        }

    def _predict_unknown_species(self, species_name: str, virus: str) -> Dict:
        # 20 neighbours; fewer made the interpolated traits jumpy
        predicted_features, confidence = self.species_db.predict_features_for_unknown_species(
            species_name,
            n_neighbors=20
        )

        logger.info(f"Interpolated features with confidence: {confidence:.2%}")

        features_dict = self._interpolated_features_to_prediction_input(
            predicted_features,
            species_name,
            virus
        )

        prediction, probability = self._make_prediction(features_dict)

        # only the closest 5 get reported back
        similar_species = self.species_db.find_similar_species(species_name, n_neighbors=5)

        return {
            'species': species_name,
            'prediction': prediction,
            'probability': probability,
            'confidence': confidence,
            'source': 'interpolated',
            'reference_species': [s[0] for s in similar_species],
            'data_quality': 'medium' if confidence > 0.7 else 'low',
            'virus': virus,
            'metadata': {
                'interpolation_confidence': confidence,
                'similar_species': [s[0] for s in similar_species[:3]],
                'estimated_order': predicted_features.get('order', 'Unknown'),
                'estimated_family': predicted_features.get('family', 'Unknown'),
            }
        }

    def _species_features_to_prediction_input(
        self,
        species_features: Dict,
        species_name: str,
        virus: str
    ) -> pd.DataFrame:
        data = {
            "virus": virus,
            "host_species": species_name,
            "host_genus": species_name.split("_")[0],
            "host_family": species_features.get('family', 'Unknown'),
            "host_order": species_features.get('order', 'Unknown'),
            "viral_genome_length": species_features.get('viral_genome_length', 18950),
            "gc_content": species_features.get('gc_content', 41),
            "cpg_frequency": species_features.get('cpg_frequency', 0.038),
            "glycoprotein_length": species_features.get('glycoprotein_length', 676),
            "gp_hydrophobicity": species_features.get('gp_hydrophobicity', -0.45),
            "body_mass_kg": species_features.get('body_mass_kg', 1.0),
            "lifespan_years": species_features.get('lifespan_years', 10),
            "geographic_range_km2": species_features.get('geographic_range_km2', 1e6),
            "diet_type": species_features.get('diet_type', 'Omnivore'),
            "social_behavior": species_features.get('social_behavior', 'Groups'),
            "habitat_type": species_features.get('habitat_type', 'Forest'),
            "human_contact_frequency": species_features.get('human_contact_frequency', 'Medium'),
        }

        return pd.DataFrame([data])

    def _interpolated_features_to_prediction_input(
        self,
        predicted_features: Dict,
        species_name: str,
        virus: str
    ) -> pd.DataFrame:
        data = {
            "virus": virus,
            "host_species": species_name,
            "host_genus": species_name.split("_")[0],
            "host_family": predicted_features.get('family', 'Unknown'),
            "host_order": predicted_features.get('order', 'Unknown'),
            "viral_genome_length": predicted_features.get('viral_genome_length', 18950),
            "gc_content": predicted_features.get('gc_content', 41),
            "cpg_frequency": predicted_features.get('cpg_frequency', 0.038),
            "glycoprotein_length": predicted_features.get('glycoprotein_length', 676),
            "gp_hydrophobicity": predicted_features.get('gp_hydrophobicity', -0.45),
            "body_mass_kg": predicted_features.get('body_mass_kg', 1.0),
            "lifespan_years": predicted_features.get('lifespan_years', 10),
            "geographic_range_km2": predicted_features.get('geographic_range_km2', 1e6),
            "diet_type": predicted_features.get('diet_type', 'Omnivore'),
            "social_behavior": predicted_features.get('social_behavior', 'Groups'),
            "habitat_type": predicted_features.get('habitat_type', 'Forest'),
            "human_contact_frequency": predicted_features.get('human_contact_frequency', 'Medium'),
        }

        return pd.DataFrame([data])

    def _make_prediction(self, df: pd.DataFrame) -> Tuple[int, float]:
        try:
            # transform, not fit_transform - a single row would rescale everything
            df_engineered = self.feature_engineer.transform(df)

            # engineer hands back the target column; drop it before the model sees it
            if 'infection_status' in df_engineered.columns:
                df_engineered = df_engineered.drop(columns=['infection_status'])

            if hasattr(self._raw_model, "feature_names_in_"):
                required_cols = self._raw_model.feature_names_in_
                for col in required_cols:
                    if col not in df_engineered.columns:
                        df_engineered[col] = 0
                df_engineered = df_engineered[required_cols]

            # fitted on bare arrays, so drop the column names
            X_pred = df_engineered.values if isinstance(df_engineered, pd.DataFrame) else df_engineered

            predictions = self._raw_model.predict(X_pred)
            probabilities = self._raw_model.predict_proba(X_pred)[:, 1]

            return int(predictions[0]), float(probabilities[0])

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

    def predict_batch(
        self,
        species_list: List[str],
        virus: str = "BDBV"
    ) -> pd.DataFrame:
        """One row per species; a failure becomes a row with prediction ERROR instead of raising."""
        results = []

        for species in species_list:
            try:
                prediction_result = self.predict_for_species(species, virus)

                result = {
                    'host_species': species,
                    'prediction': 'SUSCEPTIBLE' if prediction_result['prediction'] == 1 else 'RESISTANT',
                    'probability': prediction_result['probability'],
                    'probability_percent': f"{prediction_result['probability']*100:.1f}%",
                    'confidence': prediction_result['confidence'],
                    'risk_category': self._get_risk_category(prediction_result['probability']),
                    'source': prediction_result['source'],
                    'data_quality': prediction_result['data_quality'],
                }
                results.append(result)

            except Exception as e:
                logger.error(f"Error predicting for {species}: {e}")
                results.append({
                    'host_species': species,
                    'prediction': 'ERROR',
                    'probability': None,
                    'confidence': 0,
                    'error': str(e)
                })

        return pd.DataFrame(results)

    def _get_risk_category(self, probability: float) -> str:
        if probability >= 0.8:
            return "High"
        elif probability >= 0.5:
            return "Medium"
        else:
            return "Low"

    def get_species_list(self) -> List[str]:
        return self.species_db.get_species_list()

    def search_species(self, query: str) -> List[str]:
        query = query.lower()
        all_species = self.get_species_list()
        return [s for s in all_species if query in s.lower()]


def get_advanced_pipeline(model_name: str = "SVM") -> AdvancedPredictionPipeline:
    return AdvancedPredictionPipeline(model_name=model_name)
