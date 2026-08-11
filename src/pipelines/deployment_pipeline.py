"""
Deployment pipeline for production inference.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import logging

from src.pipelines.prediction_pipeline import PredictionPipeline
from src.features.preprocessing import FeaturePreprocessor

logger = logging.getLogger(__name__)


class DeploymentPipeline:
    """
    Production deployment pipeline with monitoring.
    """

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize deployment pipeline.

        Args:
            model_path: Path to trained model
        """
        self.model_path = model_path
        self.pipeline = PredictionPipeline(model_path)
        self.preprocessor = FeaturePreprocessor()
        self.metrics = []

    def predict(self, data: pd.DataFrame) -> Dict:
        """
        Make predictions with monitoring.

        Args:
            data: Input data

        Returns:
            Dict: Predictions with metadata
        """
        logger.info("Making production prediction...")

        # Validate input
        self._validate_input(data)

        # Make prediction
        predictions = self.pipeline.predict(data)

        # Record metrics
        self._record_prediction_metrics(predictions)

        return {
            "predictions": predictions,
            "metadata": {
                "timestamp": pd.Timestamp.now().isoformat(),
                "n_samples": len(data),
                "model_version": self._get_model_version(),
                "prediction_stats": self._get_prediction_stats(predictions),
            },
        }

    def _validate_input(self, data: pd.DataFrame):
        """Validate input data."""
        # Check required columns
        required_columns = ["virus_species", "host_species"]

        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        # Check for missing values
        missing = data[required_columns].isnull().sum()
        if missing.sum() > 0:
            raise ValueError(f"Missing values in required columns: {missing}")

    def _record_prediction_metrics(self, predictions: pd.DataFrame):
        """Record prediction metrics for monitoring."""
        metric = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "n_samples": len(predictions),
            "high_risk_count": len(predictions[predictions["risk_category"] == "High"]),
            "medium_risk_count": len(
                predictions[predictions["risk_category"] == "Medium"]
            ),
            "low_risk_count": len(predictions[predictions["risk_category"] == "Low"]),
            "mean_susceptibility": predictions["susceptibility_score"].mean(),
            "max_susceptibility": predictions["susceptibility_score"].max(),
            "min_susceptibility": predictions["susceptibility_score"].min(),
        }

        self.metrics.append(metric)

        # Save metrics
        self._save_metrics()

    def _get_prediction_stats(self, predictions: pd.DataFrame) -> Dict:
        """Get statistics from predictions."""
        return {
            "mean_score": predictions["susceptibility_score"].mean(),
            "std_score": predictions["susceptibility_score"].std(),
            "high_risk": len(predictions[predictions["risk_category"] == "High"]),
            "medium_risk": len(predictions[predictions["risk_category"] == "Medium"]),
            "low_risk": len(predictions[predictions["risk_category"] == "Low"]),
        }

    def _get_model_version(self) -> str:
        """Get model version."""
        import hashlib

        if self.model_path and self.model_path.exists():
            with open(self.model_path, "rb") as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()[:8]
        return "unknown"

    def _save_metrics(self):
        """Save metrics to file."""
        metrics_path = Path("logs/deployment_metrics.json")

        # Load existing metrics
        if metrics_path.exists():
            with open(metrics_path) as f:
                existing = json.load(f)
        else:
            existing = []

        # Append new metrics
        existing.extend(self.metrics)

        # Save
        with open(metrics_path, "w") as f:
            json.dump(existing, f, indent=2)

    def get_metrics(self) -> pd.DataFrame:
        """Get deployment metrics."""
        return pd.DataFrame(self.metrics)
