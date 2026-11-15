#!/usr/bin/env python3
"""CLI wrapper around the prediction pipeline."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.logging_config import get_logger
from src.pipelines.prediction_pipeline import PredictionPipeline
import pandas as pd
import argparse
from datetime import datetime

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Predict BDBV host susceptibility")
    parser.add_argument("--input", type=str, help="Input CSV file path")
    parser.add_argument("--output", type=str, help="Output CSV file path")
    parser.add_argument(
        "--species", type=str, nargs="+", help="Species names to predict"
    )
    parser.add_argument("--virus", type=str, default="BDBV", help="Virus type")
    parser.add_argument(
        "--model",
        type=str,
        default="RandomForest",
        choices=["RandomForest", "LogisticRegression", "SVM", "XGBoost"],
        help="Model to use for predictions",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("BDBV HOST PREDICTION - PREDICTION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Using model: {args.model}")

    try:
        pipeline = PredictionPipeline(model_name=args.model)
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        logger.info("Available models in results/models/:")
        import os

        models_dir = Path("results/models")
        if models_dir.exists():
            for f in models_dir.glob("*.pkl"):
                logger.info(f"  - {f.stem.replace('_model', '')}")
        return

    if args.input:
        logger.info(f"Loading data from {args.input}")
        df = pd.read_csv(args.input)
        predictions = pipeline.predict(df)

    elif args.species:
        logger.info(f"Predicting for species: {args.species}")
        predictions = pipeline.predict_batch(args.species, args.virus)

    else:
        logger.info("No input provided. Using example species...")
        example_species = [
            "Rousettus_aegyptiacus",
            "Eidolon_helvum",
            "Homo_sapiens",
            "Pan_troglodytes",
            "Myotis_myotis",
        ]
        predictions = pipeline.predict_batch(example_species, "BDBV")

    output_path = args.output or "results/predictions.csv"
    pipeline.save_predictions(predictions, Path(output_path))

    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total species predicted: {len(predictions)}")

    high_risk = predictions[predictions["risk_category"] == "High"]
    medium_risk = predictions[predictions["risk_category"] == "Medium"]
    low_risk = predictions[predictions["risk_category"] == "Low"]

    logger.info(f"High risk (>80%): {len(high_risk)} species")
    logger.info(f"Medium risk (50-80%): {len(medium_risk)} species")
    logger.info(f"Low risk (<50%): {len(low_risk)} species")

    if len(high_risk) > 0:
        logger.info("\nHigh Risk Species:")
        for _, row in high_risk.head(5).iterrows():
            logger.info(
                f"  - {row.get('host_species', 'Unknown')}: {row['probability']:.2%}"
            )

    logger.info("=" * 80)
    logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
