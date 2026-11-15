#!/usr/bin/env python3
"""Sanity-check a predictions CSV, optionally against ground truth."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.logging_config import get_logger
from src.evaluation.metrics import ModelEvaluator
from src.pipelines.prediction_pipeline import PredictionPipeline
import pandas as pd
import numpy as np
import argparse
from datetime import datetime

logger = get_logger(__name__)


def validate_predictions(predictions_df: pd.DataFrame) -> dict:
    """Shape and range checks on a predictions frame."""
    results = {"valid": True, "warnings": [], "errors": [], "statistics": {}}

    required_cols = ["susceptibility_score", "risk_category"]
    missing_cols = [col for col in required_cols if col not in predictions_df.columns]

    if missing_cols:
        results["valid"] = False
        results["errors"].append(f"Missing required columns: {missing_cols}")
        return results

    scores = predictions_df["susceptibility_score"]
    if (scores < 0).any() or (scores > 1).any():
        results["errors"].append("Scores outside [0,1] range")
        results["valid"] = False

    valid_categories = ["High", "Medium", "Low"]
    invalid_categories = set(predictions_df["risk_category"]) - set(valid_categories)
    if invalid_categories:
        results["errors"].append(f"Invalid risk categories: {invalid_categories}")
        results["valid"] = False

    results["statistics"] = {
        "n_samples": len(predictions_df),
        "mean_score": scores.mean(),
        "std_score": scores.std(),
        "min_score": scores.min(),
        "max_score": scores.max(),
        "high_risk": len(predictions_df[predictions_df["risk_category"] == "High"]),
        "medium_risk": len(predictions_df[predictions_df["risk_category"] == "Medium"]),
        "low_risk": len(predictions_df[predictions_df["risk_category"] == "Low"]),
    }

    # mostly-high-risk output usually means a threshold or scaling problem
    high_risk_pct = results["statistics"]["high_risk"] / len(predictions_df)
    if high_risk_pct > 0.5:
        results["warnings"].append(
            f"High proportion of high-risk predictions: {high_risk_pct:.1%}"
        )

    return results


def validate_against_ground_truth(
    predictions_df: pd.DataFrame, ground_truth_df: pd.DataFrame
) -> dict:
    """Score predictions against labelled rows, matched on species pair."""
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    merged = predictions_df.merge(
        ground_truth_df[["host_species", "virus_species", "infection_status"]],
        on=["host_species", "virus_species"],
        how="inner",
    )

    if len(merged) == 0:
        return {"valid": False, "error": "No overlapping samples found for validation"}

    # 0.5 cut, same as everywhere else
    y_pred = (merged["susceptibility_score"] >= 0.5).astype(int)
    y_true = merged["infection_status"]

    results = {
        "n_samples": len(merged),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "valid": True,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate BDBV host predictions")
    parser.add_argument(
        "--predictions", type=str, required=True, help="Path to predictions CSV file"
    )
    parser.add_argument(
        "--ground_truth", type=str, help="Path to ground truth CSV file (optional)"
    )
    parser.add_argument("--output", type=str, help="Path to output validation report")

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("BDBV HOST PREDICTION - VALIDATION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    predictions_path = Path(args.predictions)
    if not predictions_path.exists():
        logger.error(f"Predictions file not found: {predictions_path}")
        sys.exit(1)

    predictions_df = pd.read_csv(predictions_path)
    logger.info(f"Loaded {len(predictions_df)} predictions")

    logger.info("\n[1] Validating Predictions")
    validation_results = validate_predictions(predictions_df)

    if not validation_results["valid"]:
        logger.error("Validation failed!")
        for error in validation_results["errors"]:
            logger.error(f"  ✗ {error}")
        sys.exit(1)

    logger.info("✓ Predictions format valid")

    stats = validation_results["statistics"]
    logger.info("\n[2] Prediction Statistics")
    logger.info(f"  Total samples: {stats['n_samples']}")
    logger.info(f"  Mean score: {stats['mean_score']:.3f}")
    logger.info(f"  Score range: [{stats['min_score']:.3f}, {stats['max_score']:.3f}]")
    logger.info(
        f"  High risk: {stats['high_risk']} ({stats['high_risk']/stats['n_samples']:.1%})"
    )
    logger.info(
        f"  Medium risk: {stats['medium_risk']} ({stats['medium_risk']/stats['n_samples']:.1%})"
    )
    logger.info(
        f"  Low risk: {stats['low_risk']} ({stats['low_risk']/stats['n_samples']:.1%})"
    )

    if validation_results["warnings"]:
        logger.info("\n[3] Warnings")
        for warning in validation_results["warnings"]:
            logger.warning(f"  ⚠ {warning}")

    if args.ground_truth:
        logger.info("\n[4] Validating Against Ground Truth")
        ground_truth_path = Path(args.ground_truth)
        if ground_truth_path.exists():
            ground_truth_df = pd.read_csv(ground_truth_path)
            results = validate_against_ground_truth(predictions_df, ground_truth_df)

            if results.get("valid", False):
                logger.info(f"  Matched samples: {results['n_samples']}")
                logger.info(f"  Accuracy: {results['accuracy']:.3f}")
                logger.info(f"  F1 Score: {results['f1_score']:.3f}")
                logger.info(f"  Confusion Matrix:")
                logger.info(
                    f"    [[{results['confusion_matrix'][0][0]}, {results['confusion_matrix'][0][1]}]"
                )
                logger.info(
                    f"     [{results['confusion_matrix'][1][0]}, {results['confusion_matrix'][1][1]}]"
                )
            else:
                logger.warning(f"  ⚠ {results.get('error', 'Validation failed')}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "timestamp": datetime.now().isoformat(),
            "predictions_file": str(predictions_path),
            "n_predictions": len(predictions_df),
            "statistics": stats,
            "warnings": validation_results["warnings"],
            "valid": validation_results["valid"],
        }

        if "ground_truth" in locals():
            report["ground_truth_validation"] = results

        import json

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"\nValidation report saved to: {output_path}")

    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Status: {'✓ PASSED' if validation_results['valid'] else '✗ FAILED'}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
