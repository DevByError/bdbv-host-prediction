#!/usr/bin/env python3
"""
Generate all publication figures from saved results.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import config
from config.logging_config import get_logger
from src.visualization.publication_figures import FigureGenerator
import pandas as pd
import json

logger = get_logger(__name__)


def main():
    """Generate all figures from saved results."""
    logger.info("=" * 80)
    logger.info("BDBV HOST PREDICTION - FIGURE GENERATION")
    logger.info("=" * 80)

    # Load results
    results_path = config.paths.RESULTS_DIR / "model_metrics.csv"
    metadata_path = config.paths.RESULTS_DIR / "metadata.json"

    if not results_path.exists():
        logger.error(f"Results not found: {results_path}")
        logger.error("Please run training first: python scripts/run_training.py")
        return

    # Load metrics
    metrics_df = pd.read_csv(results_path, index_col=0)

    # Load metadata
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Convert to results dict
    results = {}
    for model_name in metrics_df.index:
        results[model_name] = {
            "model": None,  # We don't need actual model for figures
            "accuracy": metrics_df.loc[model_name, "accuracy"],
            "precision": metrics_df.loc[model_name, "precision"],
            "recall": metrics_df.loc[model_name, "recall"],
            "f1": metrics_df.loc[model_name, "f1"],
        }
        if "roc_auc" in metrics_df.columns:
            results[model_name]["roc_auc"] = metrics_df.loc[model_name, "roc_auc"]

    # Generate figures
    best_model_name = metadata.get("best_model", "RandomForest")
    fig_gen = FigureGenerator(results, best_model_name)

    # Generate each figure individually
    logger.info("Generating Figure 1: Model Performance")
    fig_gen.fig1_model_performance()

    # Note: Other figures require actual data
    logger.info("To generate all figures, use TrainingPipeline instead.")
    logger.info("Run: python scripts/run_training.py")

    logger.info("\n" + "=" * 80)
    logger.info("FIGURE GENERATION COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
