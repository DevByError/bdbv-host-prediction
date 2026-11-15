#!/usr/bin/env python3
"""SHAP on a subset of the test set.

The kernel explainer is far too slow for the full data, so pick a few
hundred rows at most.

    python scripts/shap_analysis.py --samples 100 --model SVM
"""

import sys
from pathlib import Path
import argparse
import logging
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.logging_config import get_logger
from src.data.loader import DataLoader
from src.data.splitter import DataSplitter
from src.features.engineering import FeatureEngineer
from src.explainability.shap_analysis import SHAPAnalyzer
from config.config import config
import pickle

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="SHAP Analysis on Subset")
    parser.add_argument('--samples', type=int, default=100, help='Number of samples to analyze')
    parser.add_argument('--model', type=str, default='SVM', choices=['SVM', 'RandomForest', 'XGBoost', 'LogisticRegression'])
    parser.add_argument('--stratified', action='store_true', help='Stratified sampling by label')

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("BDBV SHAP EXPLAINABILITY ANALYSIS - SUBSET MODE")
    logger.info("="*80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Samples: {args.samples}")
    logger.info(f"Stratified: {args.stratified}")

    logger.info("\n[1] Loading Data")
    loader = DataLoader()
    df = loader.load_data()
    logger.info(f"Loaded {len(df)} total records")

    logger.info("\n[2] Feature Engineering")
    engineer = FeatureEngineer()
    df_engineered = engineer.fit_transform(df)

    logger.info("\n[3] Splitting Data")
    splitter = DataSplitter()
    X_train, X_test, y_train, y_test = splitter.stratified_split(
        df_engineered.drop(columns=['infection_status']),
        df_engineered['infection_status']
    )

    logger.info(f"\n[4] Sampling {args.samples} Records for Analysis")
    if args.stratified:
        pos_idx = np.where(y_test == 1)[0]
        neg_idx = np.where(y_test == 0)[0]

        n_pos = min(args.samples // 2, len(pos_idx))
        n_neg = min(args.samples // 2, len(neg_idx))

        sample_idx = np.concatenate([
            np.random.choice(pos_idx, size=n_pos, replace=False),
            np.random.choice(neg_idx, size=n_neg, replace=False)
        ])
        logger.info(f"  Positive: {n_pos}, Negative: {n_neg}")
    else:
        sample_idx = np.random.choice(len(X_test), size=min(args.samples, len(X_test)), replace=False)

    X_sample = X_test.iloc[sample_idx]
    logger.info(f"  Selected {len(X_sample)} samples for analysis")

    logger.info(f"\n[5] Loading {args.model} Model")
    model_path = config.paths.MODELS_DIR / f"{args.model}_model.pkl"

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info(f"Loaded model from {model_path}")

    # unwrap, SHAP wants the bare estimator
    if hasattr(model, 'model'):
        raw_model = model.model
    else:
        raw_model = model

    logger.info(f"\n[6] Computing SHAP Values (This may take a few minutes...)")
    logger.info(f"  Using KernelExplainer on {len(X_sample)} samples")
    logger.info(f"  Estimated time: {len(X_sample) // 2} - {len(X_sample)} seconds")

    try:
        shap_analyzer = SHAPAnalyzer(raw_model, X_test.columns.tolist())
        shap_analyzer.fit_explainer(X_sample)

        logger.info("  ✅ SHAP values computed")

        logger.info(f"\n[7] Generating Plots")
        output_dir = config.paths.RESULTS_DIR / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("  Generating summary plot...")
        shap_analyzer.plot_summary(
            output_path=str(output_dir / "shap_summary_subset.png")
        )
        logger.info(f"    ✅ Saved to {output_dir / 'shap_summary_subset.png'}")

        logger.info("  Generating bar plot...")
        shap_analyzer.plot_bar(
            output_path=str(output_dir / "shap_bar_subset.png")
        )
        logger.info(f"    ✅ Saved to {output_dir / 'shap_bar_subset.png'}")

    except Exception as e:
        logger.error(f"SHAP analysis failed: {e}")
        logger.error("This is normal for large datasets - SHAP is computationally expensive")
        return

    logger.info("\n" + "="*80)
    logger.info("SHAP ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"\nKey Findings:")
    logger.info(f"  Analyzed: {len(X_sample)} representative samples")
    logger.info(f"  Method: KernelExplainer (model-agnostic)")
    logger.info(f"  Output: figures/{output_dir.name}/*")
    logger.info(f"\nNote: This is a SUBSET analysis for interpretability")
    logger.info(f"Full 10k record analysis would take 8-12+ hours and is not recommended")
    logger.info("="*80)

if __name__ == "__main__":
    main()
