#!/usr/bin/env python3
"""End-to-end training run: load, engineer, fit, evaluate, save."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Agg backend, we never show these interactively, and it has to be set
# before anything pulls in pyplot
import matplotlib

matplotlib.use("Agg")

from config.config import config
from config.logging_config import get_logger
from src.data.loader import DataLoader
from src.data.splitter import DataSplitter
from src.features.engineering import FeatureEngineer
from src.models.classical import (
    RandomForestModel,
    LogisticRegressionModel,
    SVMModel,
)
from src.evaluation.metrics import ModelEvaluator
from src.explainability.shap_analysis import SHAPAnalyzer
from src.visualization.publication_figures import FigureGenerator

import pandas as pd
import numpy as np
from datetime import datetime
import pickle

try:
    from src.models.classical import XGBoostModel

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: XGBoost not available. Skipping XGBoost model.")

logger = get_logger(__name__)


def main():
    """Run training pipeline"""
    logger.info("=" * 80)
    logger.info("BDBV HOST SUSCEPTIBILITY PREDICTION - TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    logger.info("\n[1] Loading Data")
    loader = DataLoader()
    df = loader.load_data()
    logger.info(f"Loaded {len(df)} samples")

    logger.info("\n[2] Feature Engineering")
    engineer = FeatureEngineer()
    df_engineered = engineer.fit_transform(df)
    logger.info(f"Created {len(df_engineered.columns)} features")

    X = df_engineered.drop(columns=["infection_status"])
    y = df_engineered["infection_status"]

    logger.info("\n[3] Splitting Data")
    splitter = DataSplitter(
        random_state=config.training.random_state, test_size=config.training.test_size
    )

    X_train, X_test, y_train, y_test = splitter.stratified_split(X, y)
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    logger.info("\n[4] Training Models")

    models = {
        "RandomForest": RandomForestModel(),
        "LogisticRegression": LogisticRegressionModel(),
        "SVM": SVMModel(),
    }

    if XGB_AVAILABLE:
        models["XGBoost"] = XGBoostModel()
        logger.info("  XGBoost is available and will be trained")
    else:
        logger.info("  XGBoost is not available - skipping")

    results = {}
    for name, model in models.items():
        logger.info(f"\n  Training {name}...")
        model.fit(X_train, y_train)

        evaluator = ModelEvaluator(model)
        metrics = evaluator.evaluate(X_test, y_test)

        # keep the fitted model alongside the numbers, the figures need it
        results[name] = {
            "model": model,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics.get("roc_auc", 0),
        }

        logger.info(f"    Accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"    F1-Score: {metrics['f1']:.3f}")
        logger.info(f"    AUC: {metrics.get('roc_auc', 0):.3f}")

    logger.info("\n[5] Selecting Best Model")
    best_model_name = max(results, key=lambda x: results[x]["f1"])
    best_model = models[best_model_name]
    logger.info(f"Best model: {best_model_name}")
    logger.info(f"  F1-Score: {results[best_model_name]['f1']:.3f}")
    logger.info(f"  AUC: {results[best_model_name]['roc_auc']:.3f}")

    # skip SHAP on the full set, the kernel explainer takes hours
    logger.info("\n[6] SHAP Analysis")
    if len(X_train) > 500:
        logger.warning(f"  Skipping SHAP (dataset too large: {len(X_train)} samples)")
        logger.warning(f"  Reason: KernelExplainer on 10k records ≈ 8-12 hours")
        logger.info(f"  Use: python scripts/shap_analysis.py (for subset only)")
    else:
        try:
            logger.info("  Generating SHAP Explanations...")
            shap_analyzer = SHAPAnalyzer(best_model.model, X_train.columns.tolist())
            shap_analyzer.fit_explainer(X_train)
            shap_analyzer.plot_summary(
                output_path=str(PROJECT_ROOT / "results" / "figures" / "shap_summary.png")
            )
            shap_analyzer.plot_bar(
                output_path=str(PROJECT_ROOT / "results" / "figures" / "shap_bar.png")
            )
        except Exception as e:
            logger.warning(f"  SHAP analysis failed: {e}")

    logger.info("\n[7] Saving Models")

    models_dir = config.paths.MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        save_path = models_dir / f"{name}_model.pkl"

        try:
            with open(save_path, "wb") as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"  Saved {name} to {save_path}")
        except Exception as e:
            logger.error(f"  Failed to save {name}: {e}")

    best_path = models_dir / "best_model.pkl"
    try:
        with open(best_path, "wb") as f:
            pickle.dump(best_model, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"  Saved best model to {best_path}")
    except Exception as e:
        logger.error(f"  Failed to save best model: {e}")

    try:
        logger.info("\n[8] Generating Publication Figures")
        figure_gen = FigureGenerator(results, best_model_name)
        figure_gen.generate_all_figures(X_train, X_test, y_train, y_test)
    except Exception as e:
        logger.warning(f"Figure generation skipped: {e}")

    logger.info("\n[9] Saving Feature Engineer & Results")

    # prediction needs the fitted engineer, not just the model
    import pickle as pkl
    fe_path = config.paths.MODELS_DIR / "feature_engineer.pkl"
    with open(fe_path, 'wb') as f:
        pkl.dump(engineer, f, protocol=pkl.HIGHEST_PROTOCOL)
    logger.info(f"  Saved feature engineer to {fe_path}")

    results_df = pd.DataFrame(
        [
            {
                "Model": name,
                "Accuracy": res["accuracy"],
                "Precision": res["precision"],
                "Recall": res["recall"],
                "F1_Score": res["f1"],
                "ROC_AUC": res["roc_auc"],
            }
            for name, res in results.items()
        ]
    )
    results_df.to_csv(config.paths.RESULTS_DIR / "model_performance.csv", index=False)
    logger.info(
        f"Results saved to {config.paths.RESULTS_DIR / 'model_performance.csv'}"
    )

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Best Model: {best_model_name}")
    logger.info(f"F1-Score: {results[best_model_name]['f1']:.3f}")
    logger.info(f"ROC-AUC: {results[best_model_name]['roc_auc']:.3f}")
    logger.info(f"Results saved to: {config.paths.RESULTS_DIR}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
