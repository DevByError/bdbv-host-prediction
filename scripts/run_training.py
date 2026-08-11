#!/usr/bin/env python3
"""
Main training script for BDBV host prediction.
Runs end-to-end training pipeline.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# IMPORTANT: Set matplotlib backend BEFORE importing pyplot
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend

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
import pickle  # Make sure pickle is imported

# Check if XGBoost is available
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

    # 1. Load data
    logger.info("\n[1] Loading Data")
    loader = DataLoader()
    df = loader.load_data()
    logger.info(f"Loaded {len(df)} samples")

    # 2. Feature engineering
    logger.info("\n[2] Feature Engineering")
    engineer = FeatureEngineer()
    df_engineered = engineer.fit_transform(df)
    logger.info(f"Created {len(df_engineered.columns)} features")

    # Separate features and target
    X = df_engineered.drop(columns=["infection_status"])
    y = df_engineered["infection_status"]

    # 3. Split data
    logger.info("\n[3] Splitting Data")
    splitter = DataSplitter(
        random_state=config.training.random_state, test_size=config.training.test_size
    )

    X_train, X_test, y_train, y_test = splitter.stratified_split(X, y)
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # 4. Train models
    logger.info("\n[4] Training Models")

    # Create models dictionary
    models = {
        "RandomForest": RandomForestModel(),
        "LogisticRegression": LogisticRegressionModel(),
        "SVM": SVMModel(),
    }

    # Add XGBoost if available
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBoostModel()
        logger.info("  XGBoost is available and will be trained")
    else:
        logger.info("  XGBoost is not available - skipping")

    results = {}
    for name, model in models.items():
        logger.info(f"\n  Training {name}...")
        model.fit(X_train, y_train)

        # Evaluate
        evaluator = ModelEvaluator(model)
        metrics = evaluator.evaluate(X_test, y_test)

        # Store both metrics AND the model object
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

    # 5. Select best model
    logger.info("\n[5] Selecting Best Model")
    best_model_name = max(results, key=lambda x: results[x]["f1"])
    best_model = models[best_model_name]
    logger.info(f"Best model: {best_model_name}")
    logger.info(f"  F1-Score: {results[best_model_name]['f1']:.3f}")
    logger.info(f"  AUC: {results[best_model_name]['roc_auc']:.3f}")

    # 6. SHAP analysis (if available)
    try:
        logger.info("\n[6] Generating SHAP Explanations")
        shap_analyzer = SHAPAnalyzer(best_model.model, X_train.columns.tolist())
        shap_analyzer.fit_explainer(X_train)
        shap_analyzer.plot_summary(
            output_path=str(PROJECT_ROOT / "results" / "figures" / "shap_summary.png")
        )
        shap_analyzer.plot_bar(
            output_path=str(PROJECT_ROOT / "results" / "figures" / "shap_bar.png")
        )
    except Exception as e:
        logger.warning(f"SHAP analysis skipped: {e}")

    # 7. Save models - FIXED with proper error handling
    logger.info("\n[7] Saving Models")

    # Ensure models directory exists
    models_dir = config.paths.MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        save_path = models_dir / f"{name}_model.pkl"

        try:
            # Use pickle directly
            with open(save_path, "wb") as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"  Saved {name} to {save_path}")
        except Exception as e:
            logger.error(f"  Failed to save {name}: {e}")

    # Also save the best model separately
    best_path = models_dir / "best_model.pkl"
    try:
        with open(best_path, "wb") as f:
            pickle.dump(best_model, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"  Saved best model to {best_path}")
    except Exception as e:
        logger.error(f"  Failed to save best model: {e}")

    # 8. Generate publication figures
    try:
        logger.info("\n[8] Generating Publication Figures")
        figure_gen = FigureGenerator(results, best_model_name)
        figure_gen.generate_all_figures(X_train, X_test, y_train, y_test)
    except Exception as e:
        logger.warning(f"Figure generation skipped: {e}")

    # 9. Save results
    logger.info("\n[9] Saving Results")
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

    # 10. Summary
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
