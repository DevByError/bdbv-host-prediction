"""
End-to-end training pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging
from datetime import datetime

from config.config import config
from src.data.loader import DataLoader
from src.data.splitter import DataSplitter
from src.features.engineering import FeatureEngineer
from src.models.classical import (
    RandomForestModel,
    XGBoostModel,
    LogisticRegressionModel,
)
from src.evaluation.metrics import ModelEvaluator, CrossValidator
from src.explainability.shap_analysis import SHAPAnalyzer
from src.visualization.publication_figures import FigureGenerator

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Complete training pipeline with all steps.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize training pipeline.

        Args:
            config: Optional configuration override
        """
        self.config = config or {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None

    def run(self) -> Dict:
        """
        Run the complete training pipeline.

        Returns:
            Dict: Training results
        """
        logger.info("=" * 80)
        logger.info("BDBV HOST PREDICTION - TRAINING PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Step 1: Load data
        logger.info("\n[Step 1] Loading Data")
        loader = DataLoader()
        df = loader.load_data()
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")

        # Step 2: Feature engineering
        logger.info("\n[Step 2] Feature Engineering")
        engineer = FeatureEngineer()
        df_engineered = engineer.fit_transform(df)
        logger.info(f"Created {len(df_engineered.columns)} features")

        # Step 3: Split data
        logger.info("\n[Step 3] Splitting Data")
        X = df_engineered.drop(columns=["infection_status"])
        y = df_engineered["infection_status"]

        splitter = DataSplitter(
            random_state=config.training.random_state,
            test_size=config.training.test_size,
        )
        X_train, X_test, y_train, y_test = splitter.stratified_split(X, y)
        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

        # Step 4: Train models
        logger.info("\n[Step 4] Training Models")
        models = self._create_models()

        for name, model in models.items():
            logger.info(f"  Training {name}...")
            model.fit(X_train, y_train)

            # Evaluate
            evaluator = ModelEvaluator(model)
            metrics = evaluator.evaluate(X_test, y_test)
            self.results[name] = {"model": model, **metrics}
            logger.info(
                f"    F1: {metrics['f1']:.3f}, AUC: {metrics.get('roc_auc', 0):.3f}"
            )

        # Step 5: Select best model
        logger.info("\n[Step 5] Selecting Best Model")
        self.best_model_name = max(
            self.results.keys(), key=lambda x: self.results[x]["f1"]
        )
        self.best_model = self.results[self.best_model_name]["model"]
        logger.info(f"Best model: {self.best_model_name}")
        logger.info(f"  F1-Score: {self.results[self.best_model_name]['f1']:.3f}")

        # Step 6: Cross-validation
        logger.info("\n[Step 6] Cross-Validation")
        cv = CrossValidator(n_folds=config.training.n_folds)
        cv_results = cv.cross_validate(self.best_model, X_train.values, y_train.values)
        logger.info(
            f"  CV F1: {cv_results['mean_score']:.3f} ± {cv_results['std_score']:.3f}"
        )

        # Step 7: SHAP analysis
        logger.info("\n[Step 7] SHAP Analysis")
        shap_analyzer = SHAPAnalyzer(self.best_model, X_train.columns.tolist())
        shap_analyzer.fit_explainer(X_train)
        shap_analyzer.plot_summary(max_display=20)

        # Step 8: Save results
        logger.info("\n[Step 8] Saving Results")
        self._save_results(X_train, X_test, y_train, y_test)

        # Step 9: Generate figures
        logger.info("\n[Step 9] Generating Figures")
        fig_gen = FigureGenerator(self.results, self.best_model_name)
        fig_gen.generate_all_figures(X_train, X_test, y_train, y_test)

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Best Model: {self.best_model_name}")
        logger.info(f"F1-Score: {self.results[self.best_model_name]['f1']:.3f}")
        logger.info(
            f"ROC-AUC: {self.results[self.best_model_name].get('roc_auc', 0):.3f}"
        )
        logger.info(f"Results saved to: {config.paths.RESULTS_DIR}")
        logger.info("=" * 80)

        return self.results

    def _create_models(self) -> Dict:
        """Create dictionary of models."""
        models = {}

        # Random Forest
        models["RandomForest"] = RandomForestModel(
            {
                "n_estimators": 100,
                "max_depth": None,
                "class_weight": "balanced",
                "random_state": 42,
            }
        )

        # XGBoost
        try:
            models["XGBoost"] = XGBoostModel(
                {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "random_state": 42,
                }
            )
        except ImportError:
            logger.warning("XGBoost not available, skipping")

        # Logistic Regression
        models["LogisticRegression"] = LogisticRegressionModel(
            {"max_iter": 1000, "class_weight": "balanced", "random_state": 42}
        )

        return models

    def _save_results(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ):
        """Save all results to disk."""
        # Save metrics
        metrics_df = pd.DataFrame(self.results).T
        metrics_df.to_csv(config.paths.RESULTS_DIR / "model_metrics.csv")

        # Save best model
        best_model_path = config.paths.MODELS_DIR / "best_model.pkl"
        self.best_model.save(best_model_path)

        # Save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "best_model": self.best_model_name,
            "n_features": X_train.shape[1],
            "n_train": len(X_train),
            "n_test": len(X_test),
            "class_balance_train": {
                "negative": sum(y_train == 0),
                "positive": sum(y_train == 1),
            },
            "class_balance_test": {
                "negative": sum(y_test == 0),
                "positive": sum(y_test == 1),
            },
            "metrics": self.results[self.best_model_name],
        }

        import json

        with open(config.paths.RESULTS_DIR / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
