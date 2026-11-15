"""End-to-end training run. scripts/run_training.py is just a CLI wrapper around this."""

import json
import logging
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from config.config import config
from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.evaluation.ablation import run_ablation_study
from src.evaluation.cross_validation import CrossValidator
from src.evaluation.fairness import disparity_summary, subgroup_performance
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.statistical_tests import StatisticalTests
from src.evaluation.uncertainty import UncertaintyEstimator
from src.explainability.biological_context import BiologicalContext
from src.explainability.feature_importance import FeatureImportance
from src.explainability.shap_analysis import SHAPAnalyzer
from src.features.engineering import FeatureEngineer
from src.models.classical import (
    LogisticRegressionModel,
    RandomForestModel,
    SVMModel,
    XGBoostModel,
)
from src.utils.reproducibility import set_seed
from src.visualization.publication_figures import FigureGenerator

logger = logging.getLogger(__name__)


class TrainingPipeline:

    def __init__(
        self,
        config_override: Optional[Dict] = None,
        run_ablation: bool = True,
        run_fairness: bool = True,
        run_calibration: bool = True,
        n_bootstrap_ci: int = 1000,
        random_state: Optional[int] = None,
    ):
        # config_override isn't wired up yet, just stored
        self.config_override = config_override or {}
        self.run_ablation = run_ablation
        self.run_fairness = run_fairness
        self.run_calibration = run_calibration
        self.n_bootstrap_ci = n_bootstrap_ci
        self.random_state = (
            random_state if random_state is not None else config.training.random_state
        )

        self.results: Dict = {}
        self.best_model = None
        self.best_model_name: Optional[str] = None
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.report: Dict = {}

    def run(self) -> Dict:
        """Returns {model_name: {metrics..., 'model': fitted}}. Everything else lands in self.report."""
        run_started = datetime.now()
        logger.info("=" * 80)
        logger.info("BDBV HOST PREDICTION - TRAINING PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Started: {run_started.strftime('%Y-%m-%d %H:%M:%S')}")

        # seed before anything draws a random number
        seeded = set_seed(self.random_state)
        logger.info(f"Reproducibility: seed={self.random_state}, seeded_libraries={seeded}")

        logger.info("\n[Step 1] Loading Data")
        loader = DataLoader(validate=True, random_state=self.random_state)
        df = loader.load_data()
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")

        validator = DataValidator()
        validation_results = validator.validate(df)
        if not validation_results["valid"]:
            raise ValueError(
                f"Data validation failed with errors: {validation_results['errors']}. "
                f"Fix the input data before training (see logs for detail)."
            )

        # split raw rows first - fitting encoders/scalers on everything and
        # splitting afterwards leaks test statistics into the transform
        logger.info("\n[Step 2] Splitting Data (before feature fitting)")
        y_raw = df["infection_status"]
        from sklearn.model_selection import train_test_split

        train_df, test_df = train_test_split(
            df,
            test_size=config.training.test_size,
            random_state=self.random_state,
            stratify=y_raw,
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")

        # fit on train, transform test - never fit_transform the test frame
        logger.info("\n[Step 3] Feature Engineering (fit on train only)")
        self.feature_engineer = FeatureEngineer()
        train_engineered = self.feature_engineer.fit_transform(train_df)
        test_engineered = self.feature_engineer.transform(test_df)

        X_train = train_engineered.drop(columns=["infection_status"])
        y_train = train_engineered["infection_status"]
        X_test = test_engineered.drop(columns=["infection_status"])
        y_test = test_engineered["infection_status"]
        logger.info(f"Engineered {X_train.shape[1]} features")

        X_train_bal, y_train_bal = self._maybe_apply_smote(X_train, y_train)

        logger.info("\n[Step 5] Training Models")
        models = self._create_models()

        predictions_by_model: Dict[str, np.ndarray] = {}
        probabilities_by_model: Dict[str, np.ndarray] = {}

        for name, model in models.items():
            logger.info(f"  Training {name}...")
            model.fit(X_train_bal, y_train_bal)

            evaluator = ModelEvaluator(model)
            metrics = evaluator.evaluate(X_test, y_test)
            self.results[name] = {
                "model": model,
                **metrics,
                "confidence_intervals": evaluator.confidence_intervals,
            }
            predictions_by_model[name] = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                probabilities_by_model[name] = model.predict_proba(X_test)[:, 1]

            logger.info(
                f"    F1: {metrics['f1']:.3f}, AUC: {metrics.get('roc_auc', float('nan')):.3f}"
            )

        logger.info("\n[Step 6] Selecting Best Model")
        self.best_model_name = max(self.results, key=lambda k: self.results[k]["f1"])
        self.best_model = self.results[self.best_model_name]["model"]
        logger.info(f"Best model: {self.best_model_name}")
        logger.info(f"  F1-Score: {self.results[self.best_model_name]['f1']:.3f}")

        logger.info("\n[Step 7] Cross-Validation")
        cv = CrossValidator(n_folds=config.training.n_folds, random_state=self.random_state)
        cv_results = cv.cross_validate(self.best_model, X_train_bal, y_train_bal, scoring="f1")
        logger.info(
            f"  CV F1: {cv_results['mean_score']:.3f} "
            f"(95% CI [{cv_results['ci_lower']:.3f}, {cv_results['ci_upper']:.3f}])"
        )

        logger.info("\n[Step 8] Statistical Significance Testing")
        significance_table = self._pairwise_significance(predictions_by_model, y_test.values)

        logger.info("\n[Step 9] Explainability")
        explainability_report = self._run_explainability(X_train_bal, X_test, y_test)

        calibration_report = {}
        if self.run_calibration:
            logger.info("\n[Step 10] Uncertainty & Calibration")
            calibration_report = self._run_calibration(X_test, y_test)

        ablation_df = None
        if self.run_ablation:
            logger.info("\n[Step 11] Feature-Group Ablation Study")
            ablation_df = run_ablation_study(
                lambda: RandomForestModel(
                    {"n_estimators": 100, "random_state": self.random_state}
                ),
                X_train_bal,
                y_train_bal,
                X_test,
                y_test,
                scoring_metric="f1",
            )

        # group on the raw taxonomy columns, not the encoded matrix
        fairness_report = {}
        if self.run_fairness:
            logger.info("\n[Step 12] Taxonomic Fairness Audit")
            fairness_report = self._run_fairness(test_df, predictions_by_model, probabilities_by_model)

        logger.info("\n[Step 13] Saving Models & Preprocessors")
        self._save_models()
        self._save_feature_engineer()

        logger.info("\n[Step 14] Generating Figures")
        try:
            fig_gen = FigureGenerator(self.results, self.best_model_name)
            fig_gen.generate_all_figures(X_train_bal, X_test, y_train_bal, y_test)
        except Exception as e:
            logger.warning(f"Figure generation failed (non-fatal): {e}")

        logger.info("\n[Step 15] Saving Comprehensive Report")
        self.report = self._build_report(
            run_started=run_started,
            seeded=seeded,
            validation_results=validation_results,
            X_train=X_train_bal,
            X_test=X_test,
            cv_results=cv_results,
            significance_table=significance_table,
            explainability_report=explainability_report,
            calibration_report=calibration_report,
            ablation_df=ablation_df,
            fairness_report=fairness_report,
        )
        self._save_report(self.report)
        self._save_legacy_outputs(X_train_bal, X_test, y_train_bal, y_test)

        elapsed = (datetime.now() - run_started).total_seconds()
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Best Model: {self.best_model_name}")
        logger.info(f"F1-Score: {self.results[self.best_model_name]['f1']:.3f}")
        logger.info(
            f"ROC-AUC: {self.results[self.best_model_name].get('roc_auc', float('nan')):.3f}"
        )
        logger.info(f"Elapsed: {elapsed:.1f}s")
        logger.info(f"Results saved to: {config.paths.RESULTS_DIR}")
        logger.info("=" * 80)

        return self.results

    def _create_models(self) -> Dict:
        models = {
            "RandomForest": RandomForestModel(
                {
                    "n_estimators": 200,
                    "max_depth": None,
                    "class_weight": "balanced",
                    "random_state": self.random_state,
                }
            ),
            "LogisticRegression": LogisticRegressionModel(
                {"max_iter": 1000, "class_weight": "balanced", "random_state": self.random_state}
            ),
            "SVM": SVMModel(
                {"probability": True, "class_weight": "balanced", "random_state": self.random_state}
            ),
        }

        try:
            models["XGBoost"] = XGBoostModel(
                {
                    "n_estimators": 200,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "random_state": self.random_state,
                }
            )
        except ImportError:
            logger.warning("XGBoost not available, skipping")

        return models

    def _maybe_apply_smote(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Oversample the training split only. Returns the input untouched if SMOTE is off or infeasible."""
        if not getattr(config.training, "use_smote", False):
            return X_train, y_train

        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            logger.warning(
                "config.training.use_smote=True but imbalanced-learn is not "
                "installed; skipping SMOTE. Install with: pip install imbalanced-learn"
            )
            return X_train, y_train

        class_counts = y_train.value_counts()
        min_class_count = class_counts.min()
        k_neighbors = min(
            getattr(config.training, "smote_k_neighbors", 3), min_class_count - 1
        )

        if k_neighbors < 1:
            logger.warning(
                f"Smallest class has only {min_class_count} training samples; "
                f"not enough for SMOTE (needs >= 2). Skipping."
            )
            return X_train, y_train

        logger.info(
            f"Applying SMOTE (k_neighbors={k_neighbors}); class counts before: "
            f"{class_counts.to_dict()}"
        )
        smote = SMOTE(k_neighbors=k_neighbors, random_state=self.random_state)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        logger.info(f"Class counts after SMOTE: {pd.Series(y_res).value_counts().to_dict()}")

        return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res, name=y_train.name)

    def _pairwise_significance(
        self, predictions_by_model: Dict[str, np.ndarray], y_test: np.ndarray
    ) -> pd.DataFrame:
        """McNemar per model pair, Holm-corrected. One row per pair: b, c, chi2, p_value, significance flags."""
        model_names = list(predictions_by_model.keys())
        pairs = [
            (a, b) for i, a in enumerate(model_names) for b in model_names[i + 1 :]
        ]
        if not pairs:
            return pd.DataFrame()

        raw_results = {}
        p_values = {}
        for a, b in pairs:
            res = StatisticalTests.mcnemar_test(
                y_test, predictions_by_model[a], predictions_by_model[b]
            )
            label = f"{a}_vs_{b}"
            raw_results[label] = res
            p_values[label] = res["p_value"]

        corrected = StatisticalTests.holm_bonferroni_correction(p_values)
        corrected = corrected.set_index("comparison")

        rows = []
        for label, res in raw_results.items():
            rows.append(
                {
                    "comparison": label,
                    "b": int(res["b"]),
                    "c": int(res["c"]),
                    "chi2": res["chi2"],
                    "p_value": res["p_value"],
                    "significant_uncorrected": bool(res["significant"]),
                    "significant_holm_bonferroni": bool(
                        corrected.loc[label, "significant"]
                    ),
                }
            )
            logger.info(
                f"  {label}: chi2={res['chi2']:.3f}, p={res['p_value']:.4f}, "
                f"significant (Holm-corrected)={corrected.loc[label, 'significant']}"
            )

        return pd.DataFrame(rows)

    def _run_explainability(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict:
        report: Dict = {}

        # SHAPAnalyzer unwraps the BaseModel wrapper itself
        try:
            shap_analyzer = SHAPAnalyzer(self.best_model, X_train.columns.tolist())
            shap_analyzer.fit_explainer(X_train)
            fig_path = config.paths.FIGURES_DIR / "shap_summary.png"
            shap_analyzer.plot_summary(max_display=20, output_path=str(fig_path))
            report["shap"] = {"status": "ok", "figure": str(fig_path)}
        except Exception as e:
            logger.warning(f"SHAP analysis skipped: {e}")
            report["shap"] = {"status": "skipped", "reason": str(e)}

        try:
            fi = FeatureImportance(
                getattr(self.best_model, "model", self.best_model),
                X_train.columns.tolist(),
            )
            combined = fi.get_combined_importance(X_test.values, y_test.values)
            report["feature_importance"] = combined.head(20).to_dict(orient="index")
        except Exception as e:
            logger.warning(f"Feature importance analysis skipped: {e}")
            report["feature_importance"] = {}

        try:
            from src.explainability.lime_analysis import LIMEExplainer

            lime_explainer = LIMEExplainer(self.best_model, X_train.columns.tolist())
            lime_explainer.fit(X_train.values)

            n_examples = min(3, len(X_test))
            example_idx = np.linspace(0, len(X_test) - 1, n_examples, dtype=int)
            lime_examples = []
            for idx in example_idx:
                explanation = lime_explainer.explain_instance(X_test.values[idx])
                proba = explanation["prediction_proba"]
                lime_examples.append(
                    {
                        "test_index": int(idx),
                        "true_label": int(y_test.values[idx]),
                        "predicted_probability_positive_class": float(proba[1])
                        if len(proba) > 1
                        else float(proba[0]),
                        "top_features": explanation["features"].head(5).to_dict(orient="records"),
                    }
                )
            report["lime_examples"] = lime_examples
        except Exception as e:
            logger.warning(f"LIME analysis skipped: {e}")
            report["lime_examples"] = []

        try:
            top_features = list(report.get("feature_importance", {}).keys())[:15]
            report["biological_context"] = {
                f: BiologicalContext.map_feature(f) for f in top_features
            }
        except Exception as e:
            logger.warning(f"Biological context mapping skipped: {e}")
            report["biological_context"] = {}

        return report

    def _run_calibration(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        if not hasattr(self.best_model, "predict_proba"):
            return {"status": "skipped", "reason": "best model has no predict_proba"}

        y_prob = self.best_model.predict_proba(X_test)[:, 1]
        y_true = y_test.values

        ue = UncertaintyEstimator()
        curve = ue.calibration_curve(y_true, y_prob, n_bins=10)
        ece = ue.expected_calibration_error(y_true, y_prob, n_bins=10)
        brier = ue.brier_score_decomposition(y_true, y_prob, n_bins=10)

        report = {
            "expected_calibration_error": ece,
            "brier_decomposition": brier,
            "reliability_curve": {
                "bin_centers": curve["bin_centers"].tolist(),
                "bin_true_frac": curve["bin_true_frac"].tolist(),
                "bin_pred_mean": curve["bin_pred_mean"].tolist(),
                "bin_counts": curve["bin_counts"].tolist(),
            },
        }

        if hasattr(getattr(self.best_model, "model", None), "estimators_"):
            eu = ue.ensemble_uncertainty(self.best_model, X_test)
            if eu:
                report["ensemble_uncertainty_mean_entropy"] = float(np.mean(eu["entropy"]))
                report["ensemble_uncertainty_mean_std"] = float(np.mean(eu["std"]))

        logger.info(f"  ECE: {ece:.4f}, Brier: {brier['brier_score']:.4f}")
        return report

    def _run_fairness(
        self,
        test_df: pd.DataFrame,
        predictions_by_model: Dict[str, np.ndarray],
        probabilities_by_model: Dict[str, np.ndarray],
    ) -> Dict:
        report: Dict = {}
        y_true = test_df["infection_status"].values
        y_pred = predictions_by_model[self.best_model_name]
        y_prob = probabilities_by_model.get(self.best_model_name)

        for group_col in ("host_order", "host_family"):
            if group_col not in test_df.columns:
                continue
            try:
                subgroup_df = subgroup_performance(
                    y_true, y_pred, test_df[group_col], y_prob=y_prob, min_group_size=3
                )
                report[group_col] = {
                    "subgroups": subgroup_df.to_dict(orient="records"),
                }
                try:
                    report[group_col]["disparity"] = disparity_summary(subgroup_df, metric="f1")
                except ValueError as e:
                    report[group_col]["disparity"] = {"status": "skipped", "reason": str(e)}
                logger.info(
                    f"  {group_col}: {len(subgroup_df)} subgroups, "
                    f"f1 range={report[group_col].get('disparity', {}).get('range', float('nan'))}"
                )
            except Exception as e:
                logger.warning(f"Fairness audit for '{group_col}' skipped: {e}")

        return report

    def _save_models(self):
        """Each model under its own name, plus a copy of the winner as best_model.pkl."""
        for name, result in self.results.items():
            path = config.paths.MODELS_DIR / f"{name}_model.pkl"
            result["model"].save(path)

        best_path = config.paths.MODELS_DIR / "best_model.pkl"
        self.best_model.save(best_path)
        logger.info(f"Saved {len(self.results)} models; best -> {best_path}")

    def _save_feature_engineer(self):
        """Prediction needs this exact fitted object and must call .transform() on it."""
        path = config.paths.RESULTS_DIR / "preprocessors" / "feature_engineer.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.feature_engineer, path)
        logger.info(f"Saved fitted FeatureEngineer to {path}")

    def _build_report(self, **kwargs) -> Dict:
        run_started = kwargs["run_started"]
        model_summary = {
            name: {
                k: v
                for k, v in res.items()
                if k not in ("model", "confidence_intervals")
            }
            | {"confidence_intervals": res.get("confidence_intervals", {})}
            for name, res in self.results.items()
        }

        report = {
            "run_metadata": {
                "timestamp": run_started.isoformat(),
                "seed": self.random_state,
                "seeded_libraries": kwargs["seeded"],
                "python_version": sys.version,
                "platform": platform.platform(),
                "package_versions": self._get_package_versions(),
            },
            "data_validation": kwargs["validation_results"],
            "model_metrics": model_summary,
            "best_model": self.best_model_name,
            "n_train_samples": int(len(kwargs["X_train"])),
            "n_test_samples": int(len(kwargs["X_test"])),
            "n_features": int(kwargs["X_train"].shape[1]),
            "cross_validation": {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in kwargs["cv_results"].items()
                if k not in ("predictions", "y_true", "probabilities", "fold_indices")
            },
            "statistical_significance": kwargs["significance_table"].to_dict(orient="records")
            if not kwargs["significance_table"].empty
            else [],
            "explainability": kwargs["explainability_report"],
            "calibration": kwargs["calibration_report"],
            "ablation": kwargs["ablation_df"].to_dict(orient="records")
            if kwargs["ablation_df"] is not None
            else None,
            "fairness": kwargs["fairness_report"],
        }
        return report

    @staticmethod
    def _get_package_versions() -> Dict[str, str]:
        versions = {}
        for pkg in ["numpy", "pandas", "sklearn", "scipy", "xgboost", "shap"]:
            try:
                mod = __import__(pkg)
                versions[pkg] = getattr(mod, "__version__", "unknown")
            except ImportError:
                versions[pkg] = "not installed"
        return versions

    def _save_report(self, report: Dict):
        reports_dir = config.paths.REPORTS_DIR
        reports_dir.mkdir(parents=True, exist_ok=True)

        json_path = reports_dir / "training_report.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Saved JSON report to {json_path}")

        md_path = reports_dir / "training_report.md"
        with open(md_path, "w") as f:
            f.write(self._render_markdown_report(report))
        logger.info(f"Saved Markdown report to {md_path}")

    def _render_markdown_report(self, report: Dict) -> str:
        lines = ["# BDBV Host Prediction — Training Report", ""]
        meta = report["run_metadata"]
        lines += [
            f"- **Run timestamp:** {meta['timestamp']}",
            f"- **Random seed:** {meta['seed']} (seeded libraries: {meta['seeded_libraries']})",
            f"- **Train / test samples:** {report['n_train_samples']} / {report['n_test_samples']}",
            f"- **Features:** {report['n_features']}",
            f"- **Best model:** {report['best_model']}",
            "",
            "## Model Performance (test set, with 95% bootstrap CI)",
            "",
            "| Model | F1 | Accuracy | ROC-AUC | Precision | Recall |",
            "|---|---|---|---|---|---|",
        ]
        for name, m in report["model_metrics"].items():
            lines.append(
                f"| {name} | {m.get('f1', float('nan')):.3f} | "
                f"{m.get('accuracy', float('nan')):.3f} | "
                f"{m.get('roc_auc', float('nan')):.3f} | "
                f"{m.get('precision', float('nan')):.3f} | "
                f"{m.get('recall', float('nan')):.3f} |"
            )

        lines += ["", "## Cross-Validation (best model)", ""]
        cv = report["cross_validation"]
        lines.append(
            f"- {cv.get('scoring', 'f1')}: {cv.get('mean_score', float('nan')):.3f} "
            f"± {cv.get('std_score', float('nan')):.3f} "
            f"(95% CI [{cv.get('ci_lower', float('nan')):.3f}, "
            f"{cv.get('ci_upper', float('nan')):.3f}], "
            f"{cv.get('n_folds', '?')} folds)"
        )

        lines += ["", "## Pairwise Statistical Significance (McNemar, Holm-Bonferroni corrected)", ""]
        if report["statistical_significance"]:
            lines.append("| Comparison | p-value | Significant (corrected) |")
            lines.append("|---|---|---|")
            for row in report["statistical_significance"]:
                lines.append(
                    f"| {row['comparison']} | {row['p_value']:.4f} | "
                    f"{row['significant_holm_bonferroni']} |"
                )
        else:
            lines.append("_Only one model trained; no pairwise comparisons available._")

        if report.get("calibration"):
            lines += ["", "## Calibration", ""]
            cal = report["calibration"]
            if "expected_calibration_error" in cal:
                lines.append(f"- Expected Calibration Error: {cal['expected_calibration_error']:.4f}")
                lines.append(f"- Brier score: {cal['brier_decomposition']['brier_score']:.4f}")

        if report.get("ablation"):
            lines += ["", "## Feature-Group Ablation (F1)", ""]
            lines.append("| Condition | # Features | F1 | Δ vs full |")
            lines.append("|---|---|---|---|")
            for row in report["ablation"]:
                lines.append(
                    f"| {row['condition']} | {row['n_features_used']} | "
                    f"{row['score']:.3f} | {row['score_delta_vs_full']:+.3f} |"
                )

        if report.get("fairness"):
            lines += ["", "## Fairness / Subgroup Audit", ""]
            for group_col, data in report["fairness"].items():
                disp = data.get("disparity", {})
                if "range" in disp:
                    lines.append(
                        f"- **{group_col}**: F1 range {disp['range']:.3f} "
                        f"(worst: {disp['worst_group']}, best: {disp['best_group']}, "
                        f"{disp['n_groups_evaluated']} groups evaluated)"
                    )

        lines += [
            "",
            "---",
            "*This report was generated automatically by `TrainingPipeline`. "
            "See `training_report.json` for full machine-readable detail "
            "(per-feature importance, LIME examples, per-fold CV scores, "
            "full subgroup tables).*",
        ]
        return "\n".join(lines)

    def _save_legacy_outputs(self, X_train, X_test, y_train, y_test):
        """model_metrics.csv + metadata.json, kept for notebooks that read them directly."""
        metrics_df = pd.DataFrame(
            {
                name: {k: v for k, v in res.items() if k not in ("model", "confidence_intervals")}
                for name, res in self.results.items()
            }
        ).T
        metrics_df.to_csv(config.paths.RESULTS_DIR / "model_metrics.csv")

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "best_model": self.best_model_name,
            "n_features": X_train.shape[1],
            "n_train": len(X_train),
            "n_test": len(X_test),
            "class_balance_train": {
                "negative": int((y_train == 0).sum()),
                "positive": int((y_train == 1).sum()),
            },
            "class_balance_test": {
                "negative": int((y_test == 0).sum()),
                "positive": int((y_test == 1).sum()),
            },
            "metrics": {
                k: v
                for k, v in self.results[self.best_model_name].items()
                if k not in ("model", "confidence_intervals")
            },
        }

        with open(config.paths.RESULTS_DIR / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
