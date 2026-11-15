"""Test-set metrics for a fitted model."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
    log_loss,
    brier_score_loss,
)
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(self, model):
        self.model = model
        self.metrics = {}
        self.confidence_intervals = {}

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Also stores bootstrap CIs on self.confidence_intervals."""
        logger.info("Evaluating model...")

        y_pred = self.model.predict(X_test)
        y_prob = (
            self.model.predict_proba(X_test)[:, 1]
            if hasattr(self.model, "predict_proba")
            else None
        )

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_test, y_pred),
            "kappa": cohen_kappa_score(y_test, y_pred),
        }

        if y_prob is not None:
            metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
            metrics["log_loss"] = log_loss(y_test, y_prob)
            metrics["brier_score"] = brier_score_loss(y_test, y_prob)

            # trapezoid over the PR curve, not average_precision - slightly different number
            precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
            metrics["pr_auc"] = np.trapz(precision_vals, recall_vals)

        self.metrics = metrics

        self.confidence_intervals = self._calculate_confidence_intervals(
            y_test, y_pred, y_prob
        )

        logger.info(
            f"Evaluation complete. F1: {metrics['f1']:.3f}, AUC: {metrics.get('roc_auc', 0):.3f}"
        )

        return metrics

    def _calculate_confidence_intervals(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        n_bootstrap: int = 100,
        alpha: float = 0.05,
    ) -> Dict[str, Dict[str, float]]:
        """Percentile bootstrap CI per metric; {lower, upper, mean, std}."""
        y_true_array = np.array(y_true)
        y_pred_array = np.array(y_pred)

        n_samples = len(y_true_array)
        metric_names = ["accuracy", "precision", "recall", "f1"]
        if y_prob is not None:
            metric_names.append("roc_auc")

        bootstrap_metrics = {metric: [] for metric in metric_names}

        # uses the global numpy RNG, so these CIs aren't seed-reproducible
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)

            y_true_boot = y_true_array[indices]
            y_pred_boot = y_pred_array[indices]

            try:
                acc = accuracy_score(y_true_boot, y_pred_boot)
                prec = precision_score(y_true_boot, y_pred_boot, zero_division=0)
                rec = recall_score(y_true_boot, y_pred_boot, zero_division=0)
                f1 = f1_score(y_true_boot, y_pred_boot, zero_division=0)

                bootstrap_metrics["accuracy"].append(acc)
                bootstrap_metrics["precision"].append(prec)
                bootstrap_metrics["recall"].append(rec)
                bootstrap_metrics["f1"].append(f1)

                if y_prob is not None:
                    y_prob_boot = y_prob[indices]
                    try:
                        auc_score = roc_auc_score(y_true_boot, y_prob_boot)
                        bootstrap_metrics["roc_auc"].append(auc_score)
                    except ValueError:
                        # resample came out single-class, no AUC
                        continue
            except Exception:
                continue

        confidence_intervals = {}
        for metric, values in bootstrap_metrics.items():
            if values:
                confidence_intervals[metric] = {
                    "lower": np.percentile(values, 100 * alpha / 2),
                    "upper": np.percentile(values, 100 * (1 - alpha / 2)),
                    "mean": np.mean(values),
                    "std": np.std(values),
                }
            else:
                confidence_intervals[metric] = {
                    "lower": np.nan,
                    "upper": np.nan,
                    "mean": np.nan,
                    "std": np.nan,
                }

        return confidence_intervals

    def get_confusion_matrix(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> np.ndarray:
        y_pred = self.model.predict(X_test)
        return confusion_matrix(y_test, y_pred)

    def get_classification_report(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        target_names: Optional[List[str]] = None,
    ) -> str:
        y_pred = self.model.predict(X_test)
        return classification_report(y_test, y_pred, target_names=target_names)

    def compare_models(
        self, model_dict: Dict, X_test: np.ndarray, y_test: np.ndarray
    ) -> pd.DataFrame:
        """One row per model, indexed by name."""
        results = []

        for name, model in model_dict.items():
            evaluator = ModelEvaluator(model)
            metrics = evaluator.evaluate(X_test, y_test)
            metrics["model"] = name
            results.append(metrics)

        df = pd.DataFrame(results)
        df = df.set_index("model")

        return df

    def statistical_significance_test(
        self, model1_pred: np.ndarray, model2_pred: np.ndarray, y_true: np.ndarray
    ) -> Dict:
        """McNemar on the two models' predictions; thin wrapper over StatisticalTests."""
        from src.evaluation.statistical_tests import StatisticalTests

        result = StatisticalTests.mcnemar_test(
            np.asarray(y_true), np.asarray(model1_pred), np.asarray(model2_pred)
        )
        # callers expect this key name
        result["mcnemar_chi2"] = result["chi2"]
        return result


# re-export: CrossValidator lives in cross_validation.py now, keeps old imports working
from src.evaluation.cross_validation import CrossValidator  # noqa: E402,F401
