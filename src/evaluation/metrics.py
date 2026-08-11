"""
Comprehensive model evaluation metrics.
"""

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
    """
    Comprehensive model evaluation with multiple metrics and statistical tests.
    """

    def __init__(self, model):
        self.model = model
        self.metrics = {}
        self.confidence_intervals = {}

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model on test data.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dict: Dictionary of metrics
        """
        logger.info("Evaluating model...")

        # Get predictions
        y_pred = self.model.predict(X_test)
        y_prob = (
            self.model.predict_proba(X_test)[:, 1]
            if hasattr(self.model, "predict_proba")
            else None
        )

        # Calculate all metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_test, y_pred),
            "kappa": cohen_kappa_score(y_test, y_pred),
        }

        # Probability-based metrics
        if y_prob is not None:
            metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
            metrics["log_loss"] = log_loss(y_test, y_prob)
            metrics["brier_score"] = brier_score_loss(y_test, y_prob)

            # Calculate PR-AUC
            precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
            metrics["pr_auc"] = np.trapz(precision_vals, recall_vals)

        # Store
        self.metrics = metrics

        # Calculate confidence intervals
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
        """
        Calculate confidence intervals using bootstrap.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            n_bootstrap: Number of bootstrap samples
            alpha: Significance level (default 0.05 for 95% CI)

        Returns:
            Dict with confidence intervals for each metric
        """
        # Convert to numpy arrays for consistent indexing
        y_true_array = np.array(y_true)
        y_pred_array = np.array(y_pred)

        n_samples = len(y_true_array)
        metric_names = ["accuracy", "precision", "recall", "f1"]
        if y_prob is not None:
            metric_names.append("roc_auc")

        bootstrap_metrics = {metric: [] for metric in metric_names}

        for _ in range(n_bootstrap):
            # Bootstrap sampling with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)

            # Get bootstrap samples
            y_true_boot = y_true_array[indices]
            y_pred_boot = y_pred_array[indices]

            # Calculate metrics for this bootstrap sample
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
                        # Skip if only one class in bootstrap sample
                        continue
            except Exception:
                # Skip failed bootstrap iterations
                continue

        # Calculate confidence intervals
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
        """Get confusion matrix"""
        y_pred = self.model.predict(X_test)
        return confusion_matrix(y_test, y_pred)

    def get_classification_report(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        target_names: Optional[List[str]] = None,
    ) -> str:
        """Get classification report"""
        y_pred = self.model.predict(X_test)
        return classification_report(y_test, y_pred, target_names=target_names)

    def compare_models(
        self, model_dict: Dict, X_test: np.ndarray, y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        Compare multiple models.

        Args:
            model_dict: Dictionary of {name: model}
            X_test: Test features
            y_test: Test labels

        Returns:
            pd.DataFrame: Comparison table
        """
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
        """
        Perform statistical significance test between two models.

        Args:
            model1_pred: Predictions from model 1
            model2_pred: Predictions from model 2
            y_true: True labels

        Returns:
            Dict: Test results
        """
        # McNemar's test for paired nominal data
        from sklearn.metrics import confusion_matrix

        # Create contingency table
        cm1 = confusion_matrix(y_true, model1_pred)
        cm2 = confusion_matrix(y_true, model2_pred)

        # Calculate McNemar's test
        b = cm1[0, 1]  # Model 1 false positives, Model 2 true negatives
        c = cm1[1, 0]  # Model 1 false negatives, Model 2 true positives

        # McNemar's test statistic (chi-square)
        if b + c > 0:
            chi2 = ((b - c) ** 2) / (b + c)
            p_value = 1 - stats.chi2.cdf(chi2, 1)
        else:
            chi2 = 0
            p_value = 1.0

        return {"mcnemar_chi2": chi2, "p_value": p_value, "significant": p_value < 0.05}


class CrossValidator:
    """
    Cross-validation with multiple strategies.
    """

    def __init__(self, n_folds: int = 5, random_state: int = 42):
        self.n_folds = n_folds
        self.random_state = random_state

    def cross_validate(
        self, model, X: np.ndarray, y: np.ndarray, scoring: str = "f1"
    ) -> Dict:
        """
        Perform cross-validation.

        Args:
            model: Model to evaluate
            X: Features
            y: Labels
            scoring: Scoring metric

        Returns:
            Dict: CV results
        """
        from sklearn.model_selection import cross_val_score, StratifiedKFold

        cv = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )

        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

        return {
            "mean_score": scores.mean(),
            "std_score": scores.std(),
            "scores": scores,
            "n_folds": self.n_folds,
        }
