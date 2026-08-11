"""
Statistical tests for model comparison and validation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.metrics import confusion_matrix
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class StatisticalTests:
    """
    Statistical tests for model evaluation and comparison.
    """

    @staticmethod
    def mcnemar_test(
        y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray
    ) -> Dict:
        """
        McNemar's test for paired nominal data.

        Args:
            y_true: True labels
            y_pred1: Predictions from model 1
            y_pred2: Predictions from model 2

        Returns:
            Dict: Test results
        """
        # Create contingency table
        # b: model1 correct, model2 incorrect
        # c: model1 incorrect, model2 correct
        b = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
        c = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))

        # McNemar's test statistic
        if b + c > 0:
            chi2 = ((b - c) ** 2) / (b + c)
            p_value = 1 - stats.chi2.cdf(chi2, 1)
        else:
            chi2 = 0
            p_value = 1.0

        return {
            "b": b,
            "c": c,
            "chi2": chi2,
            "p_value": p_value,
            "significant": p_value < 0.05,
        }

    @staticmethod
    def paired_t_test(
        y_true: np.ndarray,
        y_pred1: np.ndarray,
        y_pred2: np.ndarray,
        metric: str = "accuracy",
    ) -> Dict:
        """
        Paired t-test for comparing two models.

        Args:
            y_true: True labels
            y_pred1: Predictions from model 1
            y_pred2: Predictions from model 2
            metric: Metric to compare ('accuracy' or 'f1')

        Returns:
            Dict: Test results
        """
        from sklearn.metrics import accuracy_score, f1_score

        # Calculate metrics per sample (or per fold)
        # Here we use bootstrap for paired comparison
        n_bootstrap = 1000
        n_samples = len(y_true)

        scores1 = []
        scores2 = []

        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)

            if metric == "accuracy":
                s1 = accuracy_score(y_true[indices], y_pred1[indices])
                s2 = accuracy_score(y_true[indices], y_pred2[indices])
            else:
                s1 = f1_score(y_true[indices], y_pred1[indices])
                s2 = f1_score(y_true[indices], y_pred2[indices])

            scores1.append(s1)
            scores2.append(s2)

        # Paired t-test
        differences = np.array(scores1) - np.array(scores2)
        t_stat, p_value = stats.ttest_1samp(differences, 0)

        return {
            "mean_diff": np.mean(differences),
            "std_diff": np.std(differences),
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "model1_mean": np.mean(scores1),
            "model2_mean": np.mean(scores2),
        }

    @staticmethod
    def compare_models_anova(
        y_true: np.ndarray, y_preds: Dict[str, np.ndarray]
    ) -> Dict:
        """
        One-way ANOVA to compare multiple models.

        Args:
            y_true: True labels
            y_preds: Dictionary of {model_name: predictions}

        Returns:
            Dict: ANOVA results
        """
        from sklearn.metrics import f1_score

        # Calculate F1 per model
        scores = []
        model_names = []

        for name, preds in y_preds.items():
            score = f1_score(y_true, preds)
            scores.append(score)
            model_names.append(name)

        # Perform ANOVA
        # For ANOVA, we need multiple measurements per model
        # Use bootstrap to get distribution
        n_bootstrap = 100
        n_samples = len(y_true)

        all_scores = []
        all_names = []

        for name, preds in y_preds.items():
            boot_scores = []
            for _ in range(n_bootstrap):
                indices = np.random.choice(n_samples, n_samples, replace=True)
                boot_scores.append(f1_score(y_true[indices], preds[indices]))
            all_scores.extend(boot_scores)
            all_names.extend([name] * n_bootstrap)

        # ANOVA
        groups = [
            all_scores[i : i + n_bootstrap]
            for i in range(0, len(all_scores), n_bootstrap)
        ]
        f_stat, p_value = stats.f_oneway(*groups)

        return {
            "f_statistic": f_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "model_scores": dict(zip(model_names, scores)),
        }

    @staticmethod
    def cohen_kappa_confidence(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate Cohen's Kappa with confidence interval.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dict: Kappa with CI
        """
        from sklearn.metrics import cohen_kappa_score

        # Calculate kappa
        kappa = cohen_kappa_score(y_true, y_pred)

        # Bootstrap for CI
        n_bootstrap = 1000
        n_samples = len(y_true)
        kappas = []

        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            kappas.append(cohen_kappa_score(y_true[indices], y_pred[indices]))

        ci_lower = np.percentile(kappas, 2.5)
        ci_upper = np.percentile(kappas, 97.5)

        return {
            "kappa": kappa,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "std": np.std(kappas),
            "interpretation": StatisticalTests._interpret_kappa(kappa),
        }

    @staticmethod
    def _interpret_kappa(kappa: float) -> str:
        """Interpret Cohen's Kappa value."""
        if kappa < 0:
            return "Poor agreement (worse than chance)"
        elif kappa < 0.2:
            return "Slight agreement"
        elif kappa < 0.4:
            return "Fair agreement"
        elif kappa < 0.6:
            return "Moderate agreement"
        elif kappa < 0.8:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"

    @staticmethod
    def deLong_roc_test(
        y_true: np.ndarray, y_prob1: np.ndarray, y_prob2: np.ndarray
    ) -> Dict:
        """
        DeLong's test for comparing ROC curves.

        Args:
            y_true: True labels
            y_prob1: Probabilities from model 1
            y_prob2: Probabilities from model 2

        Returns:
            Dict: Test results
        """
        # Implementation of DeLong's test
        # Simplified version using bootstrap
        n_bootstrap = 1000
        n_samples = len(y_true)

        auc1 = []
        auc2 = []

        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)

            from sklearn.metrics import roc_auc_score

            auc1.append(roc_auc_score(y_true[indices], y_prob1[indices]))
            auc2.append(roc_auc_score(y_true[indices], y_prob2[indices]))

        differences = np.array(auc1) - np.array(auc2)

        z_stat = np.mean(differences) / np.std(differences)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        return {
            "z_statistic": z_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "auc1_mean": np.mean(auc1),
            "auc2_mean": np.mean(auc2),
            "diff_mean": np.mean(differences),
        }
