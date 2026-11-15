"""Significance tests for comparing models."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.metrics import confusion_matrix
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class StatisticalTests:
    @staticmethod
    def mcnemar_test(
        y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray
    ) -> Dict:
        """b = only model1 right, c = only model2 right. No continuity correction."""
        b = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
        c = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))

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
        Paired comparison over bootstrap resamples of the test set.

        The resamples aren't independent draws, so the p-value is optimistic
        - treat it as a rough indication, McNemar is the stricter test here.
        """
        from sklearn.metrics import accuracy_score, f1_score

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
        """One-way ANOVA on bootstrap F1 distributions, one group per model."""
        from sklearn.metrics import f1_score

        # point estimates, reported alongside the test
        scores = []
        model_names = []

        for name, preds in y_preds.items():
            score = f1_score(y_true, preds)
            scores.append(score)
            model_names.append(name)

        # ANOVA needs repeated measurements per model, so bootstrap them
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

        # all_scores is laid out model-by-model, so slice it back into blocks
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
        """Kappa with a percentile bootstrap 95% CI, 1000 resamples."""
        from sklearn.metrics import cohen_kappa_score

        kappa = cohen_kappa_score(y_true, y_pred)

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
        """Landis & Koch bands."""
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
    def holm_bonferroni_correction(
        p_values: Dict[str, float], alpha: float = 0.05
    ) -> pd.DataFrame:
        """
        Holm step-down over a {comparison label: raw p} mapping.

        i-th smallest p is compared against alpha / (n - i + 1); result is
        sorted by p with the per-test threshold kept in the table.
        """
        if not p_values:
            raise ValueError("p_values must not be empty")

        n = len(p_values)
        sorted_items = sorted(p_values.items(), key=lambda kv: kv[1])

        rows = []
        still_significant = True
        for rank, (comparison, p) in enumerate(sorted_items, start=1):
            threshold = alpha / (n - rank + 1)
            # once one fails, everything above it fails too
            if p > threshold:
                still_significant = False
            rows.append(
                {
                    "comparison": comparison,
                    "p_value": p,
                    "rank": rank,
                    "threshold": threshold,
                    "significant": still_significant,
                }
            )

        return pd.DataFrame(rows)

    @staticmethod
    def deLong_roc_test(
        y_true: np.ndarray, y_prob1: np.ndarray, y_prob2: np.ndarray
    ) -> Dict:
        """
        AUC comparison between two models.

        Not actually DeLong's covariance estimator - this bootstraps the AUC
        difference and takes a normal z. Name kept for the callers.
        """
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
