"""Prediction uncertainty and calibration diagnostics."""

import numpy as np
from typing import Dict, Optional, Tuple, List, Union
import logging

logger = logging.getLogger(__name__)


class UncertaintyEstimator:
    """Tree variance, bootstrap spread, conformal sets, and calibration."""

    def __init__(self, n_estimators: int = 100):
        self.n_estimators = n_estimators
        self.models = []

    def ensemble_uncertainty(self, model, X: np.ndarray) -> Dict:
        """
        Spread across individual trees - no refitting needed.

        Empty dict if the model has no estimators_. Arrays are per-row of X.
        """
        raw_model = getattr(model, "model", model)
        X_arr = X.values if hasattr(X, "values") else np.asarray(X)

        if not hasattr(raw_model, "estimators_"):
            logger.warning("Model doesn't support ensemble uncertainty (no estimators_)")
            return {}

        tree_preds = np.array([tree.predict(X_arr) for tree in raw_model.estimators_])
        mean = np.mean(tree_preds, axis=0)
        variance = np.var(tree_preds, axis=0)

        proba_mean = np.mean(
            [tree.predict_proba(X_arr) for tree in raw_model.estimators_], axis=0
        )
        entropy = -np.sum(proba_mean * np.log(proba_mean + 1e-10), axis=1)

        return {
            "mean": mean,
            "variance": variance,
            "std": np.sqrt(variance),
            "entropy": entropy,
            "confidence": 1 - entropy / np.log(2),  # log(2) normalizes binary entropy
        }

    def conformal_prediction(
        self,
        model,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        X_test: np.ndarray,
        y_test: Optional[np.ndarray] = None,
        alpha: float = 0.1,
    ) -> Dict:
        """
        Split conformal prediction sets. alpha=0.1 targets 90% coverage.

        X_cal must be disjoint from both the training and test data or the
        coverage guarantee is void. Empirical coverage is returned only when
        y_test is passed. prediction_sets holds class indices, and can come
        back empty or with both classes.
        """
        if not hasattr(model, "predict_proba"):
            raise ValueError("conformal_prediction requires a model with predict_proba")

        X_cal_arr = X_cal.values if hasattr(X_cal, "values") else np.asarray(X_cal)
        y_cal_arr = np.asarray(y_cal).astype(int)
        X_test_arr = X_test.values if hasattr(X_test, "values") else np.asarray(X_test)

        proba_cal = model.predict_proba(X_cal_arr)
        scores = 1 - proba_cal[np.arange(len(y_cal_arr)), y_cal_arr]

        n_cal = len(scores)
        # finite-sample correction, not just (1 - alpha)
        q_level = min(1.0, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal)
        q_hat = np.quantile(scores, q_level)

        test_proba = model.predict_proba(X_test_arr)
        prediction_sets = [
            np.where(test_proba[i] >= 1 - q_hat)[0] for i in range(len(test_proba))
        ]

        max_proba = np.max(test_proba, axis=1)
        confidence = 1 - max_proba

        result = {
            "prediction_sets": prediction_sets,
            "confidence": confidence,
            "q_hat": float(q_hat),
            "alpha": alpha,
            "target_coverage": 1 - alpha,
        }

        if y_test is not None:
            y_test_arr = np.asarray(y_test).astype(int)
            result["coverage"] = float(
                np.mean(
                    [
                        1 if y in pred_set else 0
                        for y, pred_set in zip(y_test_arr, prediction_sets)
                    ]
                )
            )

        return result

    def bootstrap_uncertainty(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        X_eval: Optional[np.ndarray] = None,
        n_bootstrap: int = 100,
        random_state: int = 42,
    ) -> Dict:
        """
        Refit on bootstrap resamples and look at the spread of predictions.

        Evaluated on X_eval, defaulting to X itself. Percentile CIs are 95%,
        over probabilities where the model has them.
        """
        X_arr = X.values if hasattr(X, "values") else np.asarray(X)
        y_arr = np.asarray(y)
        X_eval_arr = (
            X_eval.values if hasattr(X_eval, "values") else np.asarray(X_eval)
        ) if X_eval is not None else X_arr

        rng = np.random.default_rng(random_state)
        n_samples = len(X_arr)
        predictions = []

        for _ in range(n_bootstrap):
            indices = rng.integers(0, n_samples, size=n_samples)
            X_boot, y_boot = X_arr[indices], y_arr[indices]

            if len(np.unique(y_boot)) < 2:
                continue  # single-class resample, skip rather than let fit blow up

            model_clone = model.clone() if hasattr(model, "clone") else _sk_clone(model)
            model_clone.fit(X_boot, y_boot)

            if hasattr(model_clone, "predict_proba"):
                y_pred = model_clone.predict_proba(X_eval_arr)[:, 1]
            else:
                y_pred = model_clone.predict(X_eval_arr).astype(float)
            predictions.append(y_pred)

        if not predictions:
            raise RuntimeError(
                "All bootstrap resamples were degenerate (single class); "
                "increase n_bootstrap or check class balance"
            )

        predictions = np.array(predictions)

        return {
            "mean": np.mean(predictions, axis=0),
            "variance": np.var(predictions, axis=0),
            "std": np.std(predictions, axis=0),
            "ci_lower": np.percentile(predictions, 2.5, axis=0),
            "ci_upper": np.percentile(predictions, 97.5, axis=0),
            "n_successful_bootstraps": len(predictions),
        }

    def prediction_interval(self, y_pred: np.ndarray, y_std: np.ndarray) -> Dict:
        """Normal 95% interval. Not clipped to [0, 1], so it can run off the ends."""
        z_score = 1.96

        lower = y_pred - z_score * y_std
        upper = y_pred + z_score * y_std

        return {
            "lower": lower,
            "upper": upper,
            "width": upper - lower,
            "relative_width": (upper - lower) / (np.abs(y_pred) + 1e-10),
        }

    @staticmethod
    def calibration_curve(
        y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
    ) -> Dict:
        """
        Reliability diagram over n_bins equal-width bins of [0, 1].

        Empty bins are dropped, so the returned arrays can be shorter than
        n_bins.
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_idx = np.clip(np.digitize(y_prob, bin_edges[1:-1]), 0, n_bins - 1)

        bin_centers, bin_true_frac, bin_pred_mean, bin_counts = [], [], [], []
        for b in range(n_bins):
            mask = bin_idx == b
            count = mask.sum()
            if count == 0:
                continue
            bin_centers.append((bin_edges[b] + bin_edges[b + 1]) / 2)
            bin_true_frac.append(y_true[mask].mean())
            bin_pred_mean.append(y_prob[mask].mean())
            bin_counts.append(int(count))

        return {
            "bin_centers": np.array(bin_centers),
            "bin_true_frac": np.array(bin_true_frac),
            "bin_pred_mean": np.array(bin_pred_mean),
            "bin_counts": np.array(bin_counts),
        }

    @staticmethod
    def expected_calibration_error(
        y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
    ) -> float:
        """Count-weighted mean gap between predicted and observed rate per bin. Lower is better."""
        curve = UncertaintyEstimator.calibration_curve(y_true, y_prob, n_bins)
        n_total = curve["bin_counts"].sum()
        if n_total == 0:
            return float("nan")
        gaps = np.abs(curve["bin_pred_mean"] - curve["bin_true_frac"])
        return float(np.sum(curve["bin_counts"] * gaps) / n_total)

    @staticmethod
    def brier_score_decomposition(
        y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
    ) -> Dict:
        """
        Murphy decomposition: brier = reliability - resolution + uncertainty.

        The identity only holds up to binning error, so the three terms
        won't reproduce brier_score exactly.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float)
        n = len(y_true)
        base_rate = y_true.mean()

        brier = float(np.mean((y_prob - y_true) ** 2))

        curve = UncertaintyEstimator.calibration_curve(y_true, y_prob, n_bins)
        counts = curve["bin_counts"]
        reliability = float(
            np.sum(counts * (curve["bin_pred_mean"] - curve["bin_true_frac"]) ** 2) / n
        )
        resolution = float(
            np.sum(counts * (curve["bin_true_frac"] - base_rate) ** 2) / n
        )
        uncertainty = float(base_rate * (1 - base_rate))

        return {
            "brier_score": brier,
            "reliability": reliability,
            "resolution": resolution,
            "uncertainty": uncertainty,
        }


def _sk_clone(model):
    from sklearn.base import clone

    return clone(model)
