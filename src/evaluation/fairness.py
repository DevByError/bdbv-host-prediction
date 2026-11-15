"""Per-subgroup performance and disparity, e.g. by host order."""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def subgroup_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: pd.Series,
    y_prob: Optional[np.ndarray] = None,
    min_group_size: int = 5,
) -> pd.DataFrame:
    """
    Metrics per group, one row each, sorted by n_samples.

    Pass groups as the raw categorical column, not the encoded one. Small
    groups are kept but marked reliable=False rather than dropped.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = pd.Series(groups).reset_index(drop=True)

    if not (len(y_true) == len(y_pred) == len(groups)):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}, "
            f"groups={len(groups)}"
        )

    rows = []
    for group_name in groups.unique():
        mask = (groups == group_name).values
        n = int(mask.sum())
        if n == 0:
            continue

        yt, yp = y_true[mask], y_pred[mask]
        row = {
            "group": group_name,
            "n_samples": n,
            "n_positive": int(yt.sum()),
            "accuracy": accuracy_score(yt, yp),
            "precision": precision_score(yt, yp, zero_division=0),
            "recall": recall_score(yt, yp, zero_division=0),
            "f1": f1_score(yt, yp, zero_division=0),
            "reliable": n >= min_group_size,
        }

        if y_prob is not None:
            yp_prob = np.asarray(y_prob)[mask]
            row["roc_auc"] = (
                roc_auc_score(yt, yp_prob) if len(np.unique(yt)) > 1 else float("nan")
            )

        rows.append(row)

    result = pd.DataFrame(rows).sort_values("n_samples", ascending=False)
    return result.reset_index(drop=True)


def disparity_summary(subgroup_df: pd.DataFrame, metric: str = "f1") -> Dict:
    """Best/worst spread over the reliable rows only, so one tiny group can't set the range."""
    if metric not in subgroup_df.columns:
        raise ValueError(f"metric '{metric}' not found in subgroup_df columns")

    reliable = subgroup_df[subgroup_df["reliable"]].dropna(subset=[metric])
    excluded = len(subgroup_df) - len(reliable)

    if reliable.empty:
        raise ValueError("No subgroups have enough samples to be considered reliable")

    worst = reliable.loc[reliable[metric].idxmin()]
    best = reliable.loc[reliable[metric].idxmax()]

    return {
        "metric": metric,
        "min": float(reliable[metric].min()),
        "max": float(reliable[metric].max()),
        "range": float(reliable[metric].max() - reliable[metric].min()),
        "worst_group": worst["group"],
        "best_group": best["group"],
        "n_groups_evaluated": len(reliable),
        "n_groups_excluded_small": int(excluded),
    }
