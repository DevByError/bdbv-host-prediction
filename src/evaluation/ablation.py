"""Leave-one-feature-group-out ablation."""

import copy
import logging
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# matched as substrings against the engineered column names, so this needs
# updating whenever FeatureEngineer grows a new family
FEATURE_GROUPS: Dict[str, List[str]] = {
    "genomic": ["gc_content", "cpg_", "genome_length", "gp_length", "gp_hydro",
                "gp_binding_energy"],
    "bdbv_specific": ["bdbv_gp_binding_affinity", "bdbv_npc1_residue_match",
                       "is_bdbv", "is_ebov", "is_marv", "is_sudv", "is_restv",
                       "bdbv_risk"],
    "host_life_history": ["body_mass", "lifespan", "geographic_range", "range_log"],
    "host_taxonomy": ["is_primate", "is_chiroptera", "is_rodentia",
                       "is_artiodactyla", "is_hominidae", "is_pteropodidae",
                       "is_vespertilionidae", "is_rhinolophidae", "is_muridae",
                       "host_order", "host_family"],
    "host_ecology_behavior": ["human_contact", "social_group", "urban_habitat",
                               "forest_habitat", "cave_habitat", "farm_habitat",
                               "is_frugivore", "is_omnivore", "is_herbivore",
                               "is_insectivore", "diet_type", "habitat_type",
                               "social_behavior", "human_contact_frequency"],
    "engineered_risk_scores": ["baseline_risk", "taxonomic_risk", "ecological_risk",
                                "contact_risk", "combined_risk", "high_risk"],
}


def _columns_for_group(all_columns: List[str], patterns: List[str]) -> List[str]:
    """Substring match, not exact - a pattern can pull in more than one column."""
    return [c for c in all_columns if any(p in c for p in patterns)]


def run_ablation_study(
    model_factory: Callable,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_groups: Optional[Dict[str, List[str]]] = None,
    scoring_metric: str = "f1",
) -> pd.DataFrame:
    """
    Refit without each group, then with each group alone.

    model_factory is a zero-arg callable giving a fresh unfitted model.
    X_train/X_test must already be engineered on the training split only.
    Returns one row per condition: condition, n_features_used, score,
    score_delta_vs_full.
    """
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("X_train and X_test must not be empty")

    from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

    def _score(model, X, y):
        pred = model.predict(X)
        if scoring_metric == "f1":
            return f1_score(y, pred, zero_division=0)
        elif scoring_metric == "accuracy":
            return accuracy_score(y, pred)
        elif scoring_metric == "roc_auc":
            proba = model.predict_proba(X)[:, 1]
            return roc_auc_score(y, proba)
        else:
            raise ValueError(f"Unsupported scoring_metric: {scoring_metric}")

    groups = feature_groups or FEATURE_GROUPS
    all_columns = list(X_train.columns)

    group_columns = {
        name: _columns_for_group(all_columns, patterns)
        for name, patterns in groups.items()
    }
    group_columns = {k: v for k, v in group_columns.items() if v}  # drop groups with no hits
    if not group_columns:
        raise ValueError(
            "No configured feature groups matched any column in X_train — "
            "check FEATURE_GROUPS patterns against the engineered feature names"
        )

    full_model = model_factory()
    full_model.fit(X_train, y_train)
    full_score = _score(full_model, X_test, y_test)

    rows = [
        {
            "condition": "full_model",
            "n_features_used": X_train.shape[1],
            "score": full_score,
            "score_delta_vs_full": 0.0,
        }
    ]

    logger.info(f"Ablation baseline ({scoring_metric}): {full_score:.4f}")

    for group_name, cols in group_columns.items():
        remaining = [c for c in all_columns if c not in cols]
        if remaining:
            model = model_factory()
            model.fit(X_train[remaining], y_train)
            score = _score(model, X_test[remaining], y_test)
        else:
            score = float("nan")
        rows.append(
            {
                "condition": f"without_{group_name}",
                "n_features_used": len(remaining),
                "score": score,
                "score_delta_vs_full": score - full_score,
            }
        )
        logger.info(
            f"  Without '{group_name}' ({len(cols)} cols removed): "
            f"{scoring_metric}={score:.4f} (delta={score - full_score:+.4f})"
        )

        # and the other direction: how far this group gets on its own
        if cols:
            model_only = model_factory()
            model_only.fit(X_train[cols], y_train)
            score_only = _score(model_only, X_test[cols], y_test)
            rows.append(
                {
                    "condition": f"only_{group_name}",
                    "n_features_used": len(cols),
                    "score": score_only,
                    "score_delta_vs_full": score_only - full_score,
                }
            )

    return pd.DataFrame(rows)
