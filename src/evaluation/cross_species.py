"""Leave-one-species-out validation and per-species breakdowns."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)


class CrossSpeciesValidator:
    def __init__(self, species_column: str = "host_species"):
        self.species_column = species_column

    def leave_one_species_out(
        self,
        model,
        X: pd.DataFrame,
        y: np.ndarray,
        species: np.ndarray,
    ) -> Dict:
        """
        Hold out one species at a time.

        Returns per_species / overall metrics plus the stitched-together
        out-of-fold predictions and probabilities.
        """
        species = np.asarray(species)
        unique_species = np.unique(species)
        n_samples = len(y)

        # -1 marks rows never predicted, filtered out before overall metrics
        loso_pred = np.full(n_samples, -1, dtype=int)
        loso_proba = np.zeros((n_samples, 2))

        per_species_results = []

        logger.info(f"Leave-One-Species-Out CV: {len(unique_species)} species")

        for test_species in unique_species:
            test_mask = species == test_species
            train_mask = ~test_mask

            X_train = X.iloc[train_mask] if isinstance(X, pd.DataFrame) else X[train_mask]
            X_test = X.iloc[test_mask] if isinstance(X, pd.DataFrame) else X[test_mask]
            y_train = y[train_mask]
            y_test = y[test_mask]

            if len(y_test) == 0:
                continue

            model_clone = model.__class__(**model.get_params()) if hasattr(model, 'get_params') else model
            model_clone.fit(X_train, y_train)

            y_pred = model_clone.predict(X_test)
            y_proba = model_clone.predict_proba(X_test)[:, 1]

            loso_pred[test_mask] = y_pred
            loso_proba[test_mask, 0] = 1 - y_proba
            loso_proba[test_mask, 1] = y_proba

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            try:
                auc = roc_auc_score(y_test, y_proba)
            except ValueError:
                auc = np.nan  # species is single-class

            per_species_results.append({
                "species": test_species,
                "n_samples": len(y_test),
                "n_positive": np.sum(y_test),
                "accuracy": acc,
                "f1": f1,
                "roc_auc": auc,
            })

            logger.debug(
                f"  {test_species}: n={len(y_test)}, acc={acc:.3f}, f1={f1:.3f}"
            )

        valid_mask = loso_pred >= 0
        overall_acc = accuracy_score(y[valid_mask], loso_pred[valid_mask])
        overall_f1 = f1_score(y[valid_mask], loso_pred[valid_mask])

        try:
            overall_auc = roc_auc_score(y[valid_mask], loso_proba[valid_mask, 1])
        except ValueError:
            overall_auc = np.nan

        per_species_df = pd.DataFrame(per_species_results).sort_values("n_samples", ascending=False)

        return {
            "per_species": per_species_df,
            "overall": {
                "accuracy": overall_acc,
                "f1": overall_f1,
                "roc_auc": overall_auc,
                "n_species": len(unique_species),
                "n_samples": np.sum(valid_mask),
            },
            "loso_predictions": loso_pred,
            "loso_probabilities": loso_proba,
        }

    def phylogenetic_distance_analysis(
        self,
        species: np.ndarray,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        phylogenetic_distances: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Per-species accuracy/f1 frame.

        phylogenetic_distances isn't used yet - the distance binning is still
        to do, so for now this is just the per-species table.
        """
        species = np.asarray(species)
        unique_species = np.unique(species)

        results = []

        for spec in unique_species:
            mask = species == spec
            acc = accuracy_score(y_true[mask], y_pred[mask])
            f1 = f1_score(y_true[mask], y_pred[mask])

            results.append({
                "species": spec,
                "accuracy": acc,
                "f1": f1,
                "n_samples": np.sum(mask),
            })

        return pd.DataFrame(results)

    @staticmethod
    def per_species_performance(
        species: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """y_proba is the full (n, 2) matrix, column 1 is used. Sorted by n_samples."""
        species = np.asarray(species)
        unique_species = np.unique(species)

        results = []

        for spec in unique_species:
            mask = species == spec
            n_samples = np.sum(mask)

            if n_samples == 0:
                continue

            y_t = y_true[mask]
            y_p = y_pred[mask]

            acc = accuracy_score(y_t, y_p)
            f1 = f1_score(y_t, y_p)

            auc = np.nan
            if y_proba is not None and len(np.unique(y_t)) > 1:
                try:
                    auc = roc_auc_score(y_t, y_proba[mask, 1])
                except ValueError:
                    auc = np.nan

            results.append({
                "species": spec,
                "n_samples": n_samples,
                "n_positive": np.sum(y_t),
                "accuracy": acc,
                "f1": f1,
                "roc_auc": auc,
            })

        return pd.DataFrame(results).sort_values("n_samples", ascending=False)
