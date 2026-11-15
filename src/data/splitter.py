"""Train/test splitting."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from sklearn.model_selection import train_test_split, StratifiedKFold
import logging

logger = logging.getLogger(__name__)


class DataSplitter:
    """Stratified or species-grouped splits, plus stratified CV folds."""

    def __init__(self, random_state: int = 42, test_size: float = 0.25):
        self.random_state = random_state
        self.test_size = test_size
        self.splits = {}

    def stratified_split(
        self, X: pd.DataFrame, y: pd.Series, val_size: Optional[float] = None
    ) -> Tuple:
        """Returns 4 arrays, or 6 (train/val/test) when val_size is given.
        val_size is a fraction of the whole dataset, not of train+val."""
        logger.info(f"Performing stratified split (test={self.test_size:.1%})")

        logger.info(f"Class distribution: {np.bincount(y)}")

        if val_size:
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y,
            )

            # rescale so val_size stays a fraction of the full dataset
            val_relative = val_size / (1 - self.test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val,
                y_train_val,
                test_size=val_relative,
                random_state=self.random_state,
                stratify=y_train_val,
            )

            self.splits = {
                "train": (X_train, y_train),
                "val": (X_val, y_val),
                "test": (X_test, y_test),
            }

            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y,
            )

            self.splits = {"train": (X_train, y_train), "test": (X_test, y_test)}

            return X_train, X_test, y_train, y_test

    def species_split(
        self, X: pd.DataFrame, y: pd.Series, species_column: str
    ) -> Tuple:
        """Whole species go to one side or the other, so the model can't
        memorise a species it will be tested on."""
        logger.info("Performing species-based split")

        species = X[species_column].unique()
        n_species = len(species)
        n_test_species = max(1, int(n_species * self.test_size))

        test_species = np.random.choice(species, n_test_species, replace=False)

        test_mask = X[species_column].isin(test_species)
        train_mask = ~test_mask

        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]

        logger.info(
            f"Train: {len(X_train)} samples, {len(train_mask[train_mask].unique())} species"
        )
        logger.info(f"Test: {len(X_test)} samples, {len(test_species)} species")

        self.splits = {"train": (X_train, y_train), "test": (X_test, y_test)}

        return X_train, X_test, y_train, y_test

    def get_cv_splits(self, X: pd.DataFrame, y: pd.Series, n_folds: int = 5) -> list:
        """List of (train_idx, test_idx) index arrays."""
        logger.info(f"Generating {n_folds}-fold CV splits")

        skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=self.random_state
        )

        splits = list(skf.split(X, y))

        logger.info(f"Generated {len(splits)} CV splits")

        return splits

    def get_split_info(self) -> Dict:
        info = {
            "test_size": self.test_size,
            "random_state": self.random_state,
            "n_splits": len(self.splits),
        }

        for name, (X, y) in self.splits.items():
            info[f"{name}_samples"] = len(X)
            info[f"{name}_positive"] = y.sum()
            info[f"{name}_negative"] = len(y) - y.sum()

        return info
