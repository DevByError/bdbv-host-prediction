"""Non-ML baselines to compare the models against."""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, List, Tuple
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaselineModel(ABC):
    """Abstract base for baseline models."""

    def __init__(self, name: str):
        self.name = name
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray):
        pass

    @abstractmethod
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        pass

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def __repr__(self) -> str:
        return f"{self.name}(fitted={self._is_fitted})"


class MajorityClassBaseline(BaselineModel):
    """Always predicts the training majority class."""

    def __init__(self):
        super().__init__("MajorityClassBaseline")
        self.majority_class = None
        self.class_proportions = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray):
        y = np.asarray(y)
        unique, counts = np.unique(y, return_counts=True)
        self.majority_class = unique[np.argmax(counts)]
        self.class_proportions = counts / len(y)
        self._is_fitted = True
        logger.info(f"MajorityClassBaseline fitted: majority class = {self.majority_class}")
        return self

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        # degenerate: all mass on one class, so AUC is undefined for this one
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")

        n_samples = len(X) if isinstance(X, (list, np.ndarray)) else len(X)
        proba = np.zeros((n_samples, 2))
        proba[:, self.majority_class] = 1.0
        return proba


class RandomBaseline(BaselineModel):
    """Uniform random scores. Chance level reference."""

    def __init__(self, random_state: int = 42):
        super().__init__("RandomBaseline")
        self.random_state = random_state
        self.class_proportions = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray):
        y = np.asarray(y)
        unique, counts = np.unique(y, return_counts=True)
        self.class_proportions = counts / len(y)
        self._is_fitted = True
        logger.info(f"RandomBaseline fitted: class proportions = {self.class_proportions}")
        return self

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        # fixed seed, so repeated calls on the same X give the same scores
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")

        rng = np.random.RandomState(self.random_state)
        n_samples = len(X) if isinstance(X, (list, np.ndarray)) else len(X)

        proba = np.zeros((n_samples, 2))
        for i in range(n_samples):
            proba[i, 1] = rng.uniform(0, 1)
            proba[i, 0] = 1 - proba[i, 1]

        return proba


class PhylogeneticDistanceBaseline(BaselineModel):
    """
    Score = max Hamming similarity to any positive training sequence.

    Needs a 'virus_sequence' or 'sequence' column, so it only works on
    DataFrame input, not on the engineered feature matrix.
    """

    def __init__(self, similarity_threshold: float = 0.8):
        super().__init__("PhylogeneticDistanceBaseline")
        self.positive_sequences: List[str] = []
        self.similarity_threshold = similarity_threshold

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray):
        if isinstance(X, np.ndarray):
            raise ValueError("PhylogeneticDistanceBaseline requires DataFrame input with sequence column")

        if "virus_sequence" not in X.columns and "sequence" not in X.columns:
            raise ValueError("X must contain 'virus_sequence' or 'sequence' column")

        seq_col = "virus_sequence" if "virus_sequence" in X.columns else "sequence"
        y = np.asarray(y)

        self.positive_sequences = X.loc[y == 1, seq_col].tolist()
        self._is_fitted = True
        logger.info(
            f"PhylogeneticDistanceBaseline fitted: {len(self.positive_sequences)} "
            "positive reference sequences"
        )
        return self

    @staticmethod
    def _hamming_similarity(seq1: str, seq2: str) -> float:
        """1 - normalized Hamming distance; 0 for unequal lengths (no alignment here)."""
        if len(seq1) != len(seq2):
            return 0.0
        distance = sum(c1 != c2 for c1, c2 in zip(seq1, seq2)) / len(seq1)
        return 1 - distance

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")

        if isinstance(X, np.ndarray):
            raise ValueError("PhylogeneticDistanceBaseline requires DataFrame input")

        seq_col = "virus_sequence" if "virus_sequence" in X.columns else "sequence"
        sequences = X[seq_col].tolist()

        proba = np.zeros((len(sequences), 2))

        for i, query_seq in enumerate(sequences):
            if not self.positive_sequences:
                # nothing to compare against, abstain
                proba[i] = [0.5, 0.5]
                continue

            similarities = [
                self._hamming_similarity(query_seq, ref_seq)
                for ref_seq in self.positive_sequences
            ]
            max_similarity = max(similarities) if similarities else 0.0

            # similarity used directly as a score, not a calibrated probability
            proba[i, 1] = min(max_similarity, 1.0)
            proba[i, 0] = 1 - proba[i, 1]

        return proba


class EcologicalSimilarityBaseline(BaselineModel):
    """Cosine similarity to the mean ecological profile of the positives."""

    def __init__(self, ecological_features: Optional[List[str]] = None):
        super().__init__("EcologicalSimilarityBaseline")
        self.ecological_features = ecological_features or [
            "body_mass", "lifespan", "habitat_type", "diet_type",
            "social_behavior", "human_contact_frequency", "geographic_range"
        ]
        self.positive_ecology: Optional[np.ndarray] = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        y = np.asarray(y)

        # use whichever of the expected columns are actually present
        available_features = [f for f in self.ecological_features if f in X.columns]

        if not available_features:
            raise ValueError(
                f"None of {self.ecological_features} found in X.columns: {X.columns.tolist()}"
            )

        X_eco = X[available_features].fillna(0).values
        X_eco = (X_eco - X_eco.mean(axis=0)) / (X_eco.std(axis=0) + 1e-8)

        self.positive_ecology = X_eco[y == 1].mean(axis=0)
        self._is_fitted = True
        logger.info(
            f"EcologicalSimilarityBaseline fitted on {available_features}"
        )
        return self

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        available_features = [f for f in self.ecological_features if f in X.columns]
        X_eco = X[available_features].fillna(0).values
        # standardized against this batch, not the training stats - fine for a baseline
        X_eco = (X_eco - X_eco.mean(axis=0)) / (X_eco.std(axis=0) + 1e-8)

        proba = np.zeros((len(X_eco), 2))
        for i in range(len(X_eco)):
            similarity = np.dot(X_eco[i], self.positive_ecology) / (
                np.linalg.norm(X_eco[i]) * np.linalg.norm(self.positive_ecology) + 1e-8
            )
            # cosine is in [-1, 1], rescale to [0, 1]
            proba[i, 1] = (similarity + 1) / 2
            proba[i, 0] = 1 - proba[i, 1]

        return proba
