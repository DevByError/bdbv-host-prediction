"""Project-wide settings. Everything tunable lives here."""

import os
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass
class PathConfig:
    ROOT_DIR: Path = Path(__file__).parent.parent

    DATA_DIR: Path = ROOT_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    EXTERNAL_DATA_DIR: Path = DATA_DIR / "external"

    RESULTS_DIR: Path = ROOT_DIR / "results"
    FIGURES_DIR: Path = RESULTS_DIR / "figures"
    MODELS_DIR: Path = RESULTS_DIR / "models"
    REPORTS_DIR: Path = RESULTS_DIR / "reports"

    LOGS_DIR: Path = ROOT_DIR / "logs"

    DATA_TEMPLATE: Path = RAW_DATA_DIR / "filovirus_data_template.csv"
    BDBV_REFERENCE: Path = RAW_DATA_DIR / "bdbv_reference_data.csv"

    def __post_init__(self):
        # make sure the tree exists before anything tries to write into it
        for dir_path in [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.EXTERNAL_DATA_DIR,
            self.RESULTS_DIR,
            self.FIGURES_DIR,
            self.MODELS_DIR,
            self.REPORTS_DIR,
            self.LOGS_DIR,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    # Random Forest
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = None
    rf_min_samples_split: int = 2
    rf_min_samples_leaf: int = 1
    rf_class_weight: str = "balanced"

    # XGBoost
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8

    # GNN
    gnn_hidden_dim: int = 128
    gnn_num_layers: int = 3
    gnn_learning_rate: float = 0.001
    gnn_dropout: float = 0.2


@dataclass
class TrainingConfig:
    # Split
    test_size: float = 0.25
    val_size: float = 0.2
    random_state: int = 42

    # Cross-validation
    n_folds: int = 5
    stratified: bool = True

    # SMOTE
    use_smote: bool = True
    smote_k_neighbors: int = 3

    # Training
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10
    device: str = "cuda"  # or "cpu"


@dataclass
class FeatureConfig:
    # Biological features
    use_codon_analysis: bool = True
    use_protein_analysis: bool = True
    use_phylogenetic_features: bool = True
    use_npc1_binding: bool = True

    # Feature selection
    max_features: int = 50
    variance_threshold: float = 0.01
    correlation_threshold: float = 0.9

    # Encoding
    categorical_encoding: str = "onehot"  # or "target"


@dataclass
class VisualizationConfig:
    # Figure settings
    figure_dpi: int = 600
    figure_format: str = "png"
    vector_format: str = "svg"  # For publications

    # colourblind-safe, same palette as the paper figures
    primary_color: str = "#2E86AB"  # Blue
    secondary_color: str = "#A23B72"  # Purple
    success_color: str = "#3D8C40"  # Green
    warning_color: str = "#F18F01"  # Orange
    danger_color: str = "#C73E1D"  # Red

    # Font settings
    font_family: str = "Arial"
    font_size_title: int = 16
    font_size_label: int = 13
    font_size_tick: int = 11


@dataclass
class Config:
    paths: PathConfig = PathConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    features: FeatureConfig = FeatureConfig()
    visualization: VisualizationConfig = VisualizationConfig()

    # Global
    verbose: bool = True
    debug: bool = False
    seed: int = 42

    def __post_init__(self):
        # imported here, top-level numpy import makes config import heavy
        import random
        import numpy as np
        import os

        random.seed(self.seed)
        np.random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)

        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Config initialized with seed={self.seed}")

    def to_dict(self) -> dict:
        """Flat summary of the settings we actually report in runs."""
        return {
            "seed": self.seed,
            "verbose": self.verbose,
            "debug": self.debug,
            "paths": {
                "root": str(self.paths.ROOT_DIR),
                "data": str(self.paths.DATA_DIR),
                "results": str(self.paths.RESULTS_DIR),
            },
            "model": {
                "random_forest": {
                    "n_estimators": self.model.rf_n_estimators,
                    "max_depth": self.model.rf_max_depth,
                },
                "xgboost": {
                    "n_estimators": self.model.xgb_n_estimators,
                    "max_depth": self.model.xgb_max_depth,
                    "learning_rate": self.model.xgb_learning_rate,
                },
            },
            "training": {
                "test_size": self.training.test_size,
                "n_folds": self.training.n_folds,
                "use_smote": self.training.use_smote,
            },
        }


config = Config()
