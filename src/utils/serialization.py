"""Save/load helpers for models, config and frames."""

import joblib
import pickle
import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class Serializer:
    @staticmethod
    def save_model(model: Any, path: Path, protocol: int = 4):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, path, protocol=protocol)
        logger.info(f"Model saved to {path}")

    @staticmethod
    def load_model(path: Path) -> Any:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return model

    @staticmethod
    def save_json(data: Dict, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"JSON saved to {path}")

    @staticmethod
    def load_json(path: Path) -> Dict:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path) as f:
            return json.load(f)

    @staticmethod
    def save_yaml(data: Dict, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

        logger.info(f"YAML saved to {path}")

    @staticmethod
    def load_yaml(path: Path) -> Dict:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path) as f:
            return yaml.safe_load(f)

    @staticmethod
    def save_pickle(data: Any, path: Path, protocol: int = 4):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=protocol)

        logger.info(f"Pickle saved to {path}")

    @staticmethod
    def load_pickle(path: Path) -> Any:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def save_dataframe(df: pd.DataFrame, path: Path):
        """Format comes from the extension; anything unrecognised falls back to CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        ext = path.suffix.lower()

        if ext == ".csv":
            df.to_csv(path, index=False)
        elif ext in [".xlsx", ".xls"]:
            df.to_excel(path, index=False)
        elif ext == ".parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)

        logger.info(f"DataFrame saved to {path}")

    @staticmethod
    def load_dataframe(path: Path) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()

        if ext == ".csv":
            return pd.read_csv(path)
        elif ext in [".xlsx", ".xls"]:
            return pd.read_excel(path)
        elif ext == ".parquet":
            return pd.read_parquet(path)
        else:
            return pd.read_csv(path)
