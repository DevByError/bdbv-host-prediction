"""Data loading for the BDBV prediction pipeline."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any
import logging
import json
from datetime import datetime
import hashlib

from config.config import config
from src.data.generator import DataGenerator

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads CSV/Excel/Parquet/JSON/feather/HDF5 by extension, or SQL."""

    def __init__(
        self,
        data_path: Optional[Path] = None,
        validate: bool = True,
        cache_enabled: bool = True,
        random_state: int = 42,
    ):
        """No data_path (or a missing file) falls back to synthetic data."""
        self.data_path = data_path or config.paths.DATA_TEMPLATE
        self.validate = validate
        self.cache_enabled = cache_enabled
        self.random_state = random_state
        self.metadata = {}
        self._cache = {}
        self._data_hash = None

        if self.data_path and self.data_path.parent:
            self.data_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"DataLoader initialized with validate={validate}")

    def load_data(self, **kwargs) -> pd.DataFrame:
        """kwargs go straight through to the pandas read_* call."""
        logger.info(f"Loading data from {self.data_path}")

        if self.cache_enabled and self.data_path:
            cache_key = str(self.data_path)
            if cache_key in self._cache:
                logger.info("Returning cached data")
                return self._cache[cache_key].copy()

        if not self.data_path or not self.data_path.exists():
            logger.warning(f"Data file not found: {self.data_path}")
            logger.info("Generating synthetic data for testing...")
            df = self._generate_test_data()
        else:
            df = self._load_by_extension(kwargs)

        self._data_hash = self._compute_hash(df)

        if self.validate:
            df = self._validate_data(df)

        self._add_metadata(df)

        if self.cache_enabled and self.data_path:
            self._cache[str(self.data_path)] = df.copy()

        logger.info(f"Successfully loaded {len(df)} rows, {len(df.columns)} columns")
        return df

    def _load_by_extension(self, kwargs: Dict[str, Any]) -> pd.DataFrame:
        file_ext = self.data_path.suffix.lower()

        try:
            if file_ext == ".csv":
                return pd.read_csv(self.data_path, **kwargs)
            elif file_ext in [".xlsx", ".xls"]:
                return pd.read_excel(self.data_path, **kwargs)
            elif file_ext == ".parquet":
                return pd.read_parquet(self.data_path, **kwargs)
            elif file_ext == ".json":
                return pd.read_json(self.data_path, **kwargs)
            elif file_ext == ".feather":
                return pd.read_feather(self.data_path, **kwargs)
            elif file_ext == ".h5" or file_ext == ".hdf5":
                return pd.read_hdf(self.data_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
        except Exception as e:
            logger.error(f"Failed to load {file_ext} file: {e}")
            logger.info("Falling back to synthetic data generation...")
            return self._generate_test_data()

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Runs DataValidator and repairs what's safely repairable."""
        from src.data.validator import DataValidator

        validator = DataValidator()
        results = validator.validate(df)
        self.metadata["validation"] = results

        for warning in results.get("warnings", []):
            logger.warning(warning)

        if not results["valid"]:
            error_msg = "; ".join(results.get("errors", []))
            logger.error(f"Data validation failed: {error_msg}")
            logger.info("Attempting to auto-repair fixable data issues...")
            df = self._fix_data_issues(df)

        # sklearn needs a numeric target downstream
        if "infection_status" in df.columns and not pd.api.types.is_numeric_dtype(
            df["infection_status"]
        ):
            logger.warning("infection_status is not numeric; coercing to int")
            df["infection_status"] = (
                pd.to_numeric(df["infection_status"], errors="coerce")
                .fillna(0)
                .astype(int)
            )

        return df

    def _fix_data_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Median fill for numerics, mode for categoricals, 0 for the target."""
        if "infection_status" in df.columns:
            df["infection_status"] = df["infection_status"].fillna(0).astype(int)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())

        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if not df[col].empty:
                df[col] = df[col].fillna(
                    df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                )

        logger.info("Data issues fixed")
        return df

    def _generate_test_data(self) -> pd.DataFrame:
        """Fallback when there's no real data. Goes through DataGenerator so
        the schema matches what the rest of the pipeline expects."""
        logger.info("Generating synthetic fallback data via DataGenerator...")
        generator = DataGenerator(random_state=self.random_state)
        df = generator.generate_dataset(n_samples=200, bdbv_ratio=0.3)
        logger.info(f"Generated {len(df)} synthetic rows")
        return df

    def _add_metadata(self, df: pd.DataFrame) -> None:
        self.metadata.update(
            {
                "n_samples": len(df),
                "n_features": len(df.columns),
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.to_dict(),
                "n_categorical": len(df.select_dtypes(include=["object"]).columns),
                "n_numerical": len(df.select_dtypes(include=[np.number]).columns),
                "missing_values": df.isnull().sum().sum(),
                "data_hash": self._data_hash,
                "load_time": datetime.now().isoformat(),
                "data_path": str(self.data_path) if self.data_path else None,
            }
        )

    def _compute_hash(self, df: pd.DataFrame) -> str:
        """md5 of the CSV rendering, used to spot data changes between runs."""
        df_bytes = df.to_csv(index=False).encode("utf-8")
        return hashlib.md5(df_bytes).hexdigest()

    def load_from_database(
        self, query: str, connection_string: str, **kwargs
    ) -> pd.DataFrame:
        try:
            import sqlalchemy
        except ImportError:
            logger.error(
                "sqlalchemy not installed. Install with: pip install sqlalchemy"
            )
            raise

        logger.info("Loading data from database")
        engine = sqlalchemy.create_engine(connection_string)

        try:
            df = pd.read_sql(query, engine, **kwargs)
            logger.info(f"Loaded {len(df)} rows from database")
            return df
        except Exception as e:
            logger.error(f"Failed to load from database: {e}")
            raise

    def load_bdbv_reference(self) -> pd.DataFrame:
        """Known-susceptibility table; falls back to a small built-in template."""
        try:
            from src.data.bdbv_reference import BDBVReference

            reference = BDBVReference()
            return reference.load()
        except ImportError:
            logger.warning("BDBV reference module not found")
            return self._create_bdbv_template()

    def _create_bdbv_template(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "species": [
                    "Homo_sapiens",
                    "Pan_troglodytes",
                    "Gorilla_gorilla",
                    "Rousettus_aegyptiacus",
                    "Eidolon_helvum",
                ],
                "susceptibility": [1, 1, 1, 1, 1],
                "evidence_level": ["High", "High", "High", "High", "Medium"],
                "source": ["Literature", "Literature", "Literature", "Lab", "Lab"],
            }
        )

    def get_metadata(self) -> dict:
        return self.metadata

    def clear_cache(self) -> None:
        self._cache.clear()
        logger.info("Cache cleared")

    def get_data_hash(self) -> Optional[str]:
        return self._data_hash

    def save_metadata(self, path: Optional[Path] = None) -> None:
        """Defaults to RESULTS_DIR/data_metadata.json."""
        save_path = path or config.paths.RESULTS_DIR / "data_metadata.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)

        logger.info(f"Metadata saved to {save_path}")


def load_data(
    data_path: Optional[Path] = None, validate: bool = True, **kwargs
) -> pd.DataFrame:
    loader = DataLoader(data_path=data_path, validate=validate)
    return loader.load_data(**kwargs)
