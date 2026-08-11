"""
Data validation for quality control.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates data integrity, completeness, and quality.

    Checks:
    - Missing values
    - Duplicate rows
    - Class balance
    - Outliers
    - Data types
    - Feature correlations
    """

    def __init__(self):
        self.results = {}

    def validate(self, df: pd.DataFrame) -> Dict:
        """
        Perform comprehensive data validation.

        Args:
            df: DataFrame to validate

        Returns:
            Dict: Validation results
        """
        logger.info("Validating data...")

        results = {"valid": True, "warnings": [], "errors": [], "statistics": {}}

        # 1. Check for required columns
        required_cols = ["virus_species", "host_species", "infection_status"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            results["valid"] = False
            results["errors"].append(f"Missing required columns: {missing_cols}")
            return results

        # 2. Check sample size
        n_samples = len(df)
        results["statistics"]["n_samples"] = n_samples

        if n_samples < 50:
            results["warnings"].append(f"Small sample size: {n_samples} samples")

        # 3. Check class balance
        if "infection_status" in df.columns:
            n_positive = df["infection_status"].sum()
            n_negative = n_samples - n_positive
            balance = n_positive / n_samples if n_samples > 0 else 0

            results["statistics"]["n_positive"] = n_positive
            results["statistics"]["n_negative"] = n_negative
            results["statistics"]["class_balance"] = balance

            if balance < 0.1 or balance > 0.9:
                results["warnings"].append(
                    f"Severe class imbalance: {balance:.1%} positive"
                )
            elif balance < 0.2 or balance > 0.8:
                results["warnings"].append(
                    f"Moderate class imbalance: {balance:.1%} positive"
                )

        # 4. Check missing values
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]

        if len(missing_cols) > 0:
            results["statistics"]["missing_values"] = missing_counts.sum()
            results["warnings"].append(
                f"Missing values in columns: {list(missing_cols.index)}"
            )

            # Check for high missingness (>50%)
            high_missing = missing_cols[missing_cols > len(df) * 0.5]
            if len(high_missing) > 0:
                results["errors"].append(
                    f"High missingness (>50%) in: {list(high_missing.index)}"
                )
                results["valid"] = False

        # 5. Check duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            results["statistics"]["duplicates"] = duplicates
            results["warnings"].append(f"Found {duplicates} duplicate rows")

        # 6. Check outliers in numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        outlier_cols = []

        for col in numerical_cols:
            if col == "infection_status":
                continue

            # Use IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR

            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                outlier_cols.append((col, outliers))

        if outlier_cols:
            results["warnings"].append(
                f"Outliers detected in: {[col for col, _ in outlier_cols]}"
            )

        # 7. Check feature correlations (if enough columns)
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            high_corr = upper_tri[upper_tri > 0.9].stack().index.tolist()

            if high_corr:
                results["warnings"].append(
                    f"High correlations (>0.9) detected: {high_corr}"
                )

        # 8. Check virus distribution
        if "virus_species" in df.columns:
            virus_counts = df["virus_species"].value_counts()
            results["statistics"]["virus_distribution"] = virus_counts.to_dict()

            # Check BDBV representation
            if "BDBV" in virus_counts:
                bdbv_count = virus_counts["BDBV"]
                if bdbv_count < 10:
                    results["warnings"].append(
                        f"Low BDBV representation: {bdbv_count} samples"
                    )

        results["statistics"]["n_features"] = len(df.columns)
        results["statistics"]["n_categorical"] = len(
            df.select_dtypes(include=["object"]).columns
        )
        results["statistics"]["n_numerical"] = len(
            df.select_dtypes(include=[np.number]).columns
        )

        logger.info(f"Validation complete. Valid: {results['valid']}")
        if results["warnings"]:
            logger.warning(f"Found {len(results['warnings'])} warnings")

        return results

    def validate_for_bdbv(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Specialized validation for BDBV analysis.

        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        errors = []

        # Check BDBV samples
        bdbv_mask = df["virus_species"] == "BDBV"
        if not bdbv_mask.any():
            errors.append("No BDBV samples found in dataset")
            return False, errors

        # Check BDBV features
        required_bdbv_features = ["bdbv_gp_binding_affinity", "bdbv_npc1_residue_match"]

        for feature in required_bdbv_features:
            if feature in df.columns:
                bdbv_values = df.loc[bdbv_mask, feature]
                if bdbv_values.isnull().all():
                    errors.append(f"All BDBV {feature} values are missing")
            else:
                errors.append(f"Missing BDBV feature: {feature}")

        return len(errors) == 0, errors
