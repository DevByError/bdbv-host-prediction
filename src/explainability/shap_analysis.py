"""
SHAP analysis for model interpretability.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Union, Tuple
import shap
import logging

logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """
    Comprehensive SHAP analysis for model interpretability.

    Features:
    - Global feature importance
    - Local explanations
    - Feature interaction analysis
    - Biological pathway mapping
    """

    def __init__(self, model, feature_names: List[str], model_type: str = "tree"):
        """
        Initialize SHAP analyzer.

        Args:
            model: Trained model
            feature_names: List of feature names
            model_type: 'tree', 'linear', or 'kernel'
        """
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None
        self.background_data = None
        self.expected_value = None

        logger.info(f"Initialized SHAP analyzer with {len(feature_names)} features")

    def fit_explainer(
        self, X_sample: Union[np.ndarray, pd.DataFrame], max_samples: int = 100
    ) -> "SHAPAnalyzer":
        """
        Fit the SHAP explainer.

        Args:
            X_sample: Sample data for background
            max_samples: Maximum samples to use

        Returns:
            SHAPAnalyzer: Self
        """
        logger.info("Fitting SHAP explainer...")

        if isinstance(X_sample, pd.DataFrame):
            X_sample = X_sample.values

        # Limit samples
        if len(X_sample) > max_samples:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(X_sample), max_samples, replace=False)
            X_sample = X_sample[indices]

        self.background_data = X_sample

        # Choose explainer
        try:
            if self.model_type == "tree":
                self.explainer = shap.TreeExplainer(self.model)
            elif self.model_type == "linear":
                self.explainer = shap.LinearExplainer(self.model, X_sample)
            else:
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba, X_sample
                )
        except Exception as e:
            logger.warning(
                f"Could not create TreeExplainer, falling back to KernelExplainer: {e}"
            )
            self.explainer = shap.KernelExplainer(self.model.predict_proba, X_sample)
            self.model_type = "kernel"

        # Calculate SHAP values
        try:
            self.shap_values = self.explainer.shap_values(X_sample)

            # Handle different return types
            if isinstance(self.shap_values, list):
                # For binary classification, use class 1
                if len(self.shap_values) > 1:
                    self.shap_values = self.shap_values[1]
                else:
                    self.shap_values = self.shap_values[0]

            # Store expected value
            if hasattr(self.explainer, "expected_value"):
                if isinstance(self.explainer.expected_value, (list, np.ndarray)):
                    self.expected_value = (
                        self.explainer.expected_value[1]
                        if len(self.explainer.expected_value) > 1
                        else self.explainer.expected_value[0]
                    )
                else:
                    self.expected_value = self.explainer.expected_value

        except Exception as e:
            logger.error(f"Failed to calculate SHAP values: {e}")
            raise

        logger.info(f"SHAP explainer fitted with {len(X_sample)} samples")
        return self

    def plot_summary(
        self, max_display: int = 20, output_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Generate SHAP summary plot.

        Args:
            max_display: Maximum features to display
            output_path: Path to save figure

        Returns:
            plt.Figure: Figure object
        """
        if self.shap_values is None:
            raise ValueError("Must call fit_explainer first")

        logger.info(f"Generating SHAP summary plot (top {max_display})")

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Create summary plot without ax parameter
        shap.summary_plot(
            self.shap_values,
            self.background_data,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
        )

        # Get the current figure and set title
        current_fig = plt.gcf()
        current_fig.suptitle(
            "SHAP Feature Importance Summary", fontsize=16, fontweight="bold"
        )
        plt.xlabel(
            "SHAP Value (Impact on Model Output)", fontsize=12, fontweight="bold"
        )

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved summary plot to {output_path}")

        return current_fig

    def plot_bar(
        self, max_display: int = 20, output_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Generate SHAP bar plot (feature importance).

        Args:
            max_display: Maximum features to display
            output_path: Path to save figure

        Returns:
            plt.Figure: Figure object
        """
        if self.shap_values is None:
            raise ValueError("Must call fit_explainer first")

        logger.info(f"Generating SHAP bar plot (top {max_display})")

        # Create bar plot
        shap.summary_plot(
            self.shap_values,
            self.background_data,
            feature_names=self.feature_names,
            max_display=max_display,
            plot_type="bar",
            show=False,
        )

        # Get the current figure
        current_fig = plt.gcf()
        current_fig.suptitle(
            "Top Feature Importance (Mean |SHAP|)", fontsize=16, fontweight="bold"
        )
        plt.xlabel("Mean |SHAP Value|", fontsize=12, fontweight="bold")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved bar plot to {output_path}")

        return current_fig

    def plot_waterfall(
        self, instance_index: int, output_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Generate waterfall plot for a single instance.

        Args:
            instance_index: Index of instance to explain
            output_path: Path to save figure

        Returns:
            plt.Figure: Figure object
        """
        if self.shap_values is None:
            raise ValueError("Must call fit_explainer first")

        logger.info(f"Generating waterfall plot for instance {instance_index}")

        # Create waterfall plot
        shap.waterfall_plot(
            shap.Explanation(
                values=self.shap_values[instance_index],
                base_values=(
                    self.expected_value if self.expected_value is not None else 0
                ),
                data=self.background_data[instance_index],
                feature_names=self.feature_names,
            ),
            show=False,
            max_display=15,
        )

        # Get the current figure
        current_fig = plt.gcf()
        current_fig.suptitle(
            f"SHAP Waterfall Plot - Instance {instance_index}",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved waterfall plot to {output_path}")

        return current_fig

    def plot_force(self, instance_index: int, output_path: Optional[str] = None):
        """
        Generate force plot for a single instance.

        Args:
            instance_index: Index of instance to explain
            output_path: Path to save HTML file
        """
        if self.shap_values is None:
            raise ValueError("Must call fit_explainer first")

        logger.info(f"Generating force plot for instance {instance_index}")

        # Create force plot
        force_plot = shap.force_plot(
            self.expected_value if self.expected_value is not None else 0,
            self.shap_values[instance_index],
            self.background_data[instance_index],
            feature_names=self.feature_names,
            matplotlib=False,
            show=False,
        )

        if output_path:
            # Save as HTML
            with open(output_path, "w") as f:
                f.write(str(force_plot))
            logger.info(f"Saved force plot to {output_path}")

        return force_plot

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from SHAP values.

        Returns:
            pd.DataFrame: Feature importance
        """
        if self.shap_values is None:
            raise ValueError("Must call fit_explainer first")

        importance = np.abs(self.shap_values).mean(axis=0)

        df = pd.DataFrame(
            {"feature": self.feature_names, "shap_importance": importance}
        ).sort_values("shap_importance", ascending=False)

        return df

    def get_biological_insights(self) -> Dict:
        """
        Map SHAP importance to biological categories.

        Returns:
            Dict: Biological insights
        """
        if self.shap_values is None:
            raise ValueError("Must call fit_explainer first")

        # Define biological categories
        categories = {
            "Genomic": [
                "gc_content",
                "gc_content_squared",
                "gc_content_log",
                "cpg_frequency",
                "cpg_density",
                "cpg_log",
                "viral_genome_length",
            ],
            "Glycoprotein": [
                "glycoprotein_length",
                "gp_length_ratio",
                "gp_hydrophobicity",
                "gp_hydro_abs",
            ],
            "Host_Traits": [
                "body_mass_kg",
                "body_mass_log",
                "body_mass_sqrt",
                "lifespan_years",
                "lifespan_log",
                "geographic_range_km2",
                "range_log",
            ],
            "Taxonomy": [
                "is_primate",
                "is_chiroptera",
                "is_rodentia",
                "is_artiodactyla",
                "is_hominidae",
                "is_pteropodidae",
                "is_vespertilionidae",
                "is_rhinolophidae",
                "is_muridae",
            ],
            "Ecological": [
                "is_frugivore",
                "is_omnivore",
                "is_herbivore",
                "is_insectivore",
                "social_group_score",
                "urban_habitat",
                "forest_habitat",
                "cave_habitat",
                "farm_habitat",
                "human_contact_score",
            ],
            "Virus": [
                "is_bdbv",
                "is_ebov",
                "is_marv",
                "is_sudv",
                "is_restv",
                "bdbv_risk",
            ],
            "Risk_Scores": [
                "taxonomic_risk",
                "ecological_risk",
                "contact_risk",
                "baseline_risk",
                "combined_risk",
                "high_risk",
            ],
        }

        # Calculate importance per category
        importance = self.get_feature_importance()

        category_importance = {}
        category_features = {}

        for category, features in categories.items():
            cat_importance = importance[importance["feature"].isin(features)]
            if not cat_importance.empty:
                category_importance[category] = cat_importance["shap_importance"].sum()
                category_features[category] = cat_importance.head(3).to_dict("records")

        # Rank categories
        sorted_categories = sorted(
            category_importance.items(), key=lambda x: x[1], reverse=True
        )

        insights = {
            "top_category": sorted_categories[0][0] if sorted_categories else "Unknown",
            "category_importance": sorted_categories,
            "category_top_features": category_features,
            "key_features": importance.head(10).to_dict("records"),
        }

        return insights

    def explain_instance(self, instance: Union[np.ndarray, pd.DataFrame]) -> Dict:
        """
        Get detailed explanation for a single instance.

        Args:
            instance: Single instance to explain

        Returns:
            Dict: Explanation details
        """
        if self.shap_values is None:
            raise ValueError("Must call fit_explainer first")

        if isinstance(instance, pd.DataFrame):
            instance = instance.values

        # Get SHAP values for this instance
        shap_values = self.explainer.shap_values(instance.reshape(1, -1))

        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        # Create explanation
        explanation = {
            "base_value": self.expected_value if self.expected_value is not None else 0,
            "prediction": self.model.predict(instance.reshape(1, -1))[0],
            "shap_values": shap_values[0],
            "features": self.feature_names,
            "contributions": {},
        }

        # Get top contributing features
        shap_df = pd.DataFrame(
            {"feature": self.feature_names, "shap_value": shap_values[0]}
        )
        shap_df["abs_shap"] = np.abs(shap_df["shap_value"])
        shap_df = shap_df.sort_values("abs_shap", ascending=False)

        # Top positive contributors
        explanation["top_positive"] = (
            shap_df[shap_df["shap_value"] > 0].head(5).to_dict("records")
        )

        # Top negative contributors
        explanation["top_negative"] = (
            shap_df[shap_df["shap_value"] < 0].head(5).to_dict("records")
        )

        return explanation
