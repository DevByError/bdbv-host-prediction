"""SHAP wrappers: explainer setup, the usual plots, and a roll-up by feature group."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Union, Tuple
import logging

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    def __init__(self, model, feature_names: List[str], model_type: str = "tree"):
        """model may be a bare estimator or one of our BaseModel wrappers; both work."""
        if not SHAP_AVAILABLE:
            raise ImportError(
                "shap is not installed. Install with: pip install shap"
            )

        self.model = model
        # tree/linear explainers poke at the estimator internals, so unwrap;
        # predict_proba stays on the wrapper so its input handling still runs
        self._raw_model = getattr(model, "model", model)
        self._predict_proba = (
            model.predict_proba if hasattr(model, "predict_proba") else None
        )
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
        """X_sample doubles as the background set and as what the plots are drawn over."""
        logger.info("Fitting SHAP explainer...")

        if isinstance(X_sample, pd.DataFrame):
            X_sample = X_sample.values

        # kernel explainer gets very slow past a few hundred background rows
        if len(X_sample) > max_samples:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(X_sample), max_samples, replace=False)
            X_sample = X_sample[indices]

        self.background_data = X_sample

        if self._predict_proba is None and self.model_type != "linear":
            raise ValueError(
                "Model has no predict_proba method; required for kernel/tree "
                "fallback explainers"
            )

        try:
            if self.model_type == "tree":
                self.explainer = shap.TreeExplainer(self._raw_model)
            elif self.model_type == "linear":
                self.explainer = shap.LinearExplainer(self._raw_model, X_sample)
            else:
                self.explainer = shap.KernelExplainer(self._predict_proba, X_sample)
        except Exception as e:
            logger.warning(
                f"Could not create {self.model_type} explainer, "
                f"falling back to KernelExplainer: {e}"
            )
            self.explainer = shap.KernelExplainer(self._predict_proba, X_sample)
            self.model_type = "kernel"

        try:
            self.shap_values = self.explainer.shap_values(X_sample)

            # older shap returns one array per class; we want the positive one
            if isinstance(self.shap_values, list):
                if len(self.shap_values) > 1:
                    self.shap_values = self.shap_values[1]
                else:
                    self.shap_values = self.shap_values[0]

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
        if self.shap_values is None:
            raise ValueError("Must call fit_explainer first")

        logger.info(f"Generating SHAP summary plot (top {max_display})")

        fig, ax = plt.subplots(figsize=(12, 10))

        # summary_plot takes no ax, it draws on the current figure
        shap.summary_plot(
            self.shap_values,
            self.background_data,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
        )

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
        if self.shap_values is None:
            raise ValueError("Must call fit_explainer first")

        logger.info(f"Generating SHAP bar plot (top {max_display})")

        shap.summary_plot(
            self.shap_values,
            self.background_data,
            feature_names=self.feature_names,
            max_display=max_display,
            plot_type="bar",
            show=False,
        )

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
        if self.shap_values is None:
            raise ValueError("Must call fit_explainer first")

        logger.info(f"Generating waterfall plot for instance {instance_index}")

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
        """Writes HTML, not an image - force plots are javascript."""
        if self.shap_values is None:
            raise ValueError("Must call fit_explainer first")

        logger.info(f"Generating force plot for instance {instance_index}")

        force_plot = shap.force_plot(
            self.expected_value if self.expected_value is not None else 0,
            self.shap_values[instance_index],
            self.background_data[instance_index],
            feature_names=self.feature_names,
            matplotlib=False,
            show=False,
        )

        if output_path:
            with open(output_path, "w") as f:
                f.write(str(force_plot))
            logger.info(f"Saved force plot to {output_path}")

        return force_plot

    def get_feature_importance(self) -> pd.DataFrame:
        """feature / shap_importance, where importance is mean |SHAP| over the background rows."""
        if self.shap_values is None:
            raise ValueError("Must call fit_explainer first")

        importance = np.abs(self.shap_values).mean(axis=0)

        df = pd.DataFrame(
            {"feature": self.feature_names, "shap_importance": importance}
        ).sort_values("shap_importance", ascending=False)

        return df

    def get_biological_insights(self) -> Dict:
        """Sums SHAP importance within hand-assigned feature groups."""
        if self.shap_values is None:
            raise ValueError("Must call fit_explainer first")

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

        importance = self.get_feature_importance()

        category_importance = {}
        category_features = {}

        for category, features in categories.items():
            cat_importance = importance[importance["feature"].isin(features)]
            if not cat_importance.empty:
                category_importance[category] = cat_importance["shap_importance"].sum()
                category_features[category] = cat_importance.head(3).to_dict("records")

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
        if self.shap_values is None:
            raise ValueError("Must call fit_explainer first")

        if isinstance(instance, pd.DataFrame):
            instance = instance.values

        shap_values = self.explainer.shap_values(instance.reshape(1, -1))

        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        explanation = {
            "base_value": self.expected_value if self.expected_value is not None else 0,
            "prediction": self.model.predict(instance.reshape(1, -1))[0],
            "shap_values": shap_values[0],
            "features": self.feature_names,
            "contributions": {},
        }

        shap_df = pd.DataFrame(
            {"feature": self.feature_names, "shap_value": shap_values[0]}
        )
        shap_df["abs_shap"] = np.abs(shap_df["shap_value"])
        shap_df = shap_df.sort_values("abs_shap", ascending=False)

        explanation["top_positive"] = (
            shap_df[shap_df["shap_value"] > 0].head(5).to_dict("records")
        )

        explanation["top_negative"] = (
            shap_df[shap_df["shap_value"] < 0].head(5).to_dict("records")
        )

        return explanation
