"""
LIME analysis for local explanations.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union
import logging

logger = logging.getLogger(__name__)


class LIMEExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations) analyzer.
    """

    def __init__(
        self, model, feature_names: List[str], class_names: Optional[List[str]] = None
    ):
        """
        Initialize LIME explainer.

        Args:
            model: Trained model
            feature_names: List of feature names
            class_names: List of class names
        """
        try:
            import lime.lime_tabular
        except ImportError:
            raise ImportError("LIME not installed. Run: pip install lime")

        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names or ["Negative", "Positive"]
        self.explainer = None

    def fit(self, X_train: np.ndarray) -> "LIMEExplainer":
        """
        Fit LIME explainer on training data.

        Args:
            X_train: Training data

        Returns:
            LIMEExplainer: Self
        """
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode="classification",
        )
        return self

    def explain_instance(self, instance: np.ndarray) -> Dict:
        """
        Explain a single instance.

        Args:
            instance: Instance to explain

        Returns:
            Dict: Explanation details
        """
        if self.explainer is None:
            raise ValueError("Must call fit first")

        # Get explanation
        explanation = self.explainer.explain_instance(
            instance, self.model.predict_proba, num_features=10, top_labels=1
        )

        # Extract features
        local_exp = explanation.local_exp
        if len(local_exp) > 0:
            class_idx = list(local_exp.keys())[0]
            features = local_exp[class_idx]
        else:
            features = []

        # Convert to DataFrame
        explanation_df = pd.DataFrame(features, columns=["feature_idx", "weight"])
        explanation_df["feature"] = [
            self.feature_names[int(i)] for i in explanation_df["feature_idx"]
        ]
        explanation_df = explanation_df[["feature", "weight"]]

        return {
            "features": explanation_df,
            "prediction": explanation.predict_proba[0],
            "intercept": explanation.intercept[0],
        }

    def plot_explanation(self, instance: np.ndarray, output_path: Optional[str] = None):
        """
        Generate LIME explanation plot.

        Args:
            instance: Instance to explain
            output_path: Path to save figure
        """
        if self.explainer is None:
            raise ValueError("Must call fit first")

        import matplotlib.pyplot as plt

        explanation = self.explainer.explain_instance(
            instance, self.model.predict_proba, num_features=10, top_labels=1
        )

        fig = explanation.as_pyplot_figure()

        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")

        return fig
