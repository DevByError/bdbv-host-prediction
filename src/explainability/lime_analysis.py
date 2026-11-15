"""LIME local explanations for single predictions."""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union
import logging

logger = logging.getLogger(__name__)


class LIMEExplainer:
    def __init__(
        self, model, feature_names: List[str], class_names: Optional[List[str]] = None
    ):
        try:
            import lime.lime_tabular
        except ImportError:
            raise ImportError("LIME not installed. Run: pip install lime")

        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names or ["Negative", "Positive"]
        self.explainer = None

    def fit(self, X_train: np.ndarray) -> "LIMEExplainer":
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode="classification",
        )
        return self

    def explain_instance(self, instance: np.ndarray) -> Dict:
        """Keys: features (feature/weight frame), prediction_proba, explained_class, intercept.

        The weights describe explained_class, which is whatever the model predicted - not always class 1.
        """
        if self.explainer is None:
            raise ValueError("Must call fit first")

        explanation = self.explainer.explain_instance(
            instance, self.model.predict_proba, num_features=10, top_labels=1
        )

        local_exp = explanation.local_exp
        if len(local_exp) > 0:
            # with top_labels=1 the only key is the predicted class, which may be 1;
            # do not hardcode 0 here
            class_idx = list(local_exp.keys())[0]
            features = local_exp[class_idx]
            intercept = explanation.intercept[class_idx]
        else:
            class_idx = None
            features = []
            intercept = 0.0

        explanation_df = pd.DataFrame(features, columns=["feature_idx", "weight"])
        explanation_df["feature"] = [
            self.feature_names[int(i)] for i in explanation_df["feature_idx"]
        ]
        explanation_df = explanation_df[["feature", "weight"]]
        explanation_df = explanation_df.reindex(
            explanation_df["weight"].abs().sort_values(ascending=False).index
        ).reset_index(drop=True)

        return {
            "features": explanation_df,
            "prediction_proba": np.asarray(explanation.predict_proba),
            "explained_class": class_idx,
            "intercept": intercept,
        }

    def plot_explanation(self, instance: np.ndarray, output_path: Optional[str] = None):
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
