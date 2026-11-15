"""Plotly figures for poking at results interactively."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = logging.getLogger(__name__)


class InteractiveDashboard:
    def __init__(self):
        if not PLOTLY_AVAILABLE:
            raise ImportError(
                "plotly is not installed. Install with: pip install plotly"
            )
        self.figures = {}

    def create_roc_plot(self, results: Dict) -> go.Figure:
        """One curve per model; expects fpr/tpr/roc_auc in each entry."""
        fig = go.Figure()

        for model_name, model_results in results.items():
            if "roc_auc" in model_results:
                fpr, tpr = model_results.get("fpr", []), model_results.get("tpr", [])
                auc = model_results["roc_auc"]

                fig.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode="lines",
                        name=f"{model_name} (AUC={auc:.3f})",
                        line=dict(width=2),
                    )
                )

        # diagonal reference
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random",
                line=dict(dash="dash", color="gray"),
            )
        )

        fig.update_layout(
            title="ROC Curves",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            hovermode="closest",
            width=800,
            height=600,
        )

        return fig

    def create_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]
    ) -> go.Figure:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)

        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16},
                colorscale="Blues",
                hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            )
        )

        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="True",
            width=600,
            height=500,
        )

        return fig

    def create_feature_importance(self, importance_df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=importance_df["importance"],
                y=importance_df["feature"],
                orientation="h",
                marker=dict(
                    color=importance_df["importance"],
                    colorscale="Viridis",
                    showscale=True,
                ),
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
            )
        )

        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=max(400, len(importance_df) * 25),
            width=800,
        )

        return fig

    def create_prediction_scatter(self, predictions: pd.DataFrame) -> go.Figure:
        """Scores coloured by risk band, with the band cut-offs drawn in."""
        fig = go.Figure()

        colors = {"High": "red", "Medium": "orange", "Low": "green"}

        for category in ["High", "Medium", "Low"]:
            subset = predictions[predictions["risk_category"] == category]
            if not subset.empty:
                fig.add_trace(
                    go.Scatter(
                        x=subset.index,
                        y=subset["susceptibility_score"],
                        mode="markers+text",
                        name=category,
                        text=(
                            subset["host_species"]
                            if "host_species" in subset.columns
                            else None
                        ),
                        textposition="top center",
                        marker=dict(size=15, color=colors[category], symbol="circle"),
                        hovertemplate="<b>%{text}</b><br>Score: %{y:.3f}<extra></extra>",
                    )
                )

        fig.add_hline(
            y=0.8,
            line_dash="dash",
            line_color="red",
            annotation_text="High Risk",
            annotation_position="top right",
        )
        fig.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="orange",
            annotation_text="Medium Risk",
            annotation_position="top right",
        )

        fig.update_layout(
            title="Species Susceptibility Predictions",
            xaxis_title="Sample Index",
            yaxis_title="Susceptibility Score",
            height=600,
            width=900,
            showlegend=True,
        )

        return fig

    def create_correlation_heatmap(self, X: pd.DataFrame) -> go.Figure:
        corr = X.corr()

        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale="RdBu",
                zmid=0,
                hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.2f}<extra></extra>",
            )
        )

        fig.update_layout(title="Feature Correlation Matrix", width=800, height=700)

        return fig

    def create_parallel_coordinates(
        self, df: pd.DataFrame, target_col: str = "infection_status"
    ) -> go.Figure:
        fig = go.Figure(
            data=go.Parcoords(
                line=dict(
                    color=df[target_col] if target_col in df.columns else None,
                    colorscale="Viridis",
                    showscale=True,
                ),
                dimensions=[
                    dict(
                        label=col,
                        values=df[col],
                        tickvals=np.percentile(df[col], [0, 25, 50, 75, 100]),
                    )
                    for col in df.columns[:10]  # more axes than this is unreadable
                ],
            )
        )

        fig.update_layout(
            title="Parallel Coordinates - Feature Patterns", width=1000, height=600
        )

        return fig

    def save_dashboard(self, output_path: str):
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "ROC Curves",
                "Confusion Matrix",
                "Feature Importance",
                "Predictions",
            ),
        )

        # traces still need wiring up to real results

        fig.update_layout(
            title="BDBV Host Prediction Dashboard", height=900, width=1200
        )

        fig.write_html(output_path)
        logger.info(f"Dashboard saved to {output_path}")
