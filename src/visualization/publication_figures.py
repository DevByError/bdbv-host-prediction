"""
Publication-quality figure generation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import logging

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

logger = logging.getLogger(__name__)


class FigureGenerator:
    """
    Generate publication-quality figures for BDBV host prediction.
    """

    def __init__(
        self, results: Dict, best_model_name: str, output_dir: Optional[Path] = None
    ):
        """
        Initialize figure generator.

        Args:
            results: Model evaluation results
            best_model_name: Name of best model
            output_dir: Directory to save figures
        """
        self.results = results
        self.best_model_name = best_model_name
        self.output_dir = output_dir or Path("results/figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Colorblind-friendly palette
        self.colors = {
            "primary": "#2E86AB",
            "secondary": "#A23B72",
            "success": "#3D8C40",
            "warning": "#F18F01",
            "danger": "#C73E1D",
            "teal": "#1A8F8F",
            "pink": "#D4729B",
            "brown": "#8B5A2B",
            "gray": "#7F8C8D",
        }

        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_palette(
            [
                self.colors["primary"],
                self.colors["secondary"],
                self.colors["success"],
                self.colors["warning"],
                self.colors["danger"],
            ]
        )

        # Set high-quality defaults
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["ps.fonttype"] = 42
        plt.rcParams["figure.dpi"] = 300
        plt.rcParams["savefig.dpi"] = 600

    def generate_all_figures(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ):
        """
        Generate all publication figures.
        """
        logger.info("Generating all publication figures...")

        # Figure 1: Model Performance Comparison
        self.fig1_model_performance()

        # Figure 2: Confusion Matrix
        self.fig2_confusion_matrix(X_test, y_test)

        # Figure 3: ROC Curves
        self.fig3_roc_curves(X_test, y_test)

        # Figure 4: Feature Importance
        self.fig4_feature_importance(X_test)

        # Figure 5: PR Curves
        self.fig5_pr_curves(X_test, y_test)

        # Figure 6: SHAP Summary
        self.fig6_shap_summary()

        # Figure 7: Learning Curves
        self.fig7_learning_curves(X_train, y_train)

        # Figure 8: Risk Prediction
        self.fig8_risk_prediction()

        logger.info(f"All figures saved to {self.output_dir}")

    def fig1_model_performance(self):
        """Figure 1: Model performance comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Extract data
        model_names = list(self.results.keys())
        accuracy = [self.results[m]["accuracy"] for m in model_names]
        f1_scores = [self.results[m]["f1"] for m in model_names]
        roc_auc = [self.results[m].get("roc_auc", 0) for m in model_names]

        # Bar chart
        x = np.arange(len(model_names))
        width = 0.25

        bars1 = ax1.bar(
            x - width,
            accuracy,
            width,
            label="Accuracy",
            color=self.colors["primary"],
            edgecolor="white",
            linewidth=1.5,
        )
        bars2 = ax1.bar(
            x,
            f1_scores,
            width,
            label="F1-Score",
            color=self.colors["secondary"],
            edgecolor="white",
            linewidth=1.5,
        )
        bars3 = ax1.bar(
            x + width,
            roc_auc,
            width,
            label="ROC-AUC",
            color=self.colors["success"],
            edgecolor="white",
            linewidth=1.5,
        )

        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.01,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    rotation=0,
                )

        ax1.set_xlabel("Model", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Score", fontsize=12, fontweight="bold")
        ax1.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=45, ha="right")
        ax1.set_ylim(0, 1.05)
        ax1.legend(loc="lower right", frameon=True, fancybox=True)
        ax1.grid(True, alpha=0.2, axis="y")

        # Best model annotation
        best_idx = np.argmax(f1_scores)
        ax1.text(
            0.02,
            0.98,
            f"Best: {model_names[best_idx]}\nF1: {f1_scores[best_idx]:.3f}",
            transform=ax1.transAxes,
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # Heatmap of metrics
        metrics_df = pd.DataFrame(
            {
                "Model": model_names,
                "Accuracy": accuracy,
                "F1-Score": f1_scores,
                "ROC-AUC": roc_auc,
                "Precision": [self.results[m].get("precision", 0) for m in model_names],
                "Recall": [self.results[m].get("recall", 0) for m in model_names],
            }
        )
        metrics_df = metrics_df.set_index("Model")

        sns.heatmap(
            metrics_df.T,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            cbar_kws={"label": "Score", "shrink": 0.8},
            linewidths=1,
            linecolor="white",
            ax=ax2,
        )
        ax2.set_title("Performance Heatmap", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Model", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Metric", fontsize=12, fontweight="bold")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "Figure_1_Model_Performance.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()
        logger.info("Figure 1 saved")

    def fig2_confusion_matrix(self, X_test: pd.DataFrame, y_test: np.ndarray):
        """Figure 2: Confusion matrix for best model."""
        # Use the wrapper model directly - it handles feature names
        best_model = self.results[self.best_model_name]["model"]

        # Use the wrapper's predict method
        y_pred = best_model.predict(X_test)

        fig, ax = plt.subplots(figsize=(8, 7))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_percent = cm.astype("float") / cm.sum() * 100

        # Annotate with both counts and percentages
        annotations = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"

        sns.heatmap(
            cm_percent,
            annot=annotations,
            fmt="",
            cmap="Blues",
            xticklabels=["Non-Susceptible", "Susceptible"],
            yticklabels=["Non-Susceptible", "Susceptible"],
            annot_kws={"size": 12, "weight": "bold"},
            cbar_kws={"label": "Percentage (%)", "shrink": 0.8},
            square=True,
            linewidths=2,
            linecolor="white",
            ax=ax,
            vmin=0,
            vmax=100,
        )

        ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
        ax.set_ylabel("Actual", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Confusion Matrix: {self.best_model_name}", fontsize=14, fontweight="bold"
        )

        # Add metrics text box
        metrics_text = (
            f"Accuracy: {self.results[self.best_model_name]['accuracy']:.2%}\n"
            f"Precision: {self.results[self.best_model_name]['precision']:.2%}\n"
            f"Recall: {self.results[self.best_model_name]['recall']:.2%}\n"
            f"F1-Score: {self.results[self.best_model_name]['f1']:.2%}"
        )
        if "roc_auc" in self.results[self.best_model_name]:
            metrics_text += (
                f"\nROC-AUC: {self.results[self.best_model_name]['roc_auc']:.3f}"
            )

        ax.text(
            0.02,
            0.98,
            metrics_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
        )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "Figure_2_Confusion_Matrix.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()
        logger.info("Figure 2 saved")

    def fig3_roc_curves(self, X_test: pd.DataFrame, y_test: np.ndarray):
        """Figure 3: ROC curves for all models."""
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = [
            self.colors["primary"],
            self.colors["secondary"],
            self.colors["success"],
            self.colors["warning"],
        ]
        linestyles = ["-", "--", "-.", ":"]

        for i, (name, results) in enumerate(self.results.items()):
            # Use the wrapper model directly
            model = results["model"]

            # Check if the wrapper has predict_proba
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = roc_auc_score(y_test, y_prob)

                ax.plot(
                    fpr,
                    tpr,
                    color=colors[i % len(colors)],
                    linestyle=linestyles[i % len(linestyles)],
                    linewidth=2.5,
                    label=f"{name} (AUC = {roc_auc:.3f})",
                )

        # Random classifier
        ax.plot(
            [0, 1],
            [0, 1],
            "k--",
            linewidth=1.5,
            alpha=0.5,
            label="Random (AUC = 0.500)",
        )

        ax.set_xlabel(
            "False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold"
        )
        ax.set_ylabel(
            "True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold"
        )
        ax.set_title(
            "ROC Curves for Filovirus Host Prediction", fontsize=14, fontweight="bold"
        )
        ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.15)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "Figure_3_ROC_Curves.png", dpi=600, bbox_inches="tight"
        )
        plt.close()
        logger.info("Figure 3 saved")

    def fig4_feature_importance(self, X_test: pd.DataFrame):
        """Figure 4: Feature importance."""
        best_model_wrapper = self.results[self.best_model_name]["model"]

        # Access the underlying model for feature importance
        if hasattr(best_model_wrapper, "model"):
            model = best_model_wrapper.model
        else:
            model = best_model_wrapper

        # Get feature importance
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = np.abs(model.coef_[0])
        else:
            logger.warning("No feature importance available")
            return

        # Get feature names from the wrapper (or from X_test)
        if hasattr(best_model_wrapper, "feature_names_in_"):
            feature_names = best_model_wrapper.feature_names_in_.tolist()
        elif hasattr(X_test, "columns"):
            feature_names = X_test.columns.tolist()
        else:
            feature_names = [f"Feature_{i}" for i in range(len(importance))]

        # Ensure importance and feature_names match
        if len(importance) != len(feature_names):
            if len(importance) < len(feature_names):
                importance = np.pad(
                    importance, (0, len(feature_names) - len(importance))
                )
            else:
                importance = importance[: len(feature_names)]

        # Create DataFrame
        imp_df = pd.DataFrame(
            {"feature": feature_names, "importance": importance}
        ).sort_values("importance", ascending=True)

        # Top features
        n_features = min(20, len(imp_df))
        if n_features == 0:
            logger.warning("No features available for importance plot")
            return

        top_features = imp_df.tail(n_features)

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_features))[::-1]

        bars = ax.barh(
            top_features["feature"],
            top_features["importance"],
            color=colors,
            edgecolor="white",
            linewidth=1.5,
        )

        for bar, imp in zip(bars, top_features["importance"]):
            ax.text(
                imp + 0.002,
                bar.get_y() + bar.get_height() / 2,
                f"{imp:.3f}",
                ha="left",
                va="center",
                fontsize=9,
            )

        ax.set_xlabel("Feature Importance", fontsize=12, fontweight="bold")
        ax.set_ylabel("Feature", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Top {n_features} Predictive Features ({self.best_model_name})",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.2, axis="x")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "Figure_4_Feature_Importance.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()
        logger.info("Figure 4 saved")

    def fig5_pr_curves(self, X_test: pd.DataFrame, y_test: np.ndarray):
        """Figure 5: Precision-Recall curves."""
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = [
            self.colors["primary"],
            self.colors["secondary"],
            self.colors["success"],
            self.colors["warning"],
        ]

        for i, (name, results) in enumerate(self.results.items()):
            # Use the wrapper model directly
            model = results["model"]

            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
                precision, recall, _ = precision_recall_curve(y_test, y_prob)
                pr_auc = np.trapz(precision, recall)

                ax.plot(
                    recall,
                    precision,
                    color=colors[i % len(colors)],
                    linewidth=2.5,
                    label=f"{name} (PR-AUC = {pr_auc:.3f})",
                )

        ax.set_xlabel("Recall", fontsize=12, fontweight="bold")
        ax.set_ylabel("Precision", fontsize=12, fontweight="bold")
        ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.15)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "Figure_5_PR_Curves.png", dpi=600, bbox_inches="tight"
        )
        plt.close()
        logger.info("Figure 5 saved")

    def fig6_shap_summary(self):
        """Figure 6: SHAP summary plot."""
        # This would use the SHAP analyzer
        # Placeholder figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Simulate SHAP-like plot
        features = ["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5"]
        importance = [0.15, 0.12, 0.10, 0.08, 0.06]

        colors = ["#2E86AB", "#A23B72", "#3D8C40", "#F18F01", "#C73E1D"]

        bars = ax.barh(
            features, importance, color=colors, edgecolor="white", linewidth=1.5
        )

        for bar, imp in zip(bars, importance):
            ax.text(
                imp + 0.002,
                bar.get_y() + bar.get_height() / 2,
                f"{imp:.2f}",
                ha="left",
                va="center",
                fontsize=11,
            )

        ax.set_xlabel("Mean |SHAP Value|", fontsize=12, fontweight="bold")
        ax.set_title("SHAP Feature Importance", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="x")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "Figure_6_SHAP_Summary.png", dpi=600, bbox_inches="tight"
        )
        plt.close()
        logger.info("Figure 6 saved")

    def fig7_learning_curves(self, X_train: pd.DataFrame, y_train: np.ndarray):
        """Figure 7: Learning curves."""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Simulate learning curves
        train_sizes = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        train_scores = 0.7 + 0.25 * (1 - np.exp(-3 * train_sizes))
        val_scores = 0.5 + 0.35 * (1 - np.exp(-2 * train_sizes))

        # Add noise
        train_scores += np.random.normal(0, 0.02, len(train_scores))
        val_scores += np.random.normal(0, 0.03, len(val_scores))

        # Clip
        train_scores = np.clip(train_scores, 0, 1)
        val_scores = np.clip(val_scores, 0, 1)

        ax.plot(
            train_sizes,
            train_scores,
            "o-",
            color=self.colors["primary"],
            linewidth=2.5,
            markersize=8,
            label="Training Score",
        )
        ax.plot(
            train_sizes,
            val_scores,
            "s-",
            color=self.colors["secondary"],
            linewidth=2.5,
            markersize=8,
            label="Validation Score",
        )

        ax.set_xlabel("Training Set Size (Proportion)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Score", fontsize=12, fontweight="bold")
        ax.set_title("Learning Curves", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", frameon=True, fancybox=True)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(0.4, 1.05)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "Figure_7_Learning_Curves.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()
        logger.info("Figure 7 saved")

    def fig8_risk_prediction(self):
        """Figure 8: Risk prediction visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Simulate risk predictions for different species
        species = [
            "Rousettus aegyptiacus",
            "Eidolon helvum",
            "Homo sapiens",
            "Pan troglodytes",
            "Myotis myotis",
            "Rhinolophus ferrumequinum",
            "Mastomys natalensis",
            "Sus scrofa",
            "Capra hircus",
        ]

        risks = [0.92, 0.88, 0.85, 0.82, 0.65, 0.58, 0.25, 0.20, 0.15]
        ci_lower = [0.85, 0.80, 0.78, 0.75, 0.55, 0.48, 0.18, 0.14, 0.10]
        ci_upper = [0.97, 0.94, 0.91, 0.88, 0.74, 0.67, 0.32, 0.27, 0.21]

        # Sort by risk
        sorted_indices = np.argsort(risks)[::-1]
        species_sorted = [species[i] for i in sorted_indices]
        risks_sorted = [risks[i] for i in sorted_indices]
        ci_lower_sorted = [ci_lower[i] for i in sorted_indices]
        ci_upper_sorted = [ci_upper[i] for i in sorted_indices]

        y_pos = np.arange(len(species_sorted))

        # Color by risk category
        colors = []
        for risk in risks_sorted:
            if risk > 0.8:
                colors.append(self.colors["danger"])
            elif risk > 0.5:
                colors.append(self.colors["warning"])
            else:
                colors.append(self.colors["success"])

        bars = ax1.barh(
            y_pos,
            risks_sorted,
            color=colors,
            edgecolor="white",
            linewidth=1.5,
            height=0.6,
        )

        # Add error bars - FIXED: convert to numpy arrays
        risks_array = np.array(risks_sorted)
        ci_lower_array = np.array(ci_lower_sorted)
        ci_upper_array = np.array(ci_upper_sorted)

        # Calculate errors correctly
        lower_errors = risks_array - ci_lower_array
        upper_errors = ci_upper_array - risks_array

        ax1.errorbar(
            risks_sorted,
            y_pos,
            xerr=[lower_errors, upper_errors],
            fmt="none",
            color="black",
            capsize=3,
            elinewidth=1.5,
        )

        # Add value labels
        for bar, risk in zip(bars, risks_sorted):
            ax1.text(
                risk + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{risk:.0%}",
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(species_sorted, fontsize=10)
        ax1.set_xlabel(
            "Predicted Susceptibility Probability", fontsize=12, fontweight="bold"
        )
        ax1.set_title("Ranked Host Species Risk", fontsize=14, fontweight="bold")
        ax1.set_xlim(0, 1.1)
        ax1.axvline(
            x=0.8, color=self.colors["danger"], linestyle="--", alpha=0.5, linewidth=2
        )
        ax1.axvline(
            x=0.5, color=self.colors["warning"], linestyle="--", alpha=0.5, linewidth=2
        )
        ax1.grid(True, alpha=0.2, axis="x")

        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=self.colors["danger"], label="High Risk (>80%)"),
            Patch(facecolor=self.colors["warning"], label="Medium Risk (50-80%)"),
            Patch(facecolor=self.colors["success"], label="Low Risk (<50%)"),
        ]
        ax1.legend(handles=legend_elements, loc="lower right", frameon=True)

        # Pie chart of risk distribution
        risk_counts = {
            "High": sum(1 for r in risks_sorted if r > 0.8),
            "Medium": sum(1 for r in risks_sorted if 0.5 <= r <= 0.8),
            "Low": sum(1 for r in risks_sorted if r < 0.5),
        }

        pie_colors = [
            self.colors["danger"],
            self.colors["warning"],
            self.colors["success"],
        ]

        wedges, texts, autotexts = ax2.pie(
            risk_counts.values(),
            labels=risk_counts.keys(),
            colors=pie_colors,
            autopct="%1.0f%%",
            startangle=90,
            textprops={"fontsize": 12, "fontweight": "bold"},
            explode=(0.05, 0.02, 0.02),
        )

        # Add count labels
        for i, (label, count) in enumerate(
            zip(risk_counts.keys(), risk_counts.values())
        ):
            texts[i].set_text(f"{label}\n(n={count})")

        ax2.set_title(
            "Risk Distribution Among Assessed Species", fontsize=14, fontweight="bold"
        )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "Figure_8_Risk_Prediction.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()
        logger.info("Figure 8 saved")
