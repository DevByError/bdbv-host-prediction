"""
Geospatial visualization for risk mapping.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class RiskMapGenerator:
    """
    Generate geospatial risk maps.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("results/maps")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Color schemes
        self.risk_colors = {
            "high": "#C73E1D",
            "medium": "#F18F01",
            "low": "#3D8C40",
            "no_risk": "#EBEBEB",
        }

    def plot_risk_map(
        self,
        risk_data: pd.DataFrame,
        title: str = "BDBV Spillover Risk",
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot geospatial risk map.

        Args:
            risk_data: DataFrame with 'latitude', 'longitude', 'risk'
            title: Plot title
            output_path: Output path

        Returns:
            plt.Figure: Figure object
        """
        fig, ax = plt.subplots(figsize=(14, 10))

        # Create scatter plot
        scatter = ax.scatter(
            risk_data["longitude"],
            risk_data["latitude"],
            c=risk_data["risk"],
            cmap="YlOrRd",
            s=risk_data.get("size", 100),
            alpha=0.7,
            edgecolors="white",
            linewidth=1,
        )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Risk Probability", fontsize=12, fontweight="bold")

        # Set labels
        ax.set_xlabel("Longitude", fontsize=12, fontweight="bold")
        ax.set_ylabel("Latitude", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=16, fontweight="bold")

        # Add grid
        ax.grid(True, alpha=0.2)

        # Save
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Risk map saved to {output_path}")

        return fig

    def plot_heatmap(
        self,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        risk_grid: np.ndarray,
        title: str = "BDBV Risk Heatmap",
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot risk heatmap.

        Args:
            lat_grid: Latitude grid
            lon_grid: Longitude grid
            risk_grid: Risk values grid
            title: Plot title
            output_path: Output path

        Returns:
            plt.Figure: Figure object
        """
        fig, ax = plt.subplots(figsize=(14, 10))

        # Plot heatmap
        contour = ax.contourf(
            lon_grid, lat_grid, risk_grid, levels=20, cmap="YlOrRd", alpha=0.8
        )

        # Add contour lines
        ax.contour(
            lon_grid,
            lat_grid,
            risk_grid,
            levels=[0.3, 0.5, 0.7],
            colors="black",
            linewidths=0.5,
            alpha=0.3,
        )

        # Colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label("Risk Probability", fontsize=12, fontweight="bold")

        ax.set_xlabel("Longitude", fontsize=12, fontweight="bold")
        ax.set_ylabel("Latitude", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=16, fontweight="bold")

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Heatmap saved to {output_path}")

        return fig

    def plot_risk_timeline(
        self,
        risk_data: pd.DataFrame,
        title: str = "Risk Over Time",
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot risk timeline.

        Args:
            risk_data: DataFrame with 'date' and 'risk'
            title: Plot title
            output_path: Output path

        Returns:
            plt.Figure: Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Sort by date
        risk_data = risk_data.sort_values("date")

        # Plot
        ax.plot(
            risk_data["date"],
            risk_data["risk"],
            linewidth=2,
            color="#2E86AB",
            marker="o",
            markersize=6,
        )

        # Add trend line
        z = np.polyfit(range(len(risk_data)), risk_data["risk"], 1)
        trend = np.polyval(z, range(len(risk_data)))
        ax.plot(
            risk_data["date"],
            trend,
            linestyle="--",
            color="red",
            alpha=0.7,
            label="Trend",
        )

        ax.set_xlabel("Date", fontsize=12, fontweight="bold")
        ax.set_ylabel("Risk Probability", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.2)

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Timeline saved to {output_path}")

        return fig
