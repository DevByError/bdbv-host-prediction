"""
Visualization module for BDBV Host Prediction.

This module provides publication-quality visualizations including:
- Performance comparison plots
- Confusion matrices
- ROC and PR curves
- Feature importance plots
- SHAP summary plots
- Risk maps
- Interactive dashboards
"""

from .publication_figures import FigureGenerator
from .interactive_dashboard import InteractiveDashboard
from .maps import RiskMapGenerator
from .styles import FigureStyle

__all__ = ["FigureGenerator", "InteractiveDashboard", "RiskMapGenerator", 
           "FigureStyle"]
