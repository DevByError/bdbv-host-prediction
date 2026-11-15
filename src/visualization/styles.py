"""Shared matplotlib styling so every figure matches."""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional


class FigureStyle:
    # colourblind-safe, same palette as the paper figures
    COLORS = {
        "primary": "#2E86AB",  # Blue
        "secondary": "#A23B72",  # Purple
        "success": "#3D8C40",  # Green
        "warning": "#F18F01",  # Orange
        "danger": "#C73E1D",  # Red
        "teal": "#1A8F8F",  # Teal
        "pink": "#D4729B",  # Pink
        "brown": "#8B5A2B",  # Brown
        "gray": "#7F8C8D",  # Gray
        "gold": "#D4A017",  # Gold
    }

    FONT_FAMILY = "Arial"
    FONT_SIZES = {
        "title": 16,
        "subtitle": 14,
        "label": 12,
        "tick": 10,
        "legend": 10,
        "annotation": 9,
        "small": 8,
    }

    @classmethod
    def setup(cls, style: str = "seaborn-v0_8-whitegrid"):
        plt.style.use(style)
        sns.set_style(style)

        sns.set_palette(
            [
                cls.COLORS["primary"],
                cls.COLORS["secondary"],
                cls.COLORS["success"],
                cls.COLORS["warning"],
                cls.COLORS["danger"],
                cls.COLORS["teal"],
                cls.COLORS["pink"],
                cls.COLORS["brown"],
                cls.COLORS["gray"],
            ]
        )

        plt.rcParams["font.family"] = cls.FONT_FAMILY
        # type 42 keeps text as text in the PDF/EPS, journals ask for it
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["ps.fonttype"] = 42

        plt.rcParams["figure.dpi"] = 300
        plt.rcParams["savefig.dpi"] = 600
        plt.rcParams["savefig.bbox"] = "tight"
        plt.rcParams["savefig.facecolor"] = "white"
        plt.rcParams["savefig.edgecolor"] = "none"

        plt.rcParams["axes.titlesize"] = cls.FONT_SIZES["title"]
        plt.rcParams["axes.labelsize"] = cls.FONT_SIZES["label"]
        plt.rcParams["axes.titleweight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"

        plt.rcParams["xtick.labelsize"] = cls.FONT_SIZES["tick"]
        plt.rcParams["ytick.labelsize"] = cls.FONT_SIZES["tick"]

        plt.rcParams["legend.fontsize"] = cls.FONT_SIZES["legend"]
        plt.rcParams["legend.frameon"] = True
        plt.rcParams["legend.fancybox"] = True

        plt.rcParams["grid.alpha"] = 0.2
        plt.rcParams["grid.linestyle"] = "--"

    @classmethod
    def get_color(cls, name: str) -> str:
        return cls.COLORS.get(name, cls.COLORS["primary"])

    @classmethod
    def get_colors(cls, n: int) -> list:
        palette = list(cls.COLORS.values())
        if n <= len(palette):
            return palette[:n]
        else:
            # ran out of named colours, interpolate the rest
            import numpy as np
            from matplotlib.colors import to_rgb

            base_colors = [to_rgb(c) for c in palette]
            import colorsys

            additional = []
            for i in range(n - len(palette)):
                idx = i % (len(palette) - 1)
                t = (i // (len(palette) - 1)) / (n // (len(palette) - 1) + 1)

                c1 = base_colors[idx]
                c2 = base_colors[(idx + 1) % len(base_colors)]

                interpolated = tuple(c1[j] + t * (c2[j] - c1[j]) for j in range(3))
                additional.append(interpolated)

            return palette + additional
