"""Football Enigma house style.

One dark, restrained look across every figure in the series: near-black
canvas, quiet grey field of "everyone else", and a single signal colour
reserved for the subject of the post. Apply with `apply_theme()` once per
notebook/script; never restyle figures locally.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

COLORS = {
    "background": "#0D1117",
    "panel": "#161B22",
    "text": "#E6EDF3",
    "muted": "#8B949E",
    "grid": "#21262D",
    "field": "#6E7681",      # the crowd: every non-subject player/point
    "subject": "#F2A33C",    # the enigma: one amber signal colour
    "accent2": "#58A6FF",    # secondary comparisons (sparingly)
    "negative": "#F85149",
    "pitch_line": "#30363D",
}

FOOTER = "Football Enigma"


def apply_theme() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": COLORS["background"],
            "axes.facecolor": COLORS["background"],
            "savefig.facecolor": COLORS["background"],
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["text"],
            "axes.titlecolor": COLORS["text"],
            "text.color": COLORS["text"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "grid.color": COLORS["grid"],
            "axes.grid": True,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def credit(ax: plt.Axes, source: str) -> None:
    """Mandatory data attribution + series footer, bottom-left of figure."""
    ax.figure.text(
        0.01,
        0.005,
        f"{FOOTER}  |  Data: {source}",
        fontsize=8,
        color=COLORS["muted"],
        ha="left",
        va="bottom",
    )


def highlight_point(ax: plt.Axes, x: float, y: float, label: str) -> None:
    """Mark the subject on a crowd scatter: amber dot + halo + name."""
    ax.scatter([x], [y], s=160, color=COLORS["subject"], zorder=5,
               edgecolors=COLORS["text"], linewidths=1.2)
    ax.annotate(
        label,
        (x, y),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=12,
        fontweight="bold",
        color=COLORS["subject"],
        zorder=6,
    )
