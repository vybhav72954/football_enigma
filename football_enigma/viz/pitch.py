"""Pitch figures in the house style (mplsoccer wrappers).

Canonical-coordinate data (105x68 meters) uses `meter_pitch()`;
native StatsBomb data (120x80) uses `statsbomb_pitch()`.
"""

import matplotlib.pyplot as plt
import pandas as pd
from mplsoccer import Pitch

from football_enigma.viz.theme import COLORS


def _pitch(pitch_type: str, **kwargs) -> Pitch:
    return Pitch(
        pitch_type=pitch_type,
        pitch_color=COLORS["background"],
        line_color=COLORS["pitch_line"],
        linewidth=1.2,
        **kwargs,
    )


def meter_pitch(**kwargs) -> Pitch:
    return _pitch("custom", pitch_length=105, pitch_width=68, **kwargs)


def statsbomb_pitch(**kwargs) -> Pitch:
    return _pitch("statsbomb", **kwargs)


def draw(pitch: Pitch, figsize: tuple[float, float] = (10, 7)):
    fig, ax = pitch.draw(figsize=figsize)
    fig.set_facecolor(COLORS["background"])
    return fig, ax


def pass_arrows(
    pitch: Pitch,
    ax: plt.Axes,
    passes: pd.DataFrame,
    color: str | None = None,
    alpha: float = 0.75,
    width: float = 1.6,
) -> None:
    """Arrow per pass from (x, y) to (end_x, end_y)."""
    pitch.arrows(
        passes["x"], passes["y"], passes["end_x"], passes["end_y"],
        ax=ax,
        color=color or COLORS["subject"],
        alpha=alpha,
        width=width,
        headwidth=6,
        headlength=6,
    )
