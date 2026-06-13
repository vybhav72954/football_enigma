"""Pass-table aggregates: the explicit, documented versions of the
"advanced stats" this project can no longer source from FBref.

All functions take canonical pass tables (data.schema: meters on 105x68,
`outcome` True = completed). Definitions are deliberately simple and stated
in full — when published, the post links here rather than to a black box.

Definitions
-----------
progressive pass : completed pass that brings the ball >= 25% closer to the
                   centre of the opponent goal (105, 34), and >= 5 m closer
                   in absolute terms (filters taps around the centre circle)
final-third pass : completed pass from outside into the final third
                   (x >= 70 m)
switch           : completed pass with lateral travel >= 30 m
"""

import numpy as np
import pandas as pd

from football_enigma.data.schema import PITCH_LENGTH, PITCH_WIDTH

GOAL = np.array([PITCH_LENGTH, PITCH_WIDTH / 2])
FINAL_THIRD_X = 2 * PITCH_LENGTH / 3
SWITCH_LATERAL_M = 30.0


def _dist_to_goal(x: pd.Series, y: pd.Series) -> np.ndarray:
    return np.hypot(GOAL[0] - x.to_numpy(float), GOAL[1] - y.to_numpy(float))


def flag_progressive(passes: pd.DataFrame) -> pd.Series:
    d_start = _dist_to_goal(passes["x"], passes["y"])
    d_end = _dist_to_goal(passes["end_x"], passes["end_y"])
    closer = d_start - d_end
    return pd.Series(
        passes["outcome"].astype(bool)
        & (closer >= 0.25 * d_start)
        & (closer >= 5.0),
        index=passes.index,
        name="progressive",
    )


def flag_final_third(passes: pd.DataFrame) -> pd.Series:
    return pd.Series(
        passes["outcome"].astype(bool)
        & (passes["x"] < FINAL_THIRD_X)
        & (passes["end_x"] >= FINAL_THIRD_X),
        index=passes.index,
        name="final_third",
    )


def flag_switch(passes: pd.DataFrame) -> pd.Series:
    return pd.Series(
        passes["outcome"].astype(bool)
        & ((passes["end_y"] - passes["y"]).abs() >= SWITCH_LATERAL_M),
        index=passes.index,
        name="switch",
    )


def player_pass_aggregates(
    passes: pd.DataFrame, minutes: pd.Series | None = None
) -> pd.DataFrame:
    """Per-player totals (and per-90s if a minutes Series, indexed by
    player, is provided)."""
    passes = passes.copy()
    passes["progressive"] = flag_progressive(passes)
    passes["final_third"] = flag_final_third(passes)
    passes["switch"] = flag_switch(passes)
    passes["completed"] = passes["outcome"].astype(bool)
    fwd = (passes["end_x"] - passes["x"]).clip(lower=0)
    passes["forward_meters"] = np.where(passes["completed"], fwd, 0.0)

    agg = passes.groupby("player", observed=True).agg(
        passes=("completed", "size"),
        completed=("completed", "sum"),
        progressive=("progressive", "sum"),
        final_third=("final_third", "sum"),
        switches=("switch", "sum"),
        forward_meters=("forward_meters", "sum"),
    )
    agg["comp_pct"] = agg["completed"] / agg["passes"] * 100

    if minutes is not None:
        agg = agg.join(minutes.rename("minutes"), how="left")
        per90 = agg["minutes"] / 90
        for col in ("passes", "completed", "progressive", "final_third",
                    "switches", "forward_meters"):
            agg[f"{col}_p90"] = agg[col] / per90
    return agg.reset_index()
