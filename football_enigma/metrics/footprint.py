"""Spatial footprint of a player's on-ball actions.

The territorial-economy measure: how spread out a player's actions are. ``spread``
is the *standard distance* — sqrt(var_x + var_y) in meters — of all action start
locations, i.e. the radius of the cloud of where a player operates. A disciplined
positional pivot holds a small territory (small spread); a roaming creator a large
one. Pairs with invariance-in-time: the constant who never moves, in space.

Takes a canonical actions table (data.schema, meters) so it is source-agnostic.
"""

import numpy as np
import pandas as pd


def action_dispersion(
    actions: pd.DataFrame, min_actions: int = 1
) -> pd.DataFrame:
    """Per-player action-location centroid and spread (meters).

    One row per player with at least ``min_actions`` located actions:
    ``actions`` (count), ``cx``/``cy`` (centroid), ``std_x``/``std_y``, and
    ``spread`` = sqrt(std_x**2 + std_y**2), the standard distance.
    """
    df = actions.dropna(subset=["x", "y"])
    g = df.groupby("player", observed=True)
    out = g.agg(
        actions=("x", "size"),
        cx=("x", "mean"),
        cy=("y", "mean"),
        std_x=("x", lambda s: float(np.std(s, ddof=0))),
        std_y=("y", lambda s: float(np.std(s, ddof=0))),
    )
    out["spread"] = np.hypot(out["std_x"], out["std_y"])
    out = out[out["actions"] >= min_actions]
    return out.reset_index()
