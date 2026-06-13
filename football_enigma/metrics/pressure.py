"""Pressure-resistance: how a player's passing holds up when pressed.

Uses StatsBomb's per-event ``under_pressure`` flag (canonical pass tables
from other sources have it False everywhere — only compute this on
StatsBomb-derived data). For each player we split passes into pressured /
unpressured and compare completion rate and, optionally, xT added per pass.

The headline number is ``completion_drop``: completion% unpressured minus
completion% pressured. Elite press-resistant players hold this near zero.
"""

import pandas as pd

from football_enigma.metrics.xt import XTModel


def pressure_split(
    passes: pd.DataFrame,
    xt_model: XTModel | None = None,
    min_pressured: int = 10,
) -> pd.DataFrame:
    """Per-player pressured/unpressured passing comparison.

    `passes`: canonical pass table (data.schema) with `under_pressure`.
    Players with fewer than `min_pressured` pressured passes are dropped —
    a completion% on a handful of passes is noise.
    """
    passes = passes.copy()
    passes["under_pressure"] = passes["under_pressure"].astype(bool)
    passes["outcome"] = passes["outcome"].astype(bool)
    if xt_model is not None:
        passes["xt_value"] = xt_model.value(passes)

    def summarise(group: pd.DataFrame) -> pd.Series:
        pressured = group[group["under_pressure"]]
        calm = group[~group["under_pressure"]]
        row = {
            "passes": len(group),
            "pressured": len(pressured),
            "comp_pct": group["outcome"].mean() * 100,
            "comp_pct_pressured": pressured["outcome"].mean() * 100
            if len(pressured)
            else float("nan"),
            "comp_pct_unpressured": calm["outcome"].mean() * 100
            if len(calm)
            else float("nan"),
        }
        row["completion_drop"] = (
            row["comp_pct_unpressured"] - row["comp_pct_pressured"]
        )
        if xt_model is not None:
            row["xt_per_pass_pressured"] = (
                pressured["xt_value"].mean() if len(pressured) else float("nan")
            )
            row["xt_per_pass_unpressured"] = (
                calm["xt_value"].mean() if len(calm) else float("nan")
            )
        return pd.Series(row)

    out = (
        passes.groupby("player", observed=True)
        .apply(summarise, include_groups=False)
        .reset_index()
    )
    return out[out["pressured"] >= min_pressured].reset_index(drop=True)
