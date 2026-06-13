"""Involvement cadence — the "measuring silence" metric.

For each player: the gaps (in seconds of match time) between consecutive
on-ball involvements. Two summaries per player:

- median gap: the player's normal state — how often the game runs
  through them.
- mean gap: punishes disappearances — one 10-minute silent spell drags
  the mean far above the median.

A player with BOTH low (down-left on the chart) is never out of the game:
the metronome profile. Strikers live up-right (rare, decisive touches);
players who vanish under pressure have low median but high mean.

Works on any event table with `player`, `period`, `minute`, `second` —
source-agnostic (WhoScored types filtered to on-ball involvements by the
caller, StatsBomb likewise).
"""

import pandas as pd


def involvement_gaps(events: pd.DataFrame) -> pd.DataFrame:
    """Per-player gap list within each match period (gaps never span
    half-time or matches). Expects one row per on-ball involvement with
    columns: match_id (or game_id), player, period, minute, second."""
    events = events.copy()
    match_col = "match_id" if "match_id" in events.columns else "game_id"
    events["t"] = events["minute"].astype(float) * 60 + events["second"].fillna(0).astype(float)
    events = events.sort_values([match_col, "period", "t"])
    events["gap"] = (
        events.groupby([match_col, "period", "player"], observed=True)["t"]
        .diff()
    )
    return events.dropna(subset=["gap"])[[match_col, "player", "period", "gap"]]


def cadence_summary(events: pd.DataFrame, min_involvements: int = 100) -> pd.DataFrame:
    """Median and mean involvement gap per player."""
    gaps = involvement_gaps(events)
    out = (
        gaps.groupby("player", observed=True)["gap"]
        .agg(median_gap="median", mean_gap="mean", involvements="size")
        .reset_index()
    )
    return out[out["involvements"] >= min_involvements].reset_index(drop=True)


def spell_gaps(events: pd.DataFrame, spell_break: float = 5.0) -> pd.DataFrame:
    """Gaps between involvement *spells*: consecutive events by the same
    player less than `spell_break` seconds apart count as one continuous
    on-ball moment (receive -> carry -> pass), not several involvements.

    This is what makes cadence comparable across providers: StatsBomb logs
    a receipt, carry and pass as separate rows seconds apart, Opta logs a
    single touch chain — raw `involvement_gaps` medians collapse on the
    former. Spell gaps measure the same thing in both: how long the game
    leaves a player alone.
    """
    gaps = involvement_gaps(events)
    return gaps[gaps["gap"] > spell_break].reset_index(drop=True)


def spell_cadence_summary(
    events: pd.DataFrame,
    min_spells: int = 100,
    spell_break: float = 5.0,
) -> pd.DataFrame:
    """Median and mean spell gap per player (source-comparable cadence)."""
    gaps = spell_gaps(events, spell_break)
    out = (
        gaps.groupby("player", observed=True)["gap"]
        .agg(median_gap="median", mean_gap="mean", spells="size")
        .reset_index()
    )
    return out[out["spells"] >= min_spells].reset_index(drop=True)
