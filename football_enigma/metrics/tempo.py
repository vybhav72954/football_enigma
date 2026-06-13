"""Time on ball — how long a player holds possession before releasing.

For every completed ball receipt: the seconds until that player's next
pass *within the same possession* (so turnovers, dispossessions and
half-time never bridge a receipt to an unrelated pass). The per-player
median is the tempo signature: one-touch metronomes live near 1–2 s,
ball-stoppers far above.

StatsBomb-specific — needs raw events with sub-second `timestamp`,
`possession` ids and `Ball Receipt*` rows. No Opta equivalent exists in
our data (Opta logs no receipts), so this metric is computed only where
StatsBomb covers the tournament.
"""

import pandas as pd

# a receipt left unanswered this long was not a usable possession moment
MAX_HOLD_SECONDS = 20.0


def _seconds(timestamp: pd.Series) -> pd.Series:
    t = pd.to_timedelta(timestamp)
    return t.dt.total_seconds()


def receipt_to_release(events: pd.DataFrame, match_id: int) -> pd.DataFrame:
    """One row per resolved receipt: player, possession, hold (seconds).

    Expects raw StatsBomb events for a single match (columns: type, player,
    period, possession, timestamp).
    """
    ev = events[["type", "player", "period", "possession", "timestamp"]].copy()
    ev["t"] = _seconds(ev["timestamp"])

    receipts = ev[
        (ev["type"] == "Ball Receipt*")
        & (
            events["ball_receipt_outcome"].isna()
            if "ball_receipt_outcome" in events.columns
            else True
        )
    ]
    passes = ev[ev["type"] == "Pass"]

    merged = receipts.merge(
        passes,
        on=["player", "period", "possession"],
        suffixes=("", "_pass"),
    )
    merged["hold"] = merged["t_pass"] - merged["t"]
    merged = merged[
        (merged["hold"] >= 0) & (merged["hold"] <= MAX_HOLD_SECONDS)
    ]
    # first pass after each receipt, one resolution per receipt
    merged = (
        merged.sort_values("hold")
        .groupby(["player", "period", "t"], observed=True, as_index=False)
        .first()
    )
    out = merged[["player", "possession", "hold"]].copy()
    out.insert(0, "match_id", match_id)
    return out


def tempo_summary(holds: pd.DataFrame, min_receipts: int = 100) -> pd.DataFrame:
    """Median and mean hold per player over concatenated receipt tables."""
    out = (
        holds.groupby("player", observed=True)["hold"]
        .agg(median_hold="median", mean_hold="mean", receipts="size")
        .reset_index()
    )
    return out[out["receipts"] >= min_receipts].reset_index(drop=True)
