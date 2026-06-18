"""Defensive event profile from raw StatsBomb events.

Separates a player's ball-winning into positional *reads* (interceptions and
loose-ball recoveries — winning possession without committing) and *commits*
(ground tackles, fouls conceded, and times dribbled past — going to ground or
being beaten). The deep-pivot archetype wins the ball by reading passing lanes,
so its profile is read-heavy and commit-light; the ball-winner archetype the box
score celebrates is the opposite.

Operates on the raw StatsBomb event frame (data.statsbomb.load_events), not the
canonical schema, because these event types carry none of the on-ball location
semantics the schema models. Counts are per match — sum across matches before
forming rates or shares.
"""

import pandas as pd

# interception_outcome values that mean the ball was retained
_INTERCEPTION_WON = {"Won", "Success", "Success In Play", "Success Out"}

# StatsBomb position names for the deepest defensive line
BACKLINE = {"Goalkeeper", "Center Back", "Left Center Back", "Right Center Back"}


def defensive_profile(events: pd.DataFrame) -> pd.DataFrame:
    """Per-player defensive-action counts for one match.

    One row per player with at least one counted action, columns:
    ``interceptions`` (won), ``recoveries`` (successful ball recoveries),
    ``tackles`` (ground duels), ``fouls`` (committed), ``dribbled_past``
    (beaten), ``blocks``, ``clearances``.
    """

    def col(name: str) -> pd.Series:
        if name in events.columns:
            return events[name]
        return pd.Series(index=events.index, dtype=object)

    def count(mask: pd.Series) -> pd.Series:
        return events.loc[mask, "player"].value_counts()

    is_type = events["type"]
    masks = {
        "interceptions": (is_type == "Interception")
        & col("interception_outcome").isin(_INTERCEPTION_WON),
        "recoveries": (is_type == "Ball Recovery")
        & ~col("ball_recovery_recovery_failure").fillna(False).astype(bool),
        "tackles": (is_type == "Duel") & (col("duel_type") == "Tackle"),
        "fouls": is_type == "Foul Committed",
        "dribbled_past": is_type == "Dribbled Past",
        "blocks": is_type == "Block",
        "clearances": is_type == "Clearance",
    }
    out = pd.DataFrame({k: count(m) for k, m in masks.items()}).fillna(0).astype(int)
    out.index.name = "player"
    return out.reset_index()
