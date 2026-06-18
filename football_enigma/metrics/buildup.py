"""First-receiver / build-up role from raw StatsBomb events.

The release-valve measure: how much of a team's distribution *out of its own
back line* runs through a given player. For every completed pass we look at the
passer's position and the receiver's modal position; a pass that leaves the back
line (passer is a goalkeeper or centre-back) for a non-backline teammate is
"build-up distribution", and is credited to the receiver. The pure first
receiver absorbs a large share of these — value the goals/assists columns never
see.

Operates on the raw StatsBomb event frame (needs ``pass_recipient`` and
``position``), not the canonical schema. Counts are per match.
"""

import pandas as pd

from football_enigma.metrics.defending import BACKLINE


def backline_receptions(
    events: pd.DataFrame, backline: set[str] = BACKLINE
) -> pd.DataFrame:
    """Per-player receptions of the ball out of the team's own back line.

    One row per player who completed at least one reception, columns:
    ``team``, ``from_backline`` (completed passes received from an own
    goalkeeper/centre-back into a non-backline role) and ``total_received``
    (all completed receptions). The build-up share is
    ``from_backline / total_received``; the funnel volume is ``from_backline``.
    """
    cols = {"pass_recipient", "position", "type", "player", "team"}
    if not cols.issubset(events.columns):
        return pd.DataFrame(
            columns=["player", "team", "from_backline", "total_received"]
        )

    passes = events[events["type"] == "Pass"]
    if "pass_outcome" in passes.columns:
        passes = passes[passes["pass_outcome"].isna()]  # NaN = completed
    completed = passes.dropna(subset=["pass_recipient"])

    pos_map = (
        events.dropna(subset=["player", "position"])
        .groupby("player")["position"]
        .agg(lambda s: s.mode(dropna=True).iloc[0])
    )
    recip_pos = completed["pass_recipient"].map(pos_map)
    from_back = completed["position"].isin(backline) & ~recip_pos.isin(backline)

    out = pd.DataFrame(
        {
            "from_backline": completed[from_back]
            .groupby("pass_recipient")
            .size(),
            "total_received": completed.groupby("pass_recipient").size(),
            "team": completed.groupby("pass_recipient")["team"].agg(
                lambda s: s.mode().iloc[0]
            ),
        }
    )
    out["from_backline"] = out["from_backline"].fillna(0).astype(int)
    out.index.name = "player"
    return out.reset_index()
