"""Packing / line-breaking from StatsBomb 360 freeze-frames.

For each completed pass we count the opponents "taken out of the game":
outfield defenders whose pitch-length coordinate lies strictly between the
pass origin and the pass destination of a forward pass. This is the
freeze-frame analogue of the Impect "packing" metric.

Works in native StatsBomb coordinates (120 x 80, x toward opponent goal —
every team's events are already oriented to attack left->right). Frames are
the raw dicts returned by data.statsbomb.load_frames.

Caveat (must be stated wherever results are published): freeze-frames only
contain players inside the camera's visible area, so packing counts are a
lower bound, and passes without a frame are excluded rather than zero.
"""

import pandas as pd


def packing_count(
    start_x: float,
    end_x: float,
    freeze_frame: list[dict],
    completed: bool = True,
) -> int | None:
    """Defenders bypassed by one pass; None if the pass can't be assessed."""
    if not completed or freeze_frame is None:
        return None
    if end_x <= start_x:  # only forward passes can break lines
        return 0
    return sum(
        1
        for p in freeze_frame
        if not p["teammate"]
        and not p["keeper"]
        and start_x < p["location"][0] < end_x
    )


def match_packing(events: pd.DataFrame, frames: list[dict]) -> pd.DataFrame:
    """Per-pass packing for one match.

    `events`: decoded StatsBomb events (location / pass_end_location as
    lists). `frames`: load_frames output. Returns one row per completed
    pass that has a freeze-frame, with a `packing` column.
    """
    frame_by_event = {f["event_uuid"]: f["freeze_frame"] for f in frames}
    passes = events[
        (events["type"] == "Pass")
        & events["location"].map(lambda v: isinstance(v, list))
        & events["pass_end_location"].map(lambda v: isinstance(v, list))
    ].copy()
    if "pass_outcome" in passes.columns:
        passes = passes[passes["pass_outcome"].isna()]  # NaN = completed

    rows = []
    for _, p in passes.iterrows():
        ff = frame_by_event.get(p["id"])
        if ff is None:
            continue
        rows.append(
            {
                "id": p["id"],
                "player": p["player"],
                "team": p["team"],
                "minute": p["minute"],
                "x": p["location"][0],
                "y": p["location"][1],
                "end_x": p["pass_end_location"][0],
                "end_y": p["pass_end_location"][1],
                "packing": packing_count(
                    p["location"][0], p["pass_end_location"][0], ff
                ),
            }
        )
    return pd.DataFrame(rows)
