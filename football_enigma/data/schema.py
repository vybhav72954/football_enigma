"""Unified event/pass table schema shared by all data sources.

Every provider (StatsBomb 120x80, Opta/WhoScored 100x100) is converted to a
single canonical representation so the metrics layer is source-agnostic:

- pitch: 105 x 68 meters (UEFA standard), origin bottom-left of the
  attacking-left-to-right team perspective (mplsoccer 'custom' convention)
- one row per on-ball action, columns below

Canonical columns
-----------------
match_id, source, team, player, period, minute, second,
type        : 'pass' | 'carry' | 'shot' | 'dribble' | ...
x, y        : action start (meters)
end_x, end_y: action end (meters; NaN for actions without an end location)
outcome     : True = successful/completed
under_pressure : bool (StatsBomb only; False where unknown)
shot_assist : bool, pass led directly to a shot/goal (StatsBomb passes only)
set_piece   : bool, pass taken directly as a dead-ball restart (corner,
              free kick, throw-in, goal kick, kick-off, keeper throw)
recipient   : receiving player (StatsBomb passes only; NaN if unreceived)
"""

import numpy as np
import pandas as pd

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

CANONICAL_COLUMNS = [
    "match_id", "source", "team", "player", "period", "minute", "second",
    "type", "x", "y", "end_x", "end_y", "outcome", "under_pressure",
]


def _scale(series: pd.Series, from_max: float, to_max: float) -> pd.Series:
    return pd.to_numeric(series, errors="coerce") * (to_max / from_max)


# StatsBomb pass_type values that mark dead-ball restarts ('Recovery' and
# 'Interception' describe how possession began, not a restart)
_SB_SET_PIECE_TYPES = {"Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off"}

# Opta qualifier displayNames that mark dead-ball restarts
_OPTA_SET_PIECE_QUALIFIERS = {
    "CornerTaken", "FreekickTaken", "IndirectFreekickTaken", "ThrowIn",
    "GoalKick", "KeeperThrow", "KickOff",
}


def _opta_set_piece(qualifiers) -> bool:
    try:
        names = {q["type"]["displayName"] for q in qualifiers}
    except (TypeError, KeyError):
        return False
    return bool(names & _OPTA_SET_PIECE_QUALIFIERS)


def statsbomb_passes(events: pd.DataFrame, match_id: int) -> pd.DataFrame:
    """Convert decoded StatsBomb events to canonical pass rows.

    Expects `location` / `pass_end_location` already decoded to lists
    (see statsbomb.decode_locations).
    """
    passes = events[events["type"] == "Pass"].copy()
    loc = passes["location"].map(
        lambda v: v if isinstance(v, list) else [np.nan, np.nan]
    )
    end = passes["pass_end_location"].map(
        lambda v: v if isinstance(v, list) else [np.nan, np.nan]
    )
    out = pd.DataFrame(
        {
            "match_id": match_id,
            "source": "statsbomb",
            "team": passes["team"].values,
            "player": passes["player"].values,
            "period": passes["period"].values,
            "minute": passes["minute"].values,
            "second": passes["second"].values,
            "type": "pass",
            "x": _scale(pd.Series([p[0] for p in loc]), 120, PITCH_LENGTH).values,
            # StatsBomb y grows top-down; flip to bottom-left origin
            "y": (PITCH_WIDTH - _scale(pd.Series([p[1] for p in loc]), 80, PITCH_WIDTH)).values,
            "end_x": _scale(pd.Series([p[0] for p in end]), 120, PITCH_LENGTH).values,
            "end_y": (PITCH_WIDTH - _scale(pd.Series([p[1] for p in end]), 80, PITCH_WIDTH)).values,
            # StatsBomb convention: pass_outcome is NaN when complete
            "outcome": passes["pass_outcome"].isna().values
            if "pass_outcome" in passes.columns
            else True,
            "under_pressure": passes["under_pressure"].fillna(False).astype(bool).values
            if "under_pressure" in passes.columns
            else False,
        }
    )
    # direct shot/goal assists -> the "danger created" signal
    shot_assist = pd.Series(False, index=passes.index)
    for col in ("pass_shot_assist", "pass_goal_assist"):
        if col in passes.columns:
            shot_assist |= passes[col].fillna(False).astype(bool)
    out["shot_assist"] = shot_assist.values
    out["set_piece"] = (
        passes["pass_type"].isin(_SB_SET_PIECE_TYPES).values
        if "pass_type" in passes.columns
        else False
    )
    out["recipient"] = (
        passes["pass_recipient"].values
        if "pass_recipient" in passes.columns
        else None
    )
    return out.reset_index(drop=True)


def statsbomb_actions(events: pd.DataFrame, match_id: int) -> pd.DataFrame:
    """All xT-relevant actions (passes, carries, dribbles, shots) in
    canonical form — the input for fitting an xT model."""
    frames = [statsbomb_passes(events, match_id)]

    carries = events[events["type"] == "Carry"]
    if len(carries) and "carry_end_location" in events.columns:
        loc = carries["location"].map(
            lambda v: v if isinstance(v, list) else [np.nan, np.nan]
        )
        end = carries["carry_end_location"].map(
            lambda v: v if isinstance(v, list) else [np.nan, np.nan]
        )
        frames.append(
            pd.DataFrame(
                {
                    "match_id": match_id,
                    "source": "statsbomb",
                    "team": carries["team"].values,
                    "player": carries["player"].values,
                    "period": carries["period"].values,
                    "minute": carries["minute"].values,
                    "second": carries["second"].values,
                    "type": "carry",
                    "x": _scale(pd.Series([p[0] for p in loc]), 120, PITCH_LENGTH).values,
                    "y": (PITCH_WIDTH - _scale(pd.Series([p[1] for p in loc]), 80, PITCH_WIDTH)).values,
                    "end_x": _scale(pd.Series([p[0] for p in end]), 120, PITCH_LENGTH).values,
                    "end_y": (PITCH_WIDTH - _scale(pd.Series([p[1] for p in end]), 80, PITCH_WIDTH)).values,
                    "outcome": True,  # StatsBomb only records completed carries
                    "under_pressure": carries["under_pressure"].fillna(False).astype(bool).values
                    if "under_pressure" in carries.columns
                    else False,
                }
            )
        )

    shots = events[events["type"] == "Shot"]
    if len(shots):
        loc = shots["location"].map(
            lambda v: v if isinstance(v, list) else [np.nan, np.nan]
        )
        frames.append(
            pd.DataFrame(
                {
                    "match_id": match_id,
                    "source": "statsbomb",
                    "team": shots["team"].values,
                    "player": shots["player"].values,
                    "period": shots["period"].values,
                    "minute": shots["minute"].values,
                    "second": shots["second"].values,
                    "type": "shot",
                    "x": _scale(pd.Series([p[0] for p in loc]), 120, PITCH_LENGTH).values,
                    "y": (PITCH_WIDTH - _scale(pd.Series([p[1] for p in loc]), 80, PITCH_WIDTH)).values,
                    "end_x": np.nan,
                    "end_y": np.nan,
                    "outcome": (shots["shot_outcome"] == "Goal").values
                    if "shot_outcome" in shots.columns
                    else False,
                    "under_pressure": shots["under_pressure"].fillna(False).astype(bool).values
                    if "under_pressure" in shots.columns
                    else False,
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


_OPTA_PERIODS = {"FirstHalf": 1, "SecondHalf": 2, "FirstPeriodOfExtraTime": 3,
                 "SecondPeriodOfExtraTime": 4, "PenaltyShootout": 5}


def opta_passes(events: pd.DataFrame, match_id: int | None = None) -> pd.DataFrame:
    """Convert raw WhoScored/Opta match-centre events (as returned by
    soccerdata's read_events) to canonical pass rows.

    Opta coordinates are percentages of pitch length/width (0-100 both
    axes, attacking left->right). Completion comes from ``outcome_type``
    ('Successful'/'Unsuccessful').
    """
    passes = events[events["type"] == "Pass"].copy()
    out = pd.DataFrame(
        {
            "match_id": passes["game_id"].values
            if match_id is None and "game_id" in passes.columns
            else match_id,
            "source": "opta",
            "team": passes["team"].values,
            "player": passes["player"].values,
            "period": passes["period"]
            .map(lambda v: _OPTA_PERIODS.get(v, v))
            .astype(int)
            .values,
            "minute": passes["minute"].values,
            "second": passes["second"].values,
            "type": "pass",
            "x": _scale(passes["x"], 100, PITCH_LENGTH).values,
            "y": _scale(passes["y"], 100, PITCH_WIDTH).values,
            "end_x": _scale(passes["end_x"], 100, PITCH_LENGTH).values,
            "end_y": _scale(passes["end_y"], 100, PITCH_WIDTH).values,
            "outcome": (passes["outcome_type"] == "Successful").values,
            "under_pressure": False,
        }
    )
    out["set_piece"] = (
        passes["qualifiers"].map(_opta_set_piece).values
        if "qualifiers" in passes.columns
        else False
    )
    return out.reset_index(drop=True)
