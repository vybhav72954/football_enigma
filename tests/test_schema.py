import pandas as pd
import pytest

from football_enigma.data.schema import (
    PITCH_LENGTH,
    PITCH_WIDTH,
    opta_passes,
    statsbomb_passes,
)


def test_statsbomb_pass_conversion_scales_and_flips():
    events = pd.DataFrame(
        {
            "type": ["Pass"],
            "team": ["T"],
            "player": ["A"],
            "period": [1],
            "minute": [10],
            "second": [5],
            "location": [[60.0, 0.0]],          # SB midfield, top touchline
            "pass_end_location": [[120.0, 80.0]],  # SB right corner, bottom
            "pass_outcome": [None],             # NaN = completed
            "under_pressure": [None],
        }
    )
    out = statsbomb_passes(events, match_id=1)
    row = out.iloc[0]
    assert row["x"] == pytest.approx(52.5)            # 60/120 * 105
    assert row["y"] == pytest.approx(PITCH_WIDTH)     # SB y=0 is top -> flip
    assert row["end_x"] == pytest.approx(PITCH_LENGTH)
    assert row["end_y"] == pytest.approx(0.0)
    assert bool(row["outcome"]) is True
    assert bool(row["under_pressure"]) is False


def test_statsbomb_incomplete_pass_outcome():
    events = pd.DataFrame(
        {
            "type": ["Pass"],
            "team": ["T"],
            "player": ["A"],
            "period": [1],
            "minute": [1],
            "second": [0],
            "location": [[10.0, 40.0]],
            "pass_end_location": [[20.0, 40.0]],
            "pass_outcome": ["Incomplete"],
        }
    )
    assert bool(statsbomb_passes(events, 1)["outcome"].iloc[0]) is False


def test_opta_pass_conversion():
    events = pd.DataFrame(
        {
            "game_id": [99, 99],
            "type": ["Pass", "Tackle"],
            "team": ["T", "T"],
            "player": ["A", "B"],
            "period": ["FirstHalf", "FirstHalf"],
            "minute": [3, 4],
            "second": [12, 0],
            "x": [50.0, 10.0],
            "y": [50.0, 10.0],
            "end_x": [100.0, None],
            "end_y": [0.0, None],
            "outcome_type": ["Successful", "Successful"],
        }
    )
    out = opta_passes(events)
    assert len(out) == 1  # tackle excluded
    row = out.iloc[0]
    assert row["match_id"] == 99
    assert row["x"] == pytest.approx(52.5)   # 50% of 105
    assert row["y"] == pytest.approx(34.0)   # 50% of 68
    assert row["end_x"] == pytest.approx(PITCH_LENGTH)
    assert row["end_y"] == pytest.approx(0.0)
    assert row["period"] == 1
    assert bool(row["outcome"]) is True


def test_statsbomb_set_piece_flag():
    events = pd.DataFrame(
        {
            "type": ["Pass", "Pass", "Pass"],
            "team": ["T"] * 3,
            "player": ["A"] * 3,
            "period": [1] * 3,
            "minute": [1, 2, 3],
            "second": [0] * 3,
            "location": [[10.0, 40.0]] * 3,
            "pass_end_location": [[20.0, 40.0]] * 3,
            "pass_outcome": [None] * 3,
            "pass_type": ["Free Kick", "Recovery", None],
        }
    )
    out = statsbomb_passes(events, 1)
    assert list(out["set_piece"]) == [True, False, False]


def test_opta_set_piece_flag():
    free_kick = [{"type": {"displayName": "FreekickTaken", "value": 5}}]
    open_play = [{"type": {"displayName": "Zone", "value": 56}}]
    events = pd.DataFrame(
        {
            "game_id": [9, 9],
            "type": ["Pass", "Pass"],
            "team": ["T", "T"],
            "player": ["A", "A"],
            "period": ["FirstHalf", "FirstHalf"],
            "minute": [3, 4],
            "second": [0, 0],
            "x": [50.0, 50.0],
            "y": [50.0, 50.0],
            "end_x": [60.0, 60.0],
            "end_y": [50.0, 50.0],
            "outcome_type": ["Successful", "Successful"],
            "qualifiers": [free_kick, open_play],
        }
    )
    out = opta_passes(events)
    assert list(out["set_piece"]) == [True, False]
