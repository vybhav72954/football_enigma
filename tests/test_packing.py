import pandas as pd

from football_enigma.metrics.packing import (
    match_crowding,
    match_packing,
    opponents_within,
    packing_count,
)


def ff_player(x, y, teammate=False, keeper=False):
    return {
        "teammate": teammate,
        "keeper": keeper,
        "actor": False,
        "location": [x, y],
    }


def test_no_defenders_between_means_zero():
    frame = [ff_player(20, 40), ff_player(90, 40)]
    assert packing_count(40, 80, frame) == 0


def test_counts_only_defenders_strictly_between():
    frame = [
        ff_player(50, 40),               # defender, between -> counts
        ff_player(60, 10),               # defender, between -> counts
        ff_player(50, 40, teammate=True),  # teammate -> ignored
        ff_player(75, 40, keeper=True),    # keeper -> ignored
        ff_player(85, 40),               # beyond end -> ignored
    ]
    assert packing_count(40, 80, frame) == 2


def test_backward_pass_is_zero():
    frame = [ff_player(50, 40)]
    assert packing_count(80, 40, frame) == 0


def test_incomplete_or_frameless_pass_is_none():
    assert packing_count(40, 80, [ff_player(50, 40)], completed=False) is None
    assert packing_count(40, 80, None) is None


def test_match_packing_joins_frames_and_skips_incomplete():
    events = pd.DataFrame(
        {
            "id": ["e1", "e2", "e3"],
            "type": ["Pass", "Pass", "Pass"],
            "player": ["A", "A", "B"],
            "team": ["T", "T", "T"],
            "minute": [1, 2, 3],
            "location": [[40.0, 40.0], [40.0, 40.0], [40.0, 40.0]],
            "pass_end_location": [[80.0, 40.0], [80.0, 40.0], [80.0, 40.0]],
            # e2 incomplete; e3 has no frame
            "pass_outcome": [None, "Incomplete", None],
        }
    )
    frames = [
        {"event_uuid": "e1", "visible_area": [], "freeze_frame": [ff_player(50, 40)]},
        {"event_uuid": "e2", "visible_area": [], "freeze_frame": [ff_player(50, 40)]},
    ]
    result = match_packing(events, frames)
    assert list(result["id"]) == ["e1"]
    assert result["packing"].iloc[0] == 1


def test_opponents_within_counts_close_outfield_opponents():
    here = [60.0, 40.0]
    frame = [
        ff_player(62, 40),                 # 1.75 m -> within
        ff_player(60, 43),                 # 2.55 m -> within
        ff_player(66, 40),                 # 5.25 m -> outside
        ff_player(62, 40, teammate=True),  # teammate -> ignored
        ff_player(60, 40, keeper=True),    # keeper -> ignored
    ]
    assert opponents_within(here, frame, radius_m=3.0) == 2


def test_opponents_within_none_without_frame():
    assert opponents_within([60.0, 40.0], None) is None


def test_opponents_within_radius_is_in_meters():
    # 4 x-units = 3.5 m: outside a 3 m radius, inside a 4 m radius
    frame = [ff_player(64, 40)]
    assert opponents_within([60.0, 40.0], frame, radius_m=3.0) == 0
    assert opponents_within([60.0, 40.0], frame, radius_m=4.0) == 1


def test_match_crowding_keeps_incomplete_and_skips_frameless():
    events = pd.DataFrame(
        {
            "id": ["e1", "e2", "e3"],
            "type": ["Pass", "Pass", "Pass"],
            "player": ["A", "A", "B"],
            "team": ["T", "T", "T"],
            "minute": [1, 2, 3],
            "location": [[60.0, 40.0], [60.0, 40.0], [60.0, 40.0]],
            "pass_outcome": [None, "Incomplete", None],
            "under_pressure": [True, None, True],
        }
    )
    frames = [
        {"event_uuid": "e1", "freeze_frame": [ff_player(62, 40)]},  # 1.75 m
        {"event_uuid": "e2", "freeze_frame": [ff_player(66, 40)]},  # 5.25 m
        # e3 has no frame -> skipped
    ]
    result = match_crowding(events, frames, radius_m=3.0)
    assert list(result["id"]) == ["e1", "e2"]
    assert list(result["near"]) == [1, 0]
    assert list(result["completed"]) == [True, False]
