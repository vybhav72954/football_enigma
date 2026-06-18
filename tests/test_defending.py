import pandas as pd

from football_enigma.metrics.defending import defensive_profile


def _events(rows: list[dict]) -> pd.DataFrame:
    # union of every key seen, so absent subtype columns are NaN not missing
    cols = sorted({k for r in rows for k in r})
    return pd.DataFrame([{c: r.get(c) for c in cols} for r in rows])


def test_counts_each_defensive_action_type():
    ev = _events(
        [
            {"type": "Interception", "player": "A", "interception_outcome": "Won"},
            {"type": "Ball Recovery", "player": "A"},
            {"type": "Duel", "player": "A", "duel_type": "Tackle"},
            {"type": "Foul Committed", "player": "A"},
            {"type": "Dribbled Past", "player": "A"},
            {"type": "Block", "player": "A"},
            {"type": "Clearance", "player": "A"},
        ]
    )
    row = defensive_profile(ev).set_index("player").loc["A"]
    assert row["interceptions"] == 1
    assert row["recoveries"] == 1
    assert row["tackles"] == 1
    assert row["fouls"] == 1
    assert row["dribbled_past"] == 1
    assert row["blocks"] == 1
    assert row["clearances"] == 1


def test_only_won_interceptions_count():
    ev = _events(
        [
            {"type": "Interception", "player": "A", "interception_outcome": "Won"},
            {"type": "Interception", "player": "A", "interception_outcome": "Lost"},
        ]
    )
    assert defensive_profile(ev).set_index("player").loc["A", "interceptions"] == 1


def test_failed_recoveries_excluded():
    ev = _events(
        [
            {"type": "Ball Recovery", "player": "A"},
            {"type": "Ball Recovery", "player": "A",
             "ball_recovery_recovery_failure": True},
        ]
    )
    assert defensive_profile(ev).set_index("player").loc["A", "recoveries"] == 1


def test_aerial_duels_are_not_tackles():
    ev = _events(
        [
            {"type": "Duel", "player": "A", "duel_type": "Tackle"},
            {"type": "Duel", "player": "A", "duel_type": "Aerial Lost"},
        ]
    )
    assert defensive_profile(ev).set_index("player").loc["A", "tackles"] == 1


def test_per_player_rows_independent():
    ev = _events(
        [
            {"type": "Interception", "player": "A", "interception_outcome": "Won"},
            {"type": "Foul Committed", "player": "B"},
        ]
    )
    prof = defensive_profile(ev).set_index("player")
    assert prof.loc["A", "interceptions"] == 1 and prof.loc["A", "fouls"] == 0
    assert prof.loc["B", "fouls"] == 1 and prof.loc["B", "interceptions"] == 0
