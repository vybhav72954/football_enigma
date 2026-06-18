import pandas as pd

from football_enigma.metrics.buildup import backline_receptions


def _pass(player, recipient, position, team="T", outcome=None):
    return {
        "type": "Pass",
        "player": player,
        "pass_recipient": recipient,
        "position": position,
        "team": team,
        "pass_outcome": outcome,
    }


def test_credits_receiver_of_backline_distribution():
    # CB -> pivot (build-up out of the line); pivot's modal position is set by
    # his own later pass event
    ev = pd.DataFrame(
        [
            _pass("CB", "Pivot", "Center Back"),
            _pass("Pivot", "Winger", "Center Defensive Midfield"),
            _pass("Winger", "Pivot", "Right Wing"),
        ]
    )
    out = backline_receptions(ev).set_index("player")
    assert out.loc["Pivot", "from_backline"] == 1
    assert out.loc["Pivot", "total_received"] == 2  # from CB and from Winger


def test_centre_back_recycling_is_not_distribution():
    # CB -> CB stays inside the back line, so it is not credited
    ev = pd.DataFrame(
        [
            _pass("CB1", "CB2", "Center Back"),
            _pass("CB2", "CB1", "Center Back"),
        ]
    )
    out = backline_receptions(ev).set_index("player")
    assert out.loc["CB2", "from_backline"] == 0


def test_incomplete_passes_excluded():
    # the only ball into the pivot from the back line was incomplete, so the
    # pivot is credited with no backline reception (and, having received no
    # completed pass at all, does not appear in the table)
    ev = pd.DataFrame(
        [
            _pass("CB", "Pivot", "Center Back", outcome="Incomplete"),
            _pass("Pivot", "Winger", "Center Defensive Midfield"),
        ]
    )
    out = backline_receptions(ev).set_index("player")
    assert "Pivot" not in out.index
    assert out.loc["Winger", "from_backline"] == 0  # passer was the pivot, not backline


def test_goalkeeper_counts_as_backline():
    ev = pd.DataFrame(
        [
            _pass("GK", "Pivot", "Goalkeeper"),
            _pass("Pivot", "Winger", "Center Defensive Midfield"),
        ]
    )
    out = backline_receptions(ev).set_index("player")
    assert out.loc["Pivot", "from_backline"] == 1


def test_missing_columns_returns_empty():
    ev = pd.DataFrame({"type": ["Pass"], "player": ["A"]})
    assert backline_receptions(ev).empty
