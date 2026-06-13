import pandas as pd

from football_enigma.metrics.tempo import receipt_to_release, tempo_summary


def match_events(rows):
    return pd.DataFrame(
        rows,
        columns=["type", "player", "period", "possession", "timestamp",
                 "ball_receipt_outcome"],
    )


def test_receipt_pairs_with_first_following_pass():
    ev = match_events([
        ("Ball Receipt*", "A", 1, 1, "00:00:10.000", None),
        ("Carry", "A", 1, 1, "00:00:10.500", None),
        ("Pass", "A", 1, 1, "00:00:12.500", None),
        ("Pass", "A", 1, 1, "00:00:18.000", None),  # later pass ignored
    ])
    holds = receipt_to_release(ev, match_id=7)
    assert len(holds) == 1
    assert holds["hold"].iloc[0] == 2.5
    assert holds["match_id"].iloc[0] == 7


def test_turnover_does_not_bridge_possessions():
    # receipt in possession 1, next pass only in possession 2
    ev = match_events([
        ("Ball Receipt*", "A", 1, 1, "00:00:10.000", None),
        ("Pass", "A", 1, 2, "00:00:30.000", None),
    ])
    assert receipt_to_release(ev, 1).empty


def test_incomplete_receipt_excluded():
    ev = match_events([
        ("Ball Receipt*", "A", 1, 1, "00:00:10.000", "Incomplete"),
        ("Pass", "A", 1, 1, "00:00:11.000", None),
    ])
    assert receipt_to_release(ev, 1).empty


def test_stale_hold_discarded():
    # a pass 25s later in the same possession is not a usable resolution
    ev = match_events([
        ("Ball Receipt*", "A", 1, 1, "00:00:10.000", None),
        ("Pass", "A", 1, 1, "00:00:35.000", None),
    ])
    assert receipt_to_release(ev, 1).empty


def test_tempo_summary_median_and_filter():
    holds = pd.DataFrame(
        {"match_id": 1, "player": ["A"] * 3 + ["B"], "possession": 1,
         "hold": [1.0, 2.0, 9.0, 4.0]}
    )
    out = tempo_summary(holds, min_receipts=2)
    assert list(out["player"]) == ["A"]  # B below threshold
    assert out["median_hold"].iloc[0] == 2.0
    assert out["mean_hold"].iloc[0] == 4.0
