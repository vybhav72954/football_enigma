import pandas as pd

from football_enigma.metrics.cadence import (
    cadence_summary,
    involvement_gaps,
    spell_cadence_summary,
    spell_gaps,
)


def events_for(player, times, period=1, match_id=1):
    return pd.DataFrame(
        {
            "match_id": match_id,
            "player": player,
            "period": period,
            "minute": [t // 60 for t in times],
            "second": [t % 60 for t in times],
        }
    )


def test_gaps_are_consecutive_differences():
    ev = events_for("A", [0, 30, 90, 210])
    gaps = involvement_gaps(ev)
    assert list(gaps["gap"]) == [30.0, 60.0, 120.0]


def test_gaps_do_not_span_periods():
    first_half = events_for("A", [0, 30], period=1)
    second_half = events_for("A", [2700, 2730], period=2)
    gaps = involvement_gaps(pd.concat([first_half, second_half]))
    assert list(gaps["gap"]) == [30.0, 30.0]  # no 2670s phantom gap


def test_mean_punishes_disappearance_median_does_not():
    # steady 30s rhythm, then one 10-minute silence
    times = list(range(0, 1200, 30)) + [1800]
    summary = cadence_summary(events_for("A", times), min_involvements=10)
    row = summary.iloc[0]
    assert row["median_gap"] == 30.0
    assert row["mean_gap"] > 40.0


def test_min_involvements_filter():
    ev = events_for("A", [0, 30, 60])
    assert cadence_summary(ev, min_involvements=100).empty


def test_spell_gaps_merge_event_chains():
    # receive/carry/pass logged 1-2s apart are one spell, not three
    # involvements; only the 40s and 60s silences are real gaps
    ev = events_for("A", [0, 1, 3, 43, 44, 104])
    gaps = spell_gaps(ev, spell_break=5.0)
    assert list(gaps["gap"]) == [40.0, 60.0]


def test_spell_cadence_summary_filters_and_summarises():
    ev = events_for("A", [0, 30, 60, 90])
    out = spell_cadence_summary(ev, min_spells=3)
    assert out["median_gap"].iloc[0] == 30.0
    assert int(out["spells"].iloc[0]) == 3
    assert spell_cadence_summary(ev, min_spells=10).empty
