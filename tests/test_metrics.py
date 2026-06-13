import numpy as np
import pandas as pd
import pytest

from football_enigma.metrics.aggregates import (
    flag_final_third,
    flag_progressive,
    flag_switch,
    player_pass_aggregates,
)
from football_enigma.metrics.pressure import pressure_split


def make_passes(rows):
    df = pd.DataFrame(
        rows,
        columns=["player", "x", "y", "end_x", "end_y", "outcome", "under_pressure"],
    )
    return df


class TestFlags:
    def test_progressive_requires_completion_and_real_gain(self):
        passes = make_passes(
            [
                ("A", 30, 34, 80, 34, True, False),   # big gain -> progressive
                ("A", 30, 34, 80, 34, False, False),  # incomplete -> no
                ("A", 50, 34, 52, 34, True, False),   # 2m -> no
                ("A", 100, 34, 103, 34, True, False), # close to goal, <5m -> no
            ]
        )
        assert list(flag_progressive(passes)) == [True, False, False, False]

    def test_final_third_only_counts_entries(self):
        passes = make_passes(
            [
                ("A", 50, 34, 80, 34, True, False),   # entry
                ("A", 75, 34, 90, 34, True, False),   # already inside -> no
                ("A", 50, 34, 80, 34, False, False),  # incomplete -> no
            ]
        )
        assert list(flag_final_third(passes)) == [True, False, False]

    def test_switch_uses_lateral_travel(self):
        passes = make_passes(
            [
                ("A", 50, 5, 60, 60, True, False),   # 55m lateral -> switch
                ("A", 50, 30, 60, 40, True, False),  # 10m -> no
            ]
        )
        assert list(flag_switch(passes)) == [True, False]


class TestAggregates:
    def test_per90_normalisation(self):
        passes = make_passes(
            [("A", 30, 34, 80, 34, True, False)] * 4
            + [("B", 30, 34, 80, 34, True, False)] * 4
        )
        minutes = pd.Series({"A": 90.0, "B": 180.0}, name="minutes")
        agg = player_pass_aggregates(passes, minutes).set_index("player")
        assert agg.loc["A", "passes_p90"] == pytest.approx(4.0)
        assert agg.loc["B", "passes_p90"] == pytest.approx(2.0)


class TestPressureSplit:
    def test_partition_and_completion_drop(self):
        rows = []
        # player A: 20 unpressured all complete, 10 pressured half complete
        rows += [("A", 50, 34, 60, 34, True, False)] * 20
        rows += [("A", 50, 34, 60, 34, True, True)] * 5
        rows += [("A", 50, 34, 60, 34, False, True)] * 5
        passes = make_passes(rows)
        result = pressure_split(passes, min_pressured=10)
        row = result[result["player"] == "A"].iloc[0]
        assert row["passes"] == row["pressured"] + 20
        assert row["comp_pct_unpressured"] == pytest.approx(100.0)
        assert row["comp_pct_pressured"] == pytest.approx(50.0)
        assert row["completion_drop"] == pytest.approx(50.0)

    def test_min_pressured_filters_noise(self):
        passes = make_passes(
            [("B", 50, 34, 60, 34, True, True)] * 3
            + [("B", 50, 34, 60, 34, True, False)] * 30
        )
        assert pressure_split(passes, min_pressured=10).empty
