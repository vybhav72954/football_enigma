import numpy as np
import pandas as pd

from football_enigma.metrics.footprint import action_dispersion


def _actions(player_xy: dict[str, list[tuple[float, float]]]) -> pd.DataFrame:
    rows = []
    for player, pts in player_xy.items():
        for x, y in pts:
            rows.append({"player": player, "x": x, "y": y})
    return pd.DataFrame(rows)


def test_single_point_has_zero_spread():
    df = action_dispersion(_actions({"A": [(50.0, 34.0), (50.0, 34.0)]}))
    row = df.set_index("player").loc["A"]
    assert row["spread"] == 0.0
    assert row["cx"] == 50.0 and row["cy"] == 34.0
    assert row["actions"] == 2


def test_standard_distance_is_root_of_summed_variances():
    # x in {0,10} -> std 5; y in {0,0} -> std 0; spread = 5
    df = action_dispersion(_actions({"A": [(0.0, 0.0), (10.0, 0.0)]}))
    assert df.set_index("player").loc["A", "spread"] == 5.0
    # x spread 5 and y spread 5 -> standard distance sqrt(50)
    df2 = action_dispersion(_actions({"B": [(0.0, 0.0), (10.0, 10.0)]}))
    assert np.isclose(df2.set_index("player").loc["B", "spread"], np.hypot(5, 5))


def test_tighter_player_has_smaller_spread():
    df = action_dispersion(
        _actions({"tight": [(50, 34), (51, 35), (49, 33)],
                  "roamer": [(10, 5), (90, 60), (50, 34)]})
    ).set_index("player")
    assert df.loc["tight", "spread"] < df.loc["roamer", "spread"]


def test_min_actions_filters_low_volume_players():
    df = action_dispersion(
        _actions({"A": [(1, 1), (2, 2), (3, 3)], "B": [(4, 4)]}),
        min_actions=2,
    )
    assert set(df["player"]) == {"A"}


def test_nan_locations_dropped():
    df = pd.DataFrame(
        {"player": ["A", "A", "A"], "x": [0.0, 10.0, np.nan], "y": [0.0, 0.0, 5.0]}
    )
    row = action_dispersion(df).set_index("player").loc["A"]
    assert row["actions"] == 2 and row["spread"] == 5.0
