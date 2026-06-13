import pandas as pd
import pytest

from football_enigma.metrics.network import (
    betweenness,
    node_positions,
    pass_edges,
)


def passes_between(pairs):
    """One completed pass row per (player, recipient) tuple."""
    return pd.DataFrame(
        {
            "player": [p for p, _ in pairs],
            "recipient": [r for _, r in pairs],
            "outcome": True,
            "x": 50.0,
            "y": 30.0,
        }
    )


def test_edges_are_undirected_counts():
    passes = passes_between([("A", "B"), ("B", "A"), ("A", "B"), ("A", "C")])
    edges = pass_edges(passes)
    ab = edges[(edges["a"] == "A") & (edges["b"] == "B")]
    assert int(ab["count"].iloc[0]) == 3
    assert len(edges) == 2


def test_incomplete_and_unreceived_passes_excluded():
    passes = passes_between([("A", "B"), ("A", "B")])
    passes.loc[0, "outcome"] = False
    passes.loc[1, "recipient"] = None
    assert pass_edges(passes).empty


def test_star_centre_has_all_betweenness():
    # hub H connects A, B, C who never pass to each other
    edges = pass_edges(
        passes_between([("H", "A"), ("H", "B"), ("H", "C")])
    )
    central = betweenness(edges)
    assert central.idxmax() == "H"
    assert central["A"] == central["B"] == central["C"] == 0.0


def test_strong_link_attracts_routes():
    # A-B direct link is weak (1 pass); A-H and H-B are strong (10 each):
    # the shortest A->B route goes through H
    pairs = [("A", "B")] + [("A", "H")] * 10 + [("H", "B")] * 10
    central = betweenness(pass_edges(passes_between(pairs)))
    assert central["H"] > 0.0


def test_node_positions_median_and_volume():
    passes = passes_between([("A", "B"), ("A", "B"), ("B", "A")])
    passes["x"] = [10.0, 20.0, 90.0]
    pos = node_positions(passes)
    a = pos[pos["player"] == "A"].iloc[0]
    assert a["x"] == pytest.approx(15.0)
    assert int(a["passes"]) == 2
