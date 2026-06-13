"""Pass networks and betweenness centrality.

A team's pass network over one or more matches: nodes are players
(positioned at their median pass origin), edges are completed pass counts
between pairs. Betweenness centrality asks: when the ball travels between
two teammates who rarely pass directly, whose feet does it route through?
High betweenness = the connective tissue of the side.

Edge distance for shortest paths is 1 / pass count, the standard
inversion: frequent connections are "short". Computed with networkx.
"""

import networkx as nx
import pandas as pd


def pass_edges(passes: pd.DataFrame, min_passes: int = 1) -> pd.DataFrame:
    """Undirected completed-pass counts per player pair.

    Expects canonical passes with a `recipient` column (StatsBomb
    pass_recipient mapped in by the caller).
    """
    completed = passes[passes["outcome"] & passes["recipient"].notna()].copy()
    if completed.empty:
        return pd.DataFrame(columns=["a", "b", "count"])
    pair = completed.apply(
        lambda r: tuple(sorted((r["player"], r["recipient"]))), axis=1
    )
    edges = (
        pair.value_counts()
        .rename_axis("pair")
        .reset_index(name="count")
    )
    edges[["a", "b"]] = pd.DataFrame(edges["pair"].tolist(), index=edges.index)
    edges = edges[edges["a"] != edges["b"]]
    return edges[edges["count"] >= min_passes][["a", "b", "count"]]


def node_positions(passes: pd.DataFrame) -> pd.DataFrame:
    """Median pass origin per player (canonical meters) + pass volume."""
    return (
        passes.groupby("player", observed=True)
        .agg(x=("x", "median"), y=("y", "median"), passes=("x", "size"))
        .reset_index()
    )


def betweenness(edges: pd.DataFrame) -> pd.Series:
    """Weighted betweenness centrality per player, descending."""
    graph = nx.Graph()
    for _, e in edges.iterrows():
        graph.add_edge(e["a"], e["b"], distance=1.0 / e["count"])
    central = nx.betweenness_centrality(graph, weight="distance")
    return pd.Series(central, name="betweenness").sort_values(ascending=False)
