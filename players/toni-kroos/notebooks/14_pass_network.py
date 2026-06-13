"""Germany's Euro 2024 pass network — and who the ball routes through.

Fig 20 — all five Germany matches as one network: players at their median
pass origin, edge width = completed passes between the pair. Betweenness
centrality (printed + node label) answers: when the ball travels between
two players who rarely connect directly, whose feet carry it?

Aggregating five matches mixes lineups; node positions are median origins
across whatever minutes each player got, so fringe players sit where they
actually played, not in a formation slot.

Run:  .venv/Scripts/python -X utf8 players/toni-kroos/notebooks/14_pass_network.py
"""

import sys

import matplotlib

matplotlib.use("Agg")
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from football_enigma.data import statsbomb as sb_data
from football_enigma.data.schema import statsbomb_passes
from football_enigma.data.statsbomb_ids import EURO_2024
from football_enigma.metrics.network import (
    betweenness,
    node_positions,
    pass_edges,
)
from football_enigma.utils.paths import PROCESSED_DIR, figures_dir
from football_enigma.viz.pitch import draw, meter_pitch
from football_enigma.viz.theme import COLORS, apply_theme, credit

SUBJECT = "Toni Kroos"
MIN_EDGE = 15   # drawn edges: pairs with 15+ completed passes over 5 matches
MIN_NODE = 100  # drawn nodes: players with 100+ passes
FIGS = figures_dir("toni-kroos")

apply_theme()

germany_matches = sb_data.team_matches(EURO_2024, "Germany")
passes = pd.concat(
    [
        statsbomb_passes(
            sb_data.decode_locations(sb_data.load_events(int(mid))), int(mid)
        )
        for mid in germany_matches["match_id"]
    ],
    ignore_index=True,
)
passes = passes[passes["team"] == "Germany"]

edges = pass_edges(passes)
central = betweenness(edges)
nodes = node_positions(passes).set_index("player")
nodes = nodes[nodes["passes"] >= MIN_NODE]
nodes["betweenness"] = central

(nodes.reset_index()
 .to_parquet(PROCESSED_DIR / "euro2024_germany_network_nodes.parquet"))
edges.to_parquet(PROCESSED_DIR / "euro2024_germany_network_edges.parquet")

# ---------------------------------------------------------------- fig 20
pitch = meter_pitch(pad_top=14)
fig, ax = draw(pitch, figsize=(11, 7.5))

drawn = edges[
    (edges["count"] >= MIN_EDGE)
    & edges["a"].isin(nodes.index)
    & edges["b"].isin(nodes.index)
]
for _, e in drawn.iterrows():
    a, b = nodes.loc[e["a"]], nodes.loc[e["b"]]
    involves_subject = SUBJECT in (e["a"], e["b"])
    ax.plot(
        [a["x"], b["x"]], [a["y"], b["y"]],
        color=COLORS["subject"] if involves_subject else COLORS["field"],
        alpha=0.75 if involves_subject else 0.35,
        lw=e["count"] / 12,
        zorder=2,
        solid_capstyle="round",
    )

texts = []
for player, n in nodes.iterrows():
    is_subject = player == SUBJECT
    ax.scatter(
        n["x"], n["y"], s=n["passes"] * 0.9,
        color=COLORS["subject"] if is_subject else COLORS["panel"],
        edgecolors=COLORS["text"] if is_subject else COLORS["muted"],
        linewidths=1.4 if is_subject else 0.9, zorder=4,
    )
    label = f"{player.split()[-1]} {n['betweenness']:.2f}"
    if is_subject:
        # pinned clear of the dense midfield cluster, not auto-adjusted
        ax.annotate(
            label, (n["x"], n["y"]), xytext=(-58, 34),
            textcoords="offset points", fontsize=11, fontweight="bold",
            color=COLORS["subject"], zorder=6,
            arrowprops=dict(arrowstyle="-", color=COLORS["muted"], lw=0.9),
        )
        continue
    texts.append(ax.text(
        n["x"], n["y"] - 2.5, label,
        ha="center", va="top", fontsize=8.5,
        color=COLORS["muted"], zorder=5,
    ))
from adjustText import adjust_text

adjust_text(
    texts, ax=ax,
    arrowprops=dict(arrowstyle="-", color=COLORS["grid"], lw=0.8),
)

fig.subplots_adjust(top=0.82)
fig.text(
    0.06, 0.95, "Germany's circulation system — Euro 2024 pass network",
    color=COLORS["text"], fontsize=15, fontweight="bold", ha="left", va="top",
)
fig.text(
    0.06, 0.905,
    f"all five matches | node = median pass origin, size = pass volume, edge = {MIN_EDGE}+ completed passes "
    "| number = betweenness centrality: how often the ball's shortest route runs through a player",
    fontsize=9.5, color=COLORS["muted"], ha="left", va="top",
)
credit(ax, "StatsBomb")
fig.savefig(FIGS / "20_pass_network.png")

print("=== betweenness centrality (Germany, Euro 2024) ===")
print(nodes.sort_values("betweenness", ascending=False)[
    ["passes", "betweenness"]
].round(3).to_string())
print("DONE — figure 20 saved")
