"""Fig 8 — the Euro 2024 percentile radar.

Kroos's tournament percentiles (vs all players with 270+ minutes) on the
axes this essay actually argues about: volume, accuracy, progression,
switches, threat, creation from deep, and press resistance.

Run:  .venv/Scripts/python -X utf8 players/toni-kroos/notebooks/06_radar.py
"""

import sys

import matplotlib

matplotlib.use("Agg")
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from mplsoccer import Radar

from football_enigma.utils.paths import PROCESSED_DIR, figures_dir
from football_enigma.viz.theme import COLORS, apply_theme

SUBJECT = "Toni Kroos"
FIGS = figures_dir("toni-kroos")

apply_theme()

agg = pd.read_parquet(PROCESSED_DIR / "euro2024_player_agg.parquet")
pressure = pd.read_parquet(PROCESSED_DIR / "euro2024_pressure.parquet")
danger = pd.read_parquet(PROCESSED_DIR / "euro2024_danger_depth.parquet")

df = (
    agg.merge(
        pressure[["player", "comp_pct_pressured"]], on="player", how="left"
    ).merge(
        danger[["player", "shot_assists_p90"]], on="player", how="left"
    )
)

metrics = {
    "passes_p90": "Pass volume /90",
    "comp_pct": "Completion %",
    "progressive_p90": "Progressive /90",
    "switches_p90": "Switches /90",
    "xt_p90": "xT added /90",
    "shot_assists_p90": "Shots created /90",
    "comp_pct_pressured": "Completion %\nunder pressure",
}

pct = df.copy()
for col in metrics:
    pct[col] = df[col].rank(pct=True) * 100

kroos = pct[pct["player"] == SUBJECT].iloc[0]
values = [float(kroos[c]) for c in metrics]
print("Kroos percentiles:", {v: round(x) for v, x in zip(metrics.values(), values)})

radar = Radar(
    params=list(metrics.values()),
    min_range=[0] * len(metrics),
    max_range=[100] * len(metrics),
)
fig, ax = radar.setup_axis(figsize=(9, 9), facecolor=COLORS["background"])
radar.draw_circles(
    ax=ax, facecolor=COLORS["panel"], edgecolor=COLORS["grid"]
)
radar.draw_radar(
    values,
    ax=ax,
    kwargs_radar={"facecolor": COLORS["subject"], "alpha": 0.55},
    kwargs_rings={"facecolor": COLORS["subject"], "alpha": 0.12},
)
radar.draw_range_labels(ax=ax, fontsize=8, color=COLORS["muted"])
radar.draw_param_labels(ax=ax, fontsize=11, color=COLORS["text"])

fig.subplots_adjust(top=0.86)
fig.text(
    0.5, 0.965, "Toni Kroos — Euro 2024 tournament percentiles",
    ha="center", va="top", color=COLORS["text"], fontsize=15, fontweight="bold",
)
fig.text(
    0.5, 0.925,
    "vs all players with 270+ minutes  |  Football Enigma  |  Data: StatsBomb",
    ha="center", va="top", fontsize=9.5, color=COLORS["muted"],
)
fig.savefig(FIGS / "08_radar.png")
print("DONE — figure 08 saved")
