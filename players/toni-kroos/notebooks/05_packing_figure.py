"""Fig 7 — packing: defenders taken out of the game per pass.

Horizontal bar leaderboard of Germany's Euro 2024 squad by packing rate
(defenders bypassed per framed pass, 360 freeze-frames), with total
defenders bypassed annotated. Kroos in amber.

Run:  .venv/Scripts/python -X utf8 players/toni-kroos/notebooks/05_packing_figure.py
"""

import sys

import matplotlib

matplotlib.use("Agg")
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import pandas as pd

from football_enigma.utils.paths import PROCESSED_DIR, figures_dir
from football_enigma.viz.theme import COLORS, apply_theme, credit

SUBJECT = "Toni Kroos"
FIGS = figures_dir("toni-kroos")

apply_theme()

packing = pd.read_parquet(PROCESSED_DIR / "euro2024_germany_packing.parquet")
rank = (
    packing.groupby("player", observed=True)["packing"]
    .agg(total="sum", framed_passes="count", rate="mean")
    .query("framed_passes >= 100")
    .sort_values("rate")
    .reset_index()
)

fig, ax = plt.subplots(figsize=(10, 6.5))
colors = [
    COLORS["subject"] if p == SUBJECT else COLORS["field"] for p in rank["player"]
]
bars = ax.barh(rank["player"], rank["rate"], color=colors, height=0.62)
for bar, (_, row) in zip(bars, rank.iterrows()):
    ax.text(
        bar.get_width() + 0.03,
        bar.get_y() + bar.get_height() / 2,
        f"{row['rate']:.2f}  ({int(row['total'])} total)",
        va="center",
        fontsize=9,
        color=COLORS["text"] if row["player"] == SUBJECT else COLORS["muted"],
        fontweight="bold" if row["player"] == SUBJECT else "normal",
    )

ax.set_xlabel("Defenders bypassed per completed pass (360 freeze-frames)")
ax.set_xlim(0, rank["rate"].max() * 1.28)
ax.grid(axis="y", visible=False)
ax.set_title(
    "Line-breaking, measured with real defender positions",
    loc="left", pad=20,
)
ax.text(
    0, 1.02,
    "Germany at Euro 2024 | players with 100+ framed completed passes | "
    "lower bound: only camera-visible defenders count",
    transform=ax.transAxes, fontsize=9.5, color=COLORS["muted"], va="bottom",
)
credit(ax, "StatsBomb 360")
fig.savefig(FIGS / "07_packing_leaderboard.png")
print(rank.sort_values("rate", ascending=False).round(2).to_string(index=False))
print("DONE — figure 07 saved")
