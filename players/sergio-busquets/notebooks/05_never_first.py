"""Fig 05 — always near the top, never the top.

La Liga 2015/16 (202 players, 1500+ min). On every instrument he owns,
Busquets ranks 4th-8th: elite, but never first. The crowns belong to
specialists — and three of these six go to Toni Kroos, the subject of Post #1,
who sat one rung above Busquets on the very boards the analytics reward. You
cannot build a leaderboard out of a player who is everywhere near the top and
nowhere at it.

Run:  .venv/Scripts/python -X utf8 players/sergio-busquets/notebooks/05_never_first.py
"""

import sys

import matplotlib

matplotlib.use("Agg")
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from football_enigma.data.statsbomb_ids import BUSQUETS_PLAYER_NAME
from football_enigma.utils.paths import PROCESSED_DIR, figures_dir
from football_enigma.viz.charts import _short_name
from football_enigma.viz.theme import COLORS, apply_theme, credit

SUBJ = BUSQUETS_PLAYER_NAME
FIGS = figures_dir("sergio-busquets")

AXES = {
    "passes_p90": "Pass volume",
    "comp_pct": "Completion %",
    "comp_pct_pressured": "Completion under pressure",
    "touches_per_turnover": "Ball security",
    "switches_p90": "Switches",
    "final_third_p90": "Final-third entries",
}

apply_theme()
FIGS.mkdir(parents=True, exist_ok=True)

agg = pd.read_parquet(PROCESSED_DIR / "laliga1516_player_agg.parquet")
pr = pd.read_parquet(PROCESSED_DIR / "laliga1516_pressure.parquet")
df = agg.merge(pr[["player", "comp_pct_pressured"]], on="player", how="left")
df = df[df["minutes"] >= 1500].copy()
b = df[df["player"] == SUBJ].iloc[0]

rows = []
for c, lbl in AXES.items():
    d = df.dropna(subset=[c])
    rank = int((d[c] > b[c]).sum() + 1)
    top = d.sort_values(c, ascending=False).iloc[0]["player"]
    rows.append({"axis": lbl, "rank": rank, "top": _short_name(top)})
r = pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)  # first axis on top

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10.5, 6))
y = range(len(r))
ax.hlines(y, 1, r["rank"], color=COLORS["grid"], lw=2, zorder=1)
ax.scatter(r["rank"], y, s=180, color=COLORS["subject"], zorder=3,
           edgecolors=COLORS["background"], linewidths=1.2)
ax.scatter([1] * len(r), y, s=90, color=COLORS["field"], zorder=2)

for i, row in r.iterrows():
    ax.text(row["rank"] + 0.18, i, f"Busquets · {row['rank']}",
            va="center", ha="left", color=COLORS["subject"],
            fontsize=10, fontweight="bold")
    ax.text(0.82, i, f"#1 {row['top']}", va="center", ha="right",
            color=COLORS["muted"], fontsize=9)

ax.set_yticks(list(y))
ax.set_yticklabels(r["axis"], fontsize=11)
ax.set_xlim(-2.2, 11)
ax.set_xlabel("Rank among La Liga players (1500+ min)  ·  1 = best")
ax.set_xticks([1, 3, 5, 7, 9])
ax.grid(axis="y", visible=False)
ax.set_title("Always near the top, never the top", loc="left", pad=30)
ax.text(0, 1.03,
        "Busquets's rank on the instruments he owns, La Liga 2015/16 (202 "
        "players)  |  the crowns belong to others — three to Kroos",
        transform=ax.transAxes, fontsize=10, color=COLORS["muted"], va="bottom")
credit(ax, "StatsBomb")
fig.savefig(FIGS / "05_never_first.png")

print(r.to_string(index=False))
print("DONE — figure 05 saved")
