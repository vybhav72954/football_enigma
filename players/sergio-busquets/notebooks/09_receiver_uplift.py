"""Fig 09 — the receiver uplift: teammates unlock more after a Busquets feed.

The decisive answer to "he's just near better players". For each Barca receiver,
two rates: how often THEIR next pass is the unlock after receiving a simple ball
from Busquets, versus after the same simple ball from any other Barca player. The
same world-class names — Iniesta, Messi — get measurably more dangerous
*specifically* after his feed. La Liga 2015/16, simple (non-progressive) passes,
receivers with >=40 feeds from each side. Honest: not everyone lifts (Alves dips)
— the effect is real, not universal.

Run:  .venv/Scripts/python -X utf8 players/sergio-busquets/notebooks/09_receiver_uplift.py
"""

import sys

import matplotlib

matplotlib.use("Agg")
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import pandas as pd

from football_enigma.utils.paths import PROCESSED_DIR, figures_dir
from football_enigma.viz.theme import COLORS, apply_theme, credit

FIGS = figures_dir("sergio-busquets")
apply_theme()
FIGS.mkdir(parents=True, exist_ok=True)

upl = pd.read_parquet(PROCESSED_DIR / "laliga1516_receiver_uplift.parquet")
upl = upl.sort_values("uplift").reset_index(drop=True)  # biggest uplift on top
team_mean = upl["after_busq"].mean(), upl["after_other"].mean()

fig, ax = plt.subplots(figsize=(10.5, 0.5 * len(upl) + 2.2))
y = range(len(upl))
for i, r in upl.iterrows():
    up = r["uplift"] >= 0
    ax.plot([r["after_other"], r["after_busq"]], [i, i],
            color=COLORS["subject"] if up else COLORS["negative"],
            lw=2.2, alpha=0.55, zorder=1)
    ax.scatter(r["after_other"], i, s=90, color=COLORS["field"], zorder=3,
               edgecolors=COLORS["background"], linewidths=1)
    ax.scatter(r["after_busq"], i, s=120,
               color=COLORS["subject"] if up else COLORS["negative"], zorder=4,
               edgecolors=COLORS["background"], linewidths=1)
    ax.text(max(r["after_busq"], r["after_other"]) + 0.6, i,
            f"{r['uplift']:+.1f} pp", va="center", ha="left",
            color=COLORS["subject"] if up else COLORS["negative"], fontsize=9)

ax.set_yticks(list(y))
ax.set_yticklabels(upl["receiver"], fontsize=10)
ax.set_xlabel("Recipient's next pass is the unlock (%)")
ax.grid(axis="y", visible=False)
ax.set_xlim(0, upl[["after_busq", "after_other"]].max().max() * 1.18)
ax.set_title("The receiver uplift: more dangerous after a Busquets feed", loc="left", pad=30)
ax.text(0, 1.04,
        "Amber = after a simple Busquets pass · grey = after any other Barça player's · "
        "La Liga 2015/16",
        transform=ax.transAxes, fontsize=9.5, color=COLORS["muted"], va="bottom")
# legend-ish markers
ax.scatter([], [], s=90, color=COLORS["field"], label="after other Barça feed")
ax.scatter([], [], s=120, color=COLORS["subject"], label="after Busquets feed")
ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=COLORS["muted"])
credit(ax, "StatsBomb")
fig.savefig(FIGS / "09_receiver_uplift.png")

print(f"team avg: after Busquets {team_mean[0]:.1f}% vs after others {team_mean[1]:.1f}%")
print(upl[["receiver", "after_busq", "after_other", "uplift"]].round(1).to_string(index=False))
print("DONE — figure 09 saved")
