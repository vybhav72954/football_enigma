"""Fig 10 — the crown held: third-man creation across thirteen Barca seasons.

Replication / persistence check (#10 of the robustness battery). The 2015/16
crown is not a one-season spike: run the same third-man connector metric across
every cached Barca La Liga season 2008/09-2020/21 and his per-90 level barely
moves — top-3 among the Barca central mids on show every single season, #1 in
four, median 10.0/90 (range 8.25-11.44). The crown and the invariance thesis are
the same number: the thing about him that never changes IS the thing that makes
him singular. 2015/16 is the only complete-division open-data season (pool 67,
the full 380-match La Liga); the rest are Barca's match slates, so the honest
claim there is the stability of his level and his rank among the Barca mids
present.

Run:  .venv/Scripts/python -X utf8 players/sergio-busquets/notebooks/10_persistence.py
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

car = pd.read_parquet(PROCESSED_DIR / "busquets_thirdman_career.parquet")
car = car.sort_values("season").reset_index(drop=True)
x = range(len(car))
labels = [s[2:4] + "/" + s[7:9] for s in car["season"]]
median = car["busq_per90"].median()

fig, ax = plt.subplots(figsize=(11, 6.2))

# career median — the "one number" the crown keeps returning to
ax.axhline(median, color=COLORS["subject"], lw=1, ls="--", alpha=0.45, zorder=1)
ax.text(len(car) - 0.5, median + 0.14, f"career median {median:.1f}/90",
        color=COLORS["subject"], fontsize=9, ha="right", va="bottom", style="italic")

# the flat amber trajectory: #1 seasons get a haloed marker
ax.plot(x, car["busq_per90"], color=COLORS["subject"], lw=2, alpha=0.8, zorder=2)
for i, r in car.iterrows():
    top = r["rank"] == 1
    ax.scatter(i, r["busq_per90"], s=190 if top else 110,
               color=COLORS["subject"], zorder=4,
               edgecolors=COLORS["text"] if top else COLORS["background"],
               linewidths=1.5 if top else 1)
    ax.text(i, r["busq_per90"] + 0.34, f"#{int(r['rank'])}", ha="center",
            va="bottom", color=COLORS["subject"] if top else COLORS["muted"],
            fontsize=9, fontweight="bold" if top else "normal")

# anchor: the only full-division season — where the crown was established
anchor = int(car.index[car["season"] == "2015/2016"][0])
ax.annotate("full La Liga · #1 of 67",
            (anchor, car.loc[anchor, "busq_per90"]),
            xytext=(anchor, 7.1), textcoords="data", ha="center",
            fontsize=8.5, color=COLORS["muted"],
            arrowprops=dict(arrowstyle="-", color=COLORS["muted"], lw=0.8, alpha=0.6))

ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(6.5, 13)
ax.set_ylabel("Third-man connectors per 90")
ax.set_xlabel("La Liga season  ·  Barça")
ax.grid(axis="x", visible=False)
ax.set_title("The crown held: thirteen seasons, one number", loc="left", pad=30)
ax.text(0, 1.035,
        "The third-man metric, replicated every Barça season 2008–2021 · top-3 "
        "of the Barça mids on show all 13 years, #1 in four (amber halo)",
        transform=ax.transAxes, fontsize=9.5, color=COLORS["muted"], va="bottom")
credit(ax, "StatsBomb")
fig.savefig(FIGS / "10_persistence.png")

n1 = int((car["rank"] == 1).sum())
print(f"top-3 {int((car['rank'] <= 3).sum())}/{len(car)}; #1 {n1}/{len(car)}; "
      f"per90 {car['busq_per90'].min():.2f}-{car['busq_per90'].max():.2f} "
      f"median {median:.2f}")
print("DONE — figure 10 saved")
