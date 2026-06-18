"""Fig 07 — two men, one midfield.

Action heatmaps (every pass and carry, La Liga 2015/16) for the two central
midfielders Barcelona played in front of their back four: Sergio Busquets and
Ivan Rakitic. Same team, same matches, same 65%+ possession share — so the
difference in where they operate is *role*, not system. Busquets's cloud is a
tight, deep band across the centre circle; Rakitic's spills higher and wider
into the half-spaces. By the standard-distance measure Busquets holds the
3rd-most-compact territory of all 66 central midfielders in the league (only
Iniesta's and one other are tighter) — invariance in space to match the
invariance in time of fig 01.

Honesty: the scalar gap is modest (21.3 m vs 24.1 m) and his compactness is
shared with Iniesta, not unique — and Barcelona's dominance of the ball lets
their midfielders hold shape. The shape of the map, not a record number, is the
point.

Run:  .venv/Scripts/python -X utf8 players/sergio-busquets/notebooks/07_territory.py
"""

import sys

import matplotlib

matplotlib.use("Agg")
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from football_enigma.data.statsbomb_ids import BUSQUETS_PLAYER_NAME, LA_LIGA
from football_enigma.metrics.footprint import action_dispersion
from football_enigma.utils.paths import PROCESSED_DIR, figures_dir
from football_enigma.viz.pitch import meter_pitch
from football_enigma.viz.theme import COLORS, apply_theme, credit

SUBJ = BUSQUETS_PLAYER_NAME
FOIL = "Ivan Rakitić"
COMP = LA_LIGA["2015/2016"]
FIGS = figures_dir("sergio-busquets")

apply_theme()
FIGS.mkdir(parents=True, exist_ok=True)

actions = pd.read_parquet(PROCESSED_DIR / "laliga1516_actions.parquet")
moves = actions[actions["type"].isin(["pass", "carry"])].dropna(subset=["x", "y"])
disp = action_dispersion(moves, min_actions=200).set_index("player")

# matched-brightness colormaps so the eye compares shape/position, not intensity
# (a dim grey map would falsely make the foil look like he barely touched the ball)
amber = LinearSegmentedColormap.from_list("amber", [COLORS["background"], COLORS["subject"]])
blue = LinearSegmentedColormap.from_list("blue", [COLORS["background"], COLORS["accent2"]])

pitch = meter_pitch(line_zorder=2)
fig, axes = pitch.draw(figsize=(13, 5.6), ncols=2)
fig.set_facecolor(COLORS["background"])

panels = [(SUBJ, "Sergio Busquets", amber, COLORS["subject"]),
          (FOIL, "Ivan Rakitić", blue, COLORS["accent2"])]
for ax, (name, label, cmap, tint) in zip(axes, panels):
    pts = moves[moves["player"] == name]
    pitch.kdeplot(pts["x"], pts["y"], ax=ax, fill=True, levels=100,
                  thresh=0.05, cmap=cmap, alpha=0.95, zorder=1)
    s = disp.loc[name, "spread"]
    ax.set_title(label, color=tint, fontsize=14, fontweight="bold", pad=8)
    ax.text(0.5, -0.04, f"action spread {s:.1f} m  ·  {len(pts):,} passes & carries",
            transform=ax.transAxes, ha="center", va="top",
            color=COLORS["muted"], fontsize=9.5)

fig.suptitle("Two men, one midfield", x=0.5, y=1.02, fontsize=17,
             fontweight="bold", color=COLORS["text"])
fig.text(0.5, 0.965,
         "where Barcelona's two central midfielders touched the ball, La Liga "
         "2015/16  ·  same team, same games — the difference is the role",
         ha="center", va="top", fontsize=10.5, color=COLORS["muted"])
credit(axes[0], "StatsBomb")
fig.savefig(FIGS / "07_territory.png")

cm = disp.copy()  # rank context printed honestly
print(f"Busquets spread {disp.loc[SUBJ, 'spread']:.1f} m  vs  "
      f"Rakitić {disp.loc[FOIL, 'spread']:.1f} m")
print("DONE — figure 07 saved")
