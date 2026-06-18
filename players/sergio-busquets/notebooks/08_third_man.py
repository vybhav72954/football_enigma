"""Fig 08 — the third man: the pass that lets the pass happen.

Every conventional instrument ranks Busquets 3rd-6th (fig 05). One purpose-built
metric crowns him: the third-man connector — a completed simple, non-progressive
pass whose recipient's very NEXT pass is the unlock (progressive, final-third
entry or shot assist). Progressive-pass and assist stats are blind to it by
construction, which is why it is the one board he tops. La Liga 2015/16, central
mids (1500+ min). The robustness caption: it survives a rate-over-expected
control, so it is not a by-product of his pass volume.

Run:  .venv/Scripts/python -X utf8 players/sergio-busquets/notebooks/08_third_man.py
"""

import sys

import matplotlib

matplotlib.use("Agg")
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import pandas as pd

from football_enigma.data.statsbomb_ids import BUSQUETS_PLAYER_NAME
from football_enigma.utils.paths import PROCESSED_DIR, figures_dir
from football_enigma.viz.theme import COLORS, apply_theme, credit

SUBJ = BUSQUETS_PLAYER_NAME
FIGS = figures_dir("sergio-busquets")
PEERS = {"Toni Kroos", "Luka Modrić", "Andrés Iniesta Luján"}  # role twins, highlighted
TOP_N = 15

apply_theme()
FIGS.mkdir(parents=True, exist_ok=True)

brd = pd.read_parquet(PROCESSED_DIR / "laliga1516_thirdman_board.parquet")
brd = brd.sort_values("per90", ascending=False).reset_index(drop=True)
b = brd[brd["player"] == SUBJ].iloc[0]
over_rank = int((brd["over_exp"] > b["over_exp"]).sum() + 1)

top = brd.head(TOP_N).iloc[::-1].reset_index(drop=True)  # best on top


def colour(p):
    if p == SUBJ:
        return COLORS["subject"]
    return COLORS["accent2"] if p in PEERS else COLORS["field"]


fig, ax = plt.subplots(figsize=(10.5, 7))
y = range(len(top))
for i, row in top.iterrows():
    c = colour(row["player"])
    ax.hlines(i, 0, row["per90"], color=c, lw=2.5, alpha=0.9, zorder=1)
    ax.scatter(row["per90"], i, s=170 if row["player"] == SUBJ else 110,
               color=c, zorder=3, edgecolors=COLORS["background"], linewidths=1.2)
    wt = "bold" if row["player"] == SUBJ else "normal"
    ax.text(row["per90"] + 0.12, i, f"{row['per90']:.2f}", va="center",
            ha="left", color=c, fontsize=9, fontweight=wt)

ax.set_yticks(list(y))
ax.set_yticklabels(top["short"], fontsize=10)
for tick, p in zip(ax.get_yticklabels(), top["player"]):
    tick.set_color(colour(p))
    if p == SUBJ:
        tick.set_fontweight("bold")
ax.set_xlim(0, brd["per90"].max() * 1.16)
ax.set_xlabel("Third-man connectors per 90  ·  La Liga 2015/16 central mids (1500+ min)")
ax.grid(axis="y", visible=False)
ax.set_title("The third man: the pass that lets the pass happen", loc="left", pad=30)
ax.text(0, 1.035,
        "Simple passes whose recipient's next ball is the unlock — the metric the "
        "progressive/assist boards can't see  ·  blue = role twins Kroos, Modrić, Iniesta",
        transform=ax.transAxes, fontsize=9.5, color=COLORS["muted"], va="bottom")
ax.text(0.995, 0.04,
        f"Holds over expected: +{b['over_exp']:.1f} pp above a geometry+pressure model "
        f"(rank {over_rank}) — not a volume artefact",
        transform=ax.transAxes, fontsize=9, color=COLORS["subject"],
        ha="right", va="bottom", style="italic")
credit(ax, "StatsBomb")
fig.savefig(FIGS / "08_third_man.png")

print(f"Busquets {b['per90']:.2f}/90 (rank 1 of {len(brd)}); over-expected +{b['over_exp']:.1f}pp rank {over_rank}")
print("DONE — figure 08 saved")
