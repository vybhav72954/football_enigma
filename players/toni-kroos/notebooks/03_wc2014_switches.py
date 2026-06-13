"""WC 2014: the pitch-stretching map — every Kroos switch of play.

Draws all Kroos switches (completed passes with >= 30 m lateral travel)
from Germany's seven matches on one pitch, with every other German
outfielder's switches as the grey background. The 53-vs-15 gap, made
visible.

Run:  .venv/Scripts/python -X utf8 players/toni-kroos/notebooks/03_wc2014_switches.py
"""

import sys

import matplotlib

matplotlib.use("Agg")
sys.stdout.reconfigure(encoding="utf-8")

import glob

import pandas as pd

from football_enigma.data.schema import opta_passes
from football_enigma.metrics.aggregates import flag_switch
from football_enigma.utils.paths import RAW_DIR, figures_dir
from football_enigma.viz.pitch import draw, meter_pitch, pass_arrows
from football_enigma.viz.theme import COLORS, apply_theme, credit

SUBJECT = "Toni Kroos"
FIGS = figures_dir("toni-kroos")

apply_theme()

files = glob.glob(str(RAW_DIR / "whoscored" / "events" / "*.parquet"))
raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
# the events cache now also holds the Bayern 2013/14 scrape; isolate WC 2014
raw = raw[raw["league"] == "INT-World Cup"]
passes = opta_passes(raw)
germany = passes[passes["team"] == "Germany"].copy()
germany["switch"] = flag_switch(germany)
switches = germany[germany["switch"]]

kroos = switches[switches["player"] == SUBJECT]
others = switches[
    (switches["player"] != SUBJECT) & (switches["player"] != "Manuel Neuer")
]

pitch = meter_pitch()
fig, ax = draw(pitch, figsize=(11, 7.5))
pass_arrows(pitch, ax, others, color=COLORS["field"], alpha=0.35, width=1.2)
pass_arrows(pitch, ax, kroos, color=COLORS["subject"], alpha=0.9, width=2.0)

fig.subplots_adjust(top=0.86)
fig.text(
    0.06, 0.95, "Making the pitch huge — every Kroos switch, World Cup 2014",
    color=COLORS["text"], fontsize=15, fontweight="bold", ha="left", va="top",
)
fig.text(
    0.06, 0.905,
    f"{len(kroos)} switches (completed passes moving the ball ≥30 m across the pitch) "
    f"vs {len(others)} by all other German outfielders combined",
    fontsize=10, color=COLORS["muted"], ha="left", va="top",
)
credit(ax, "Opta via WhoScored")
fig.savefig(FIGS / "05_wc2014_switches_map.png")
print(f"Kroos switches: {len(kroos)} | other German outfielders combined: {len(others)}")
print("DONE — figure 05 saved")
