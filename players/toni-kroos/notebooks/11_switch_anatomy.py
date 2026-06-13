"""WC 2014 switch anatomy: open play vs dead balls.

The strongest skeptic attack on the 53-switch headline: "he took every
German free kick — those are set-piece deliveries, not vision." Opta's
pass qualifiers say whether each switch was a dead-ball restart (free
kick, corner, throw-in...) or came in open play.

Fig 17 — the switch leaderboard (fig 11) re-drawn with every bar split
into open-play and set-piece segments.

Run:  .venv/Scripts/python -X utf8 players/toni-kroos/notebooks/11_switch_anatomy.py
"""

import sys

import matplotlib

matplotlib.use("Agg")
sys.stdout.reconfigure(encoding="utf-8")

import glob

import matplotlib.pyplot as plt
import pandas as pd

from football_enigma.data.schema import opta_passes
from football_enigma.metrics.aggregates import flag_switch
from football_enigma.utils.paths import RAW_DIR, figures_dir
from football_enigma.viz.theme import COLORS, apply_theme, credit

SUBJECT = "Toni Kroos"
MIN_MATCHES = 4
FIGS = figures_dir("toni-kroos")

apply_theme()

files = glob.glob(str(RAW_DIR / "whoscored" / "events" / "*.parquet"))
raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
# the events cache now also holds the Bayern 2013/14 scrape; isolate WC 2014
raw = raw[raw["league"] == "INT-World Cup"]
passes = opta_passes(raw)
passes["switch"] = flag_switch(passes)
switches = passes[passes["switch"]]

games = passes.groupby("player", observed=True)["match_id"].nunique()
split = (
    switches.groupby("player", observed=True)
    .agg(switches=("switch", "sum"), set_piece=("set_piece", "sum"))
    .join(games.rename("games"))
    .query("games >= @MIN_MATCHES")
)
split["open_play"] = split["switches"] - split["set_piece"]
top = split.sort_values("switches", ascending=False).head(10).iloc[::-1]

# ---------------------------------------------------------------- fig 17
fig, ax = plt.subplots(figsize=(10, 6.5))
colors = [
    COLORS["subject"] if p == SUBJECT else COLORS["field"] for p in top.index
]
ax.barh(top.index, top["open_play"], color=colors, height=0.62)
ax.barh(top.index, top["set_piece"], left=top["open_play"], color=colors,
        height=0.62, alpha=0.35, hatch="///",
        edgecolor=COLORS["background"], linewidth=0)
for y, (player, row) in enumerate(top.iterrows()):
    is_subject = player == SUBJECT
    ax.text(
        row["switches"] + 0.6, y,
        f"{int(row['open_play'])} + {int(row['set_piece'])}",
        va="center", fontsize=10,
        color=COLORS["text"] if is_subject else COLORS["muted"],
        fontweight="bold" if is_subject else "normal",
    )
ax.set_xlabel(
    "Switches of play — solid: open play, hatched: dead-ball restarts "
    "(free kicks, corners, throw-ins)"
)
ax.grid(axis="y", visible=False)
ax.set_title("The switches were not set-piece freebies", loc="left", pad=20)
ax.text(
    0, 1.02,
    "World Cup 2014, players with 4+ matches | labels: open play + dead ball",
    transform=ax.transAxes, fontsize=9.5, color=COLORS["muted"], va="bottom",
)
credit(ax, "Opta via WhoScored")
fig.savefig(FIGS / "17_wc2014_switch_anatomy.png")

kroos = switches[switches["player"] == SUBJECT]
print(f"Kroos: {len(kroos)} switches, {int(kroos['set_piece'].sum())} from dead balls")
print("\n=== leaderboard split ===")
print(top.iloc[::-1][["games", "switches", "open_play", "set_piece"]]
      .to_string())
print("DONE — figure 17 saved")
