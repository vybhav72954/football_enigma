"""Euro 2024: the metronome chart and the quarterback chart.

Fig 3 — "measuring silence": median vs mean gap between on-ball
involvements. Down-left = the game never leaves you. The video built this
by hand for WC 2014; here it is from StatsBomb events at Euro 2024.

Fig 4 — "danger from deep": how deep a player operates (median action
start, meters from own goal line) vs the threat they add (xT per 90).
Attackers earn xT by living high; defenders live deep and add none.
A deep operator with attacker-grade xT breaks the pattern.

Run:  .venv/Scripts/python -X utf8 players/toni-kroos/notebooks/02_cadence_quarterback.py
"""

import sys

import matplotlib

matplotlib.use("Agg")
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from football_enigma.data.build import build_statsbomb_actions, player_minutes
from football_enigma.data.statsbomb_ids import EURO_2024
from football_enigma.metrics.cadence import cadence_summary
from football_enigma.metrics.xt import fit_xt
from football_enigma.utils.paths import PROCESSED_DIR, figures_dir
from football_enigma.viz.charts import crowd_scatter
from football_enigma.viz.theme import apply_theme

SUBJECT = "Toni Kroos"
MIN_MINUTES = 270
FIGS = figures_dir("toni-kroos")

apply_theme()

actions = build_statsbomb_actions(EURO_2024, "euro2024")
minutes = player_minutes(EURO_2024)
qualified = minutes[minutes >= MIN_MINUTES].index

# ------------------------------------------------------- fig 3: cadence
cadence = cadence_summary(actions, min_involvements=150)
cadence = cadence[cadence["player"].isin(qualified)]
cadence.to_parquet(PROCESSED_DIR / "euro2024_cadence.parquet")

fig, ax = crowd_scatter(
    cadence,
    x="median_gap",
    y="mean_gap",
    subject=SUBJECT,
    title="Measuring silence — who is never out of the game",
    subtitle="Euro 2024, 150+ involvements | gap between on-ball actions (s)",
    xlabel="Median gap (s) — usual involvement rhythm",
    ylabel="Mean gap (s) — punished by disappearances",
    annotate=["Rodrigo Hernández Cascante", "Declan Rice", "Harry Kane"],
)
fig.savefig(FIGS / "03_measuring_silence.png")

kroos = cadence[cadence["player"] == SUBJECT]
print("=== Cadence ===")
print(cadence.sort_values("mean_gap").head(8).to_string(index=False))
print("\nKroos:", kroos.to_string(index=False))

# --------------------------------------------- fig 4: danger from deep
passes = actions[actions["type"] == "pass"]
per_player = (
    passes.groupby("player", observed=True)
    .agg(depth=("x", "median"), shot_assists=("shot_assist", "sum"))
)
per_player = per_player.join(minutes.rename("minutes"), how="inner")
per_player = per_player[per_player["minutes"] >= MIN_MINUTES]
per_player["shot_assists_p90"] = per_player["shot_assists"] / (
    per_player["minutes"] / 90
)
per_player = per_player.reset_index()
per_player.to_parquet(PROCESSED_DIR / "euro2024_danger_depth.parquet")

fig, ax = crowd_scatter(
    per_player,
    x="depth",
    y="shot_assists_p90",
    subject=SUBJECT,
    title="Throwing touchdowns — chances created from deep",
    subtitle="Euro 2024, 270+ min | where a player passes from vs shots they create",
    xlabel="Median pass origin (m from own goal line) — left = deeper",
    ylabel="Passes leading directly to a shot, per 90",
    annotate=["Rodrigo Hernández Cascante", "Lamine Yamal Nasraoui Ebana", "Kevin De Bruyne"],
)
fig.savefig(FIGS / "04_danger_from_deep.png")

print("\n=== Chances created from deep (own-half passers, by shot assists p90) ===")
deep = per_player[per_player["depth"] <= 52.5]
print(deep.sort_values("shot_assists_p90", ascending=False).head(10)[
    ["player", "depth", "shot_assists_p90", "minutes"]].to_string(index=False))

print("\nDONE — figures in", FIGS)
