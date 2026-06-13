"""The three-peat finals (2016, 2017, 2018): the ball in a safe.

Champions League finals are the highest-leverage club matches that exist.
Across Real Madrid's three consecutive wins we measure every player who
appeared, and ask who kept the ball — and kept it moving — when pressed.

Fig 6 — crowd scatter: pressured passes faced vs completion% under
pressure, all players across the three finals pooled.

Run:  .venv/Scripts/python -X utf8 players/toni-kroos/notebooks/04_cl_finals.py
"""

import sys

import matplotlib

matplotlib.use("Agg")
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from football_enigma.data import statsbomb as sb_data
from football_enigma.data.schema import statsbomb_passes
from football_enigma.data.statsbomb_ids import CL_2015_16, CL_2016_17, CL_2017_18
from football_enigma.metrics.aggregates import flag_progressive, flag_switch
from football_enigma.utils.paths import PROCESSED_DIR, figures_dir
from football_enigma.viz.charts import crowd_scatter
from football_enigma.viz.theme import apply_theme

SUBJECT = "Toni Kroos"
FIGS = figures_dir("toni-kroos")

apply_theme()

chunks = []
for comp in (CL_2015_16, CL_2016_17, CL_2017_18):
    matches = sb_data.load_matches(comp)
    for mid in matches["match_id"]:
        events = sb_data.decode_locations(sb_data.load_events(int(mid)))
        chunks.append(statsbomb_passes(events, int(mid)))
passes = pd.concat(chunks, ignore_index=True)

passes["progressive"] = flag_progressive(passes)
passes["switch"] = flag_switch(passes)

per_player = (
    passes.groupby("player", observed=True)
    .agg(
        passes=("outcome", "size"),
        comp_pct=("outcome", lambda s: s.mean() * 100),
        pressured=("under_pressure", "sum"),
        progressive=("progressive", "sum"),
        switches=("switch", "sum"),
    )
    .reset_index()
)
pressured = (
    passes[passes["under_pressure"]]
    .groupby("player", observed=True)["outcome"]
    .agg(comp_pct_pressured=lambda s: s.mean() * 100, pressured_n="size")
    .reset_index()
)
per_player = per_player.merge(pressured, on="player", how="left")
per_player = per_player[per_player["passes"] >= 60]  # ~meaningful minutes across finals
per_player.to_parquet(PROCESSED_DIR / "cl_finals_passing.parquet")

# NOTE: pressure splits across only 3 matches are too thin to chart
# honestly (Kroos: 13 pressured passes) — volume + accuracy is the
# defensible finals story. Pressure analysis lives in the Euro 2024
# section where samples are real.
fig, ax = crowd_scatter(
    per_player,
    x="passes",
    y="comp_pct",
    subject=SUBJECT,
    title="Three finals, one constant — who the ball trusted",
    subtitle="CL finals 2016, 2017, 2018 pooled | every player with 60+ passes",
    xlabel="Passes attempted across the three finals",
    ylabel="Completion %",
    annotate=["Luka Modrić", "Marcelo Vieira da Silva Júnior", "Carlos Henrique Casimiro"],
)
fig.savefig(FIGS / "06_cl_finals_volume.png")

print("=== CL finals pooled, 60+ passes, sorted by pressured completion ===")
cols = ["player", "passes", "comp_pct", "pressured_n", "comp_pct_pressured",
        "progressive", "switches"]
print(
    per_player[per_player["pressured_n"] >= 15]
    .sort_values("comp_pct_pressured", ascending=False)
    .head(14)[cols]
    .round(1)
    .to_string(index=False)
)
print("\nDONE — figure 06 saved")
