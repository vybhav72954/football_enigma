"""Euro 2024: first pass at the flagship analysis.

Builds the canonical action set for the whole tournament, fits xT, and
produces the first real figures:
  1. crowd scatter — pass volume p90 vs completion% (the "numbers say
     average" intro chart, except at Euro 2024 he isn't average)
  2. crowd scatter — pressured volume vs completion under pressure
  3. pressure-resistance table (completion drop under pressure)
  4. packing leaderboard from 360 frames (Germany matches)

Run:  .venv/Scripts/python players/toni-kroos/notebooks/01_euro2024.py
"""

import sys

import matplotlib

matplotlib.use("Agg")
sys.stdout.reconfigure(encoding="utf-8")  # Windows cp1252 console vs player names

import pandas as pd

from football_enigma.data import statsbomb as sb_data
from football_enigma.data.build import build_statsbomb_actions, player_minutes
from football_enigma.data.statsbomb_ids import EURO_2024
from football_enigma.metrics.aggregates import player_pass_aggregates
from football_enigma.metrics.packing import match_packing
from football_enigma.metrics.pressure import pressure_split
from football_enigma.metrics.xt import fit_xt
from football_enigma.utils.paths import PROCESSED_DIR, figures_dir
from football_enigma.viz.charts import crowd_scatter
from football_enigma.viz.theme import apply_theme

SUBJECT = "Toni Kroos"
MIN_MINUTES = 270  # at least 3 full games to qualify for comparisons
FIGS = figures_dir("toni-kroos")

apply_theme()

# ---------------------------------------------------------------- data
actions = build_statsbomb_actions(EURO_2024, "euro2024")
minutes = player_minutes(EURO_2024)
passes = actions[actions["type"] == "pass"]

xt_model = fit_xt(actions)

# ------------------------------------------------- per-player aggregates
agg = player_pass_aggregates(passes, minutes)
agg = agg[agg["minutes"] >= MIN_MINUTES]

xt_vals = xt_model.value(actions)
xt_per_player = (
    actions.assign(xt=xt_vals)
    .groupby("player", observed=True)["xt"]
    .sum()
    .rename("xt_total")
)
agg = agg.merge(xt_per_player, on="player", how="left")
agg["xt_p90"] = agg["xt_total"] / (agg["minutes"] / 90)

agg.to_parquet(PROCESSED_DIR / "euro2024_player_agg.parquet")

# --------------------------------------------------------------- fig 1
fig, ax = crowd_scatter(
    agg,
    x="passes_p90",
    y="comp_pct",
    subject=SUBJECT,
    title="Everyone is accurate now — volume is the separator",
    subtitle="Euro 2024, players with 270+ minutes | pass volume vs completion",
    xlabel="Passes per 90",
    ylabel="Completion %",
    annotate=["Rodrigo Hernández Cascante", "Declan Rice", "Granit Xhaka"],
)
fig.savefig(FIGS / "01_volume_vs_completion.png")

# ----------------------------------------------------- pressure split
pressure = pressure_split(
    passes.merge(agg[["player"]], on="player"), xt_model, min_pressured=30
)
pressure.sort_values("completion_drop").to_parquet(
    PROCESSED_DIR / "euro2024_pressure.parquet"
)

# --------------------------------------- fig 2: pressure resistance
pressure_fig = pressure.merge(agg[["player", "minutes"]], on="player")
pressure_fig["pressured_p90"] = pressure_fig["pressured"] / (
    pressure_fig["minutes"] / 90
)
fig, ax = crowd_scatter(
    pressure_fig,
    x="pressured_p90",
    y="comp_pct_pressured",
    subject=SUBJECT,
    title="The press does not work",
    subtitle="Euro 2024, 270+ min & 30+ pressured passes | burden faced vs accuracy when pressed",
    xlabel="Pressured passes faced per 90",
    ylabel="Completion % under pressure",
    annotate=["Rodrigo Hernández Cascante", "Joshua Kimmich", "Bruno Miguel Borges Fernandes"],
)
fig.savefig(FIGS / "02_pressure_resistance.png")
print("\n=== Pressure resistance (lowest completion drop) ===")
print(
    pressure.sort_values("completion_drop")
    .head(12)[["player", "passes", "pressured", "comp_pct_pressured",
               "comp_pct_unpressured", "completion_drop"]]
    .to_string(index=False)
)
kroos_row = pressure[pressure["player"] == SUBJECT]
print("\nKroos:", kroos_row.to_string(index=False) if not kroos_row.empty else "below threshold")

# ------------------------------------------------------------ packing
germany = sb_data.team_matches(EURO_2024, "Germany")
packing_chunks = []
for mid in germany["match_id"]:
    events = sb_data.decode_locations(sb_data.load_events(int(mid)))
    frames = sb_data.load_frames(int(mid))
    if frames:
        packing_chunks.append(match_packing(events, frames))
packing = pd.concat(packing_chunks, ignore_index=True)
packing.to_parquet(PROCESSED_DIR / "euro2024_germany_packing.parquet")

pack_rank = (
    packing.groupby("player", observed=True)["packing"]
    .agg(["sum", "count", "mean"])
    .query("count >= 100")
    .sort_values("sum", ascending=False)
)
print("\n=== Packing (defenders bypassed), Germany matches, 100+ framed passes ===")
print(pack_rank.head(12).to_string())

print("\nDONE — figures in", FIGS)
