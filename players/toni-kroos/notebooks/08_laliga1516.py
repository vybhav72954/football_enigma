"""La Liga 2015/16 — Kroos's prime, against the best league in the world.

StatsBomb open data covers this season COMPLETELY (380 matches), so the
prime-Madrid Kroos can be measured against peak Busquets, Iniesta,
Modrić et al. on licensed data with pressure flags.

Fig 12 — risk without the price: touches per turnover (ball security) vs
         progressive passes per 90 (aggression). The video's claim that
         Kroos "breaks the risk-reward logic", on a full league season.
Fig 13 — pressure-resistance: pressured passes faced per 90 vs
         completion% under pressure.

Definitions: touch = pass/carry/shot/dribble/ball-receipt event;
turnover = incomplete pass, miscontrol, or dispossession.

Run:  .venv/Scripts/python -X utf8 players/toni-kroos/notebooks/08_laliga1516.py
"""

import sys

import matplotlib

matplotlib.use("Agg")
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from football_enigma.data import statsbomb as sb_data
from football_enigma.data.build import build_statsbomb_actions, player_minutes
from football_enigma.data.statsbomb_ids import LA_LIGA
from football_enigma.metrics.aggregates import player_pass_aggregates
from football_enigma.metrics.pressure import pressure_split
from football_enigma.utils.paths import PROCESSED_DIR, figures_dir
from football_enigma.viz.charts import crowd_scatter
from football_enigma.viz.theme import apply_theme

SUBJECT = "Toni Kroos"
COMP = LA_LIGA["2015/2016"]
MIN_MINUTES = 1800  # at least 20 full games of a league season
FIGS = figures_dir("toni-kroos")

apply_theme()

actions = build_statsbomb_actions(COMP, "laliga1516")
minutes = player_minutes(COMP)
passes = actions[actions["type"] == "pass"]

# --------------------------- touches & turnovers straight from raw events
TOUCH_TYPES = {"Pass", "Carry", "Shot", "Dribble", "Ball Receipt*"}
TURNOVER_TYPES = {"Miscontrol", "Dispossessed"}

touch_counts: dict[str, int] = {}
turnover_counts: dict[str, int] = {}
matches = sb_data.load_matches(COMP)
for mid in matches["match_id"]:
    events = sb_data.load_events(int(mid))
    on_ball = events[events["type"].isin(TOUCH_TYPES)]
    for player, n in on_ball["player"].value_counts().items():
        touch_counts[player] = touch_counts.get(player, 0) + int(n)
    lost = events[events["type"].isin(TURNOVER_TYPES)]
    for player, n in lost["player"].value_counts().items():
        turnover_counts[player] = turnover_counts.get(player, 0) + int(n)
    bad_pass = events[(events["type"] == "Pass") & events["pass_outcome"].notna()]
    for player, n in bad_pass["player"].value_counts().items():
        turnover_counts[player] = turnover_counts.get(player, 0) + int(n)

security = pd.DataFrame(
    {
        "player": list(touch_counts),
        "touches": [touch_counts[p] for p in touch_counts],
        "turnovers": [turnover_counts.get(p, 0) for p in touch_counts],
    }
)
security["touches_per_turnover"] = security["touches"] / security["turnovers"].clip(lower=1)

agg = player_pass_aggregates(passes, minutes)
agg = agg[agg["minutes"] >= MIN_MINUTES]
agg = agg.merge(security, on="player", how="left")
agg.to_parquet(PROCESSED_DIR / "laliga1516_player_agg.parquet")

# -------------------------------------------------------------- fig 12
mids = agg[agg["passes_p90"] >= 45]  # ball-dominant players only
fig, ax = crowd_scatter(
    mids,
    x="touches_per_turnover",
    y="progressive_p90",
    subject=SUBJECT,
    title="Risk without the price — La Liga 2015/16",
    subtitle="ball-dominant players (45+ passes/90, 1800+ min) | security vs aggression",
    xlabel="Touches per turnover (higher = ball safer)",
    ylabel="Progressive passes per 90",
    annotate=["Sergio Busquets i Burgos", "Andrés Iniesta Luján", "Luka Modrić"],
)
fig.savefig(FIGS / "12_laliga1516_risk_price.png")

# -------------------------------------------------------------- fig 13
pressure = pressure_split(
    passes.merge(agg[["player"]], on="player"), xt_model=None, min_pressured=60
)
pressure = pressure.merge(agg[["player", "minutes"]], on="player")
pressure["pressured_p90"] = pressure["pressured"] / (pressure["minutes"] / 90)
pressure.to_parquet(PROCESSED_DIR / "laliga1516_pressure.parquet")

fig, ax = crowd_scatter(
    pressure,
    x="pressured_p90",
    y="comp_pct_pressured",
    subject=SUBJECT,
    title="The press does not work — La Liga 2015/16",
    subtitle="1800+ min, 60+ pressured passes | burden faced vs completion under pressure",
    xlabel="Pressured passes faced per 90",
    ylabel="Completion % under pressure",
    annotate=["Sergio Busquets i Burgos", "Andrés Iniesta Luján", "Luka Modrić"],
)
fig.savefig(FIGS / "13_laliga1516_pressure.png")

print("=== ball security, ball-dominant players ===")
print(mids.sort_values("touches_per_turnover", ascending=False).head(10)[
    ["player", "minutes", "passes_p90", "touches_per_turnover", "progressive_p90"]
].round(1).to_string(index=False))
kroos = agg[agg["player"] == SUBJECT]
print("\nKroos:", kroos[["passes_p90", "comp_pct", "touches_per_turnover",
                         "progressive_p90", "switches_p90"]].round(2).to_string(index=False))
print("\nDONE — figures 12–13 saved")
