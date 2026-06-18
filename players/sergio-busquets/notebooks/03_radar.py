"""Fig 03 — the two halves of a midfielder (vs Kroos's detonating radar).

La Liga 2015/16 percentiles (vs all 1500+ min players) split into the axes
that make a midfielder *famous* — goals, assists, shots created, progressive
passing — and the axes Busquets actually owns — pass volume, completion,
completion under pressure, ball security. The glamour half is quiet; the
fundamentals half is maxed. Where Kroos's radar detonated on progression and
threat, Busquets's fills the half nobody builds a leaderboard from.

Run:  .venv/Scripts/python -X utf8 players/sergio-busquets/notebooks/03_radar.py
"""

import sys

import matplotlib

matplotlib.use("Agg")
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from mplsoccer import Radar

from football_enigma.data import statsbomb as sb_data
from football_enigma.data.build import player_minutes
from football_enigma.data.statsbomb_ids import BUSQUETS_PLAYER_NAME, LA_LIGA
from football_enigma.utils.paths import PROCESSED_DIR, figures_dir
from football_enigma.viz.theme import COLORS, apply_theme

SUBJ = BUSQUETS_PLAYER_NAME
COMP = LA_LIGA["2015/2016"]
FIGS = figures_dir("sergio-busquets")
MIN_MINUTES = 1500

apply_theme()
FIGS.mkdir(parents=True, exist_ok=True)

# glamour counts straight from raw events
goals, assists, shot_assists = {}, {}, {}
for mid in sb_data.load_matches(COMP)["match_id"]:
    ev = sb_data.load_events(int(mid))
    for p, n in ev[(ev["type"] == "Shot") & (ev["shot_outcome"] == "Goal")][
        "player"].value_counts().items():
        goals[p] = goals.get(p, 0) + int(n)
    if "pass_goal_assist" in ev.columns:
        for p, n in ev[ev["pass_goal_assist"].fillna(False).astype(bool)][
            "player"].value_counts().items():
            assists[p] = assists.get(p, 0) + int(n)
    sa = pd.Series(False, index=ev.index)
    for col in ("pass_shot_assist", "pass_goal_assist"):
        if col in ev.columns:
            sa |= ev[col].fillna(False).astype(bool)
    for p, n in ev[sa]["player"].value_counts().items():
        shot_assists[p] = shot_assists.get(p, 0) + int(n)

minutes = player_minutes(COMP)
agg = pd.read_parquet(PROCESSED_DIR / "laliga1516_player_agg.parquet")
pr = pd.read_parquet(PROCESSED_DIR / "laliga1516_pressure.parquet")
df = agg.merge(pr[["player", "comp_pct_pressured"]], on="player", how="left")
df = df[df["minutes"] >= MIN_MINUTES].copy()
per90 = df["minutes"] / 90
df["goals_p90"] = df["player"].map(lambda p: goals.get(p, 0)).values / per90.values
df["assists_p90"] = df["player"].map(lambda p: assists.get(p, 0)).values / per90.values
df["shot_assists_p90"] = (
    df["player"].map(lambda p: shot_assists.get(p, 0)).values / per90.values
)

# glamour first, then fundamentals — so the radar visibly splits
metrics = {
    "goals_p90": "Goals /90",
    "assists_p90": "Assists /90",
    "shot_assists_p90": "Shots created /90",
    "progressive_p90": "Progressive\npasses /90",
    "passes_p90": "Pass volume /90",
    "comp_pct": "Completion %",
    "comp_pct_pressured": "Completion %\nunder pressure",
    "touches_per_turnover": "Ball security",
}
pct = df.copy()
for c in metrics:
    pct[c] = df[c].rank(pct=True) * 100
b = pct[pct["player"] == SUBJ].iloc[0]
values = [float(b[c]) for c in metrics]
print("Busquets percentiles:",
      {lbl.replace(chr(10), " "): round(v) for lbl, v in zip(metrics.values(), values)})

radar = Radar(params=list(metrics.values()),
              min_range=[0] * len(metrics), max_range=[100] * len(metrics))
fig, ax = radar.setup_axis(figsize=(9, 9), facecolor=COLORS["background"])
radar.draw_circles(ax=ax, facecolor=COLORS["panel"], edgecolor=COLORS["grid"])
radar.draw_radar(values, ax=ax,
                 kwargs_radar={"facecolor": COLORS["subject"], "alpha": 0.55},
                 kwargs_rings={"facecolor": COLORS["subject"], "alpha": 0.12})
radar.draw_range_labels(ax=ax, fontsize=8, color=COLORS["muted"])
radar.draw_param_labels(ax=ax, fontsize=10.5, color=COLORS["text"])
fig.subplots_adjust(top=0.86)
fig.text(0.5, 0.965, "Sergio Busquets — La Liga 2015/16 percentiles",
         ha="center", va="top", color=COLORS["text"], fontsize=15, fontweight="bold")
fig.text(0.5, 0.925,
         "top: what makes a midfielder famous   ·   bottom: what he owns   |   "
         "vs 1500+ min players  |  Data: StatsBomb",
         ha="center", va="top", fontsize=9.5, color=COLORS["muted"])
fig.savefig(FIGS / "03_radar.png")
print("DONE — figure 03 saved")
