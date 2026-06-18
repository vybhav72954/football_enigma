"""Build the three new-axis processed tables for La Liga 2015/16 and report
where Busquets actually lands (grounds the figure framing — run before the
06/07/08 notebooks). Persists *_defending.parquet and *_buildup.parquet.

Run:  .venv/Scripts/python -X utf8 players/sergio-busquets/notebooks/_build_axes.py
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from football_enigma.data.build import (
    build_backline_receptions,
    build_defensive_profile,
    player_minutes,
)
from football_enigma.data.statsbomb_ids import BUSQUETS_PLAYER_NAME, LA_LIGA
from football_enigma.metrics.footprint import action_dispersion
from football_enigma.utils.paths import PROCESSED_DIR

SUBJ = BUSQUETS_PLAYER_NAME
COMP = LA_LIGA["2015/2016"]
MIN_MIN = 1500

minutes = player_minutes(COMP).rename("minutes")
per90 = minutes / 90


def rank_report(label, df, col, ascending=False, n=8):
    d = df.dropna(subset=[col]).copy()
    d = d.sort_values(col, ascending=ascending).reset_index(drop=True)
    pos = d.index[d["player"] == SUBJ]
    r = int(pos[0]) + 1 if len(pos) else -1
    print(f"\n=== {label}  (n={len(d)})  Busquets rank {r} ===")
    print(d.head(n)[["player", col]].round(2).to_string(index=False))
    if r > n:
        print("  ...")
        print(d[d["player"] == SUBJ][["player", col]].round(2).to_string(index=False))


# ---- B: screening, not tackling ----
dfn = build_defensive_profile(COMP, "laliga1516").set_index("player")
dfn = dfn.join(minutes, how="inner")
dfn = dfn[dfn["minutes"] >= MIN_MIN].copy()
dfn["reads"] = dfn["interceptions"] + dfn["recoveries"]
dfn["commits"] = dfn["tackles"] + dfn["fouls"] + dfn["dribbled_past"]
dfn["engagements"] = dfn["reads"] + dfn["commits"]
dfn["read_share"] = dfn["reads"] / dfn["engagements"].clip(lower=1) * 100
dfn["engagements_p90"] = dfn["engagements"] / (dfn["minutes"] / 90)
dfn = dfn.reset_index()
rank_report("READ SHARE % (reads / engagements)", dfn, "read_share")
b = dfn[dfn["player"] == SUBJ].iloc[0]
print(f"  Busquets: reads {b['reads']:.0f}  commits {b['commits']:.0f}  "
      f"read_share {b['read_share']:.0f}%  engagements/90 {b['engagements_p90']:.1f}")

# ---- A: the smallest map ----
actions = pd.read_parquet(PROCESSED_DIR / "laliga1516_actions.parquet")
disp = action_dispersion(actions, min_actions=200)
disp = disp.set_index("player").join(minutes, how="inner")
disp = disp[disp["minutes"] >= MIN_MIN].reset_index()
disp["actions_p90"] = disp["actions"] / (disp["minutes"] / 90)
rank_report("SPREAD m (smallest = most concentrated)", disp, "spread", ascending=True)
b = disp[disp["player"] == SUBJ].iloc[0]
print(f"  Busquets: spread {b['spread']:.1f} m  std_x {b['std_x']:.1f}  "
      f"std_y {b['std_y']:.1f}  actions/90 {b['actions_p90']:.0f}")
# among midfielder-ish only: spread between 18 and 30 excludes CBs(<18) & wide(>30)? report neighbours
print("  five most-concentrated outfielders:")
print(disp.sort_values("spread").head(5)[["player", "spread", "actions_p90"]].round(1).to_string(index=False))

# ---- C: the first receiver ----
bu = build_backline_receptions(COMP, "laliga1516").set_index("player")
bu = bu.join(minutes, how="inner")
bu = bu[bu["minutes"] >= MIN_MIN].copy()
bu["from_backline_p90"] = bu["from_backline"] / (bu["minutes"] / 90)
bu["backline_share"] = bu["from_backline"] / bu["total_received"].clip(lower=1) * 100
bu = bu.reset_index()
rank_report("RECEPTIONS FROM OWN BACKLINE /90", bu, "from_backline_p90")
rank_report("BACKLINE SHARE OF RECEPTIONS %", bu, "backline_share")
b = bu[bu["player"] == SUBJ].iloc[0]
print(f"  Busquets: from_backline {b['from_backline']:.0f}  total_received "
      f"{b['total_received']:.0f}  /90 {b['from_backline_p90']:.1f}  "
      f"share {b['backline_share']:.0f}%")

print("\nDONE — axes built")
