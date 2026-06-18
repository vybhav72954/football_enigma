"""IDEA #4 — third-man creation ("the simple connector that unlocks progression").

Progressive-pass and assist stats credit the player who plays the dangerous
ball. They are blind to the man who makes the simple, safe lay-off that LETS
that ball be played — the third-man pattern. This counts exactly that: a
completed pass by the player that is itself NOT progressive (a simple recycle),
whose recipient's very next pass IS the progression (progressive, final-third
entry, or a shot assist). It is the invisible-connector metric, by construction.
Ranked per 90 among central mids; Barca teammates printed as the honest control
(a circulating side manufactures these for everyone, so watch the cluster).

Run:  .venv/Scripts/python -X utf8 players/sergio-busquets/notebooks/_third_man.py
"""

import ast
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from football_enigma.data import statsbomb as sb_data
from football_enigma.data.build import build_positions
from football_enigma.data.statsbomb_ids import BUSQUETS_PLAYER_NAME, LA_LIGA
from football_enigma.utils.paths import PROCESSED_DIR
from football_enigma.viz.charts import _short_name

SUBJ = BUSQUETS_PLAYER_NAME
COMP = LA_LIGA["2015/2016"]
MIN_MIN = 1500
MIN_PASSES = 300
GOAL = (105.0, 34.0)
CENTRAL = {
    "Center Defensive Midfield", "Left Defensive Midfield",
    "Right Defensive Midfield", "Center Midfield",
    "Left Center Midfield", "Right Center Midfield",
}
CACHE = PROCESSED_DIR / "laliga1516_third_man.parquet"


def to_m(v):                       # cache stores location as a string "[x, y]"
    xy = ast.literal_eval(v)
    return xy[0] * 105 / 120, xy[1] * 68 / 80


def is_loc(v):
    return isinstance(v, str) and v.startswith("[")


if CACHE.exists():
    tab = pd.read_parquet(CACHE)
    print(f"loaded cached third-man table ({len(tab)} players)")
else:
    m = sb_data.load_matches(COMP)
    conn, cpass = {}, {}      # connector count / completed passes per player
    for i, mid in enumerate(m["match_id"].astype(int)):
        ev = sb_data.load_events(mid)
        for c in ["pass_outcome", "pass_recipient", "pass_end_location",
                  "location", "pass_shot_assist", "pass_goal_assist"]:
            if c not in ev:
                ev[c] = np.nan
        P = ev[(ev["type"] == "Pass") & ev["pass_outcome"].isna()
               & (ev["team"] == ev["possession_team"]) & ev["player"].notna()
               & ev["location"].apply(is_loc)
               & ev["pass_end_location"].apply(is_loc)
               ].rename(columns={"index": "ev_index"}).copy()
        if not len(P):
            continue
        s = np.array([to_m(v) for v in P["location"]])
        e = np.array([to_m(v) for v in P["pass_end_location"]])
        d0 = np.hypot(GOAL[0] - s[:, 0], GOAL[1] - s[:, 1])
        d1 = np.hypot(GOAL[0] - e[:, 0], GOAL[1] - e[:, 1])
        prog = (d1 <= 0.75 * d0) & ((d0 - d1) >= 5)
        fte = (s[:, 0] < 70) & (e[:, 0] >= 70)            # final-third entry
        assist = (P["pass_shot_assist"] == True) | (P["pass_goal_assist"] == True)  # noqa: E712
        P["unlock"] = prog | fte | assist.to_numpy()
        for p in P["player"].unique():
            cpass[p] = cpass.get(p, 0) + int((P["player"] == p).sum())
        # recipient's next pass per (possession, player)
        groups = {}
        for (poss, plyr), grp in P.groupby(["possession", "player"]):
            grp = grp.sort_values("ev_index")
            groups[(poss, plyr)] = (grp["ev_index"].to_numpy(),
                                    grp["unlock"].to_numpy())
        for row in P.itertuples(index=False):
            if row.unlock:                                # connector = simple pass
                continue
            key = (row.possession, row.pass_recipient)
            g = groups.get(key)
            if g is None:
                continue
            j = np.searchsorted(g[0], row.ev_index, side="right")
            if j < len(g[0]) and g[1][j]:                 # recipient's next pass unlocks
                conn[row.player] = conn.get(row.player, 0) + 1
        if (i + 1) % 50 == 0:
            print(f"  {i + 1} matches...")
    tab = pd.DataFrame({"player": list(cpass)})
    tab["connectors"] = tab["player"].map(conn).fillna(0).astype(int)
    tab["passes"] = tab["player"].map(cpass)
    tab.to_parquet(CACHE)
    print(f"cached third-man table ({len(tab)} players)")

agg = pd.read_parquet(PROCESSED_DIR / "laliga1516_player_agg.parquet")[
    ["player", "minutes"]]
team = pd.read_parquet(PROCESSED_DIR / "laliga1516_buildup.parquet")[
    ["player", "team"]]
pos = build_positions(COMP, "laliga1516")
tab = tab.merge(agg, on="player").merge(team, on="player").merge(pos, on="player")
tab["per90"] = tab["connectors"] / (tab["minutes"] / 90)
tab["share"] = tab["connectors"] / tab["passes"].clip(lower=1) * 100

pool = tab[(tab["minutes"] >= MIN_MIN) & (tab["passes"] >= MIN_PASSES)
           & tab["position"].isin(CENTRAL)].copy()
pool["short"] = [_short_name(p) for p in pool["player"]]


def board(col, label):
    s = pool.sort_values(col, ascending=False).reset_index(drop=True)
    r = int(s.index[s["player"] == SUBJ][0]) + 1
    print(f"\n=== {label}: Busquets rank {r} of {len(s)} ===")
    print(s.head(10)[["short", "team", "connectors", col]]
          .round(2).to_string(index=False))
    return r


print(f"CM pool (1500+ min, {MIN_PASSES}+ passes): {len(pool)}")
r90 = board("per90", "THIRD-MAN CONNECTORS per 90")
rsh = board("share", "THIRD-MAN CONNECTORS as % of completed passes")

barca = pool[pool["team"] == "Barcelona"]
print("\nBarca CMs (connector a Busquets trait or a Barca trait?):")
print(barca[["short", "connectors", "per90", "share"]].round(2).to_string(index=False))

b = pool[pool["player"] == SUBJ].iloc[0]
print(f"\nBusquets: {b['connectors']:.0f} third-man connectors = {b['per90']:.2f} "
      f"per 90 (rank {r90}), {b['share']:.1f}% of his passes (rank {rsh})")
print("DONE - third man")
