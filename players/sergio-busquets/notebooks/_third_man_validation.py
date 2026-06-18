"""VALIDATION — is the third-man #1 a trick of definition, or real?

Builds ONE pass-level table for La Liga 2015/16 (every completed pass: location,
direction, pressure, zone, whether it is a simple recycle, and whether the
recipient's next pass unlocks the game), then answers the five skeptic attacks
the user prioritised, in order:

  A. THIRD-MAN OVER EXPECTED  - model P(simple pass becomes a third-man unlock)
     from geometry+pressure+zone, score actual-minus-expected. Kills "it's just
     volume / role": volume cancels in a rate-over-expected.
  B. PRESSURED & CENTRAL third-man - the Busquets-specific context (pressed,
     central third). The bait-and-release essence.
  C. BACKWARD/SIDEWAYS -> FORWARD - of his non-advancing passes, how often the
     team goes forward on the next pass. "Safe passes, loaded not dead."
  D. RECEIVER UPLIFT - do teammates unlock MORE right after a Busquets feed than
     after anyone else's, controlling for zone via the expected model. "He makes
     others better."

Run:  .venv/Scripts/python -X utf8 players/sergio-busquets/notebooks/_third_man_validation.py
"""

import ast
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from football_enigma.data import statsbomb as sb_data
from football_enigma.data.build import build_positions
from football_enigma.data.statsbomb_ids import BUSQUETS_PLAYER_NAME, LA_LIGA
from football_enigma.utils.paths import PROCESSED_DIR
from football_enigma.viz.charts import _short_name

SUBJ = BUSQUETS_PLAYER_NAME
COMP = LA_LIGA["2015/2016"]
MIN_MIN = 1500
GOAL = (105.0, 34.0)
CENTRAL = {
    "Center Defensive Midfield", "Left Defensive Midfield",
    "Right Defensive Midfield", "Center Midfield",
    "Left Center Midfield", "Right Center Midfield",
}
CACHE = PROCESSED_DIR / "laliga1516_pass_chains.parquet"


def to_m(v):
    xy = ast.literal_eval(v)
    return xy[0] * 105 / 120, xy[1] * 68 / 80


def is_loc(v):
    return isinstance(v, str) and v.startswith("[")


if CACHE.exists():
    pc = pd.read_parquet(CACHE)
    print(f"loaded cached pass chains ({len(pc)} rows)")
else:
    m = sb_data.load_matches(COMP)
    frames = []
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
        fte = (s[:, 0] < 70) & (e[:, 0] >= 70)
        assist = ((P["pass_shot_assist"] == True)                       # noqa: E712
                  | (P["pass_goal_assist"] == True)).to_numpy()          # noqa: E712
        P["x"], P["y"] = s[:, 0], s[:, 1]
        P["dx"] = e[:, 0] - s[:, 0]
        P["length"] = np.hypot(e[:, 0] - s[:, 0], e[:, 1] - s[:, 1])
        P["pressed"] = (P["under_pressure"] == True).astype(int)         # noqa: E712
        P["central"] = (np.abs(s[:, 1] - 34) <= 11.33).astype(int)
        P["own"] = (s[:, 0] < 35).astype(int)
        P["final"] = (s[:, 0] >= 70).astype(int)
        P["adv"] = (P["dx"] > 2).astype(int)                             # advanced > 2m
        P["unlock_self"] = (prog | fte | assist).astype(int)
        P["simple"] = 1 - P["unlock_self"]
        P["recipient"] = P["pass_recipient"]
        # recipient's immediate next pass, and the team's next-2 passes
        rec_seq, poss_seq = {}, {}
        for (poss, plyr), g in P.groupby(["possession", "player"]):
            g = g.sort_values("ev_index")
            rec_seq[(poss, plyr)] = (g["ev_index"].to_numpy(),
                                     g["unlock_self"].to_numpy())
        for poss, g in P.groupby("possession"):
            g = g.sort_values("ev_index")
            poss_seq[poss] = (g["ev_index"].to_numpy(),
                              g["unlock_self"].to_numpy(), g["adv"].to_numpy())
        tm, tm2, fwd = [], [], []
        for row in P.itertuples(index=False):
            rk = rec_seq.get((row.possession, row.recipient))
            if rk is None:
                tm.append(0)
            else:
                j = np.searchsorted(rk[0], row.ev_index, side="right")
                tm.append(int(rk[1][j]) if j < len(rk[0]) else 0)
            pk = poss_seq[row.possession]
            j2 = np.searchsorted(pk[0], row.ev_index, side="right")
            tm2.append(int(pk[1][j2:j2 + 2].any()) if j2 < len(pk[0]) else 0)
            fwd.append(int(pk[2][j2:j2 + 2].any()) if j2 < len(pk[0]) else 0)
        P["tm"], P["tm2"], P["fwd2"] = tm, tm2, fwd
        frames.append(P[["player", "recipient", "team", "x", "y", "dx",
                         "length", "pressed", "central", "own", "final", "adv",
                          "unlock_self", "simple", "tm", "tm2", "fwd2"]])
        if (i + 1) % 50 == 0:
            print(f"  {i + 1} matches...")
    pc = pd.concat(frames, ignore_index=True)
    pc.to_parquet(CACHE)
    print(f"cached pass chains ({len(pc)} rows)")

# ---- pool metadata ----
agg = pd.read_parquet(PROCESSED_DIR / "laliga1516_player_agg.parquet")[
    ["player", "minutes"]]
team = pd.read_parquet(PROCESSED_DIR / "laliga1516_buildup.parquet")[["player", "team"]]
pos = build_positions(COMP, "laliga1516")
meta = agg.merge(team, on="player").merge(pos, on="player")
meta = meta[(meta["minutes"] >= MIN_MIN) & meta["position"].isin(CENTRAL)]
CMS = set(meta["player"])
mins = dict(zip(meta["player"], meta["minutes"]))
tm_team = dict(zip(meta["player"], meta["team"]))


def board(series_by_player, label, mincount_by_player=None, minn=1, n=10):
    s = series_by_player.dropna()
    s = s[[p in CMS for p in s.index]]
    if mincount_by_player is not None:
        s = s[[mincount_by_player.get(p, 0) >= minn for p in s.index]]
    s = s.sort_values(ascending=False)
    rank = list(s.index).index(SUBJ) + 1 if SUBJ in s.index else -1
    df = pd.DataFrame({"short": [_short_name(p) for p in s.index],
                       "team": [tm_team.get(p, "?") for p in s.index],
                       label: s.values})
    print(f"\n=== {label}: Busquets rank {rank} of {len(s)} ===")
    print(df.head(n).round(2).to_string(index=False))
    return rank


simple = pc[(pc["simple"] == 1) & pc["player"].isin(CMS)].copy()
print(f"\nBusquets simple passes: {int((simple['player'] == SUBJ).sum())}, "
      f"of which third-man unlocks: {int(simple.loc[simple['player'] == SUBJ, 'tm'].sum())}")

# ---- A. THIRD-MAN OVER EXPECTED ----
allsimple = pc[pc["simple"] == 1].copy()
F = ["x", "y", "dx", "length", "pressed", "central", "own", "final"]
Xs = StandardScaler().fit_transform(allsimple[F].values)
allsimple["xtm"] = LogisticRegression(max_iter=1000).fit(
    Xs, allsimple["tm"].values).predict_proba(Xs)[:, 1]
allsimple["over"] = allsimple["tm"] - allsimple["xtm"]
cnt = allsimple.groupby("player").size()
over = allsimple.groupby("player")["over"].mean() * 100
rA = board(over, "THIRD-MAN OVER EXPECTED (pp above model)", cnt, minn=200)
barca_over = over[[p for p in over.index if tm_team.get(p) == "Barcelona" and p in CMS]]
print("Barca CMs (over-expected):",
      ", ".join(f"{_short_name(p)} {over[p]:+.2f}" for p in barca_over.sort_values(ascending=False).index))

# ---- B. PRESSURED & CENTRAL third-man ----
ps = simple[simple["pressed"] == 1]
rB1 = board(ps.groupby("player")["tm"].mean() * 100,
            "THIRD-MAN under PRESSURE (% of pressed simple passes)",
            ps.groupby("player").size(), minn=30)
pc_ = simple[(simple["pressed"] == 1) & (simple["central"] == 1)]
rB2 = board(pc_.groupby("player")["tm"].mean() * 100,
            "THIRD-MAN pressed & CENTRAL third (the bait-and-release move)",
            pc_.groupby("player").size(), minn=12)

# ---- C. BACKWARD/SIDEWAYS -> FORWARD ----
bl = pc[(pc["adv"] == 0) & pc["player"].isin(CMS)]      # non-advancing passes
rC = board(bl.groupby("player")["fwd2"].mean() * 100,
           "BACKWARD/LATERAL -> team FORWARD within 2 passes (%)",
           bl.groupby("player").size(), minn=150)
barca_bl = bl[bl["team"] == "Barcelona"].groupby("player")["fwd2"].mean() * 100
print("Barca CMs (back/lat -> forward %):",
      ", ".join(f"{_short_name(p)} {barca_bl[p]:.1f}" for p in barca_bl.sort_values(ascending=False).index))

# ---- D. RECEIVER UPLIFT (Barca) ----
bsimple = allsimple[allsimple["team"] == "Barcelona"].copy()
bsimple["from_busq"] = bsimple["player"] == SUBJ
busq = bsimple[bsimple["from_busq"]]
other = bsimple[~bsimple["from_busq"]]
print("\n=== RECEIVER UPLIFT (Barca, simple passes) ===")
print(f"recipient next-pass UNLOCK rate:  after BUSQUETS feed "
      f"{busq['tm'].mean() * 100:.1f}%  vs after other-Barca feed "
      f"{other['tm'].mean() * 100:.1f}%")
print(f"zone-controlled (actual - expected): BUSQUETS {busq['over'].mean() * 100 if 'over' in busq else float('nan'):+.2f}pp  "
      f"others {other['over'].mean() * 100:+.2f}pp")
print("per named receiver — next-pass unlock rate after a Busquets feed vs others "
      "(n = feeds from Busquets):")
rows = []
for rcpt, g in bsimple.groupby("recipient"):
    gb = g[g["from_busq"]]
    go = g[~g["from_busq"]]
    if len(gb) >= 40 and len(go) >= 40:
        rows.append((_short_name(rcpt), len(gb), gb["tm"].mean() * 100,
                     go["tm"].mean() * 100, (gb["tm"].mean() - go["tm"].mean()) * 100))
upl = pd.DataFrame(rows, columns=["receiver", "n_busq", "after_busq", "after_other", "uplift"])
print(upl.sort_values("uplift", ascending=False).round(1).to_string(index=False))

print("\nDONE - third-man validation")
