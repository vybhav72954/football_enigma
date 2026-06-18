"""SEASON REPLICATION (#10) — is the third-man crown a one-season fluke?

Runs the headline third-man connector metric (primary definition) across every
cached Barca La Liga season 2008/09-2020/21. Each non-2015/16 open-data season is
essentially Barca's match slate (~31-38 games), so the pool is Barca plus the
opponents they met; the honest persistence claim is therefore his rank among
BARCA central mids year after year, alongside the stability of his per-90 level.

A connector = a completed simple (non-progressive) pass whose recipient's very
next pass is the unlock (progressive / final-third entry / shot assist).

Run:  .venv/Scripts/python -X utf8 players/sergio-busquets/notebooks/_third_man_replication.py
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from _chains import CENTRAL_MID, build_chains
from football_enigma.data.build import build_positions, player_minutes
from football_enigma.data.statsbomb_ids import BUSQUETS_PLAYER_NAME, LA_LIGA
from football_enigma.utils.paths import PROCESSED_DIR

SUBJ = BUSQUETS_PLAYER_NAME
MIN_MIN, MIN_PASSES = 900, 150
CACHE = PROCESSED_DIR / "busquets_thirdman_career.parquet"


def season_row(season: str):
    comp = LA_LIGA[season]
    name = "laliga" + season[2:4] + season[7:9]  # 2015/2016 -> laliga1516
    pc = build_chains(comp, name)
    if SUBJ not in set(pc["player"]):
        return None
    pos = build_positions(comp, name)
    mins = player_minutes(comp).rename_axis("player").reset_index(name="minutes")
    conn = (pc["u_primary"] == 0) & (pc["n_primary"] == 1)
    g = pc.groupby("player").agg(passes=("u_primary", "size")).reset_index()
    g["conn"] = g["player"].map(pc[conn].groupby("player").size()).fillna(0).astype(int)
    g = g.merge(pos, on="player").merge(mins, on="player")
    g["per90"] = g["conn"] / (g["minutes"] / 90)
    pool = g[(g["minutes"] >= MIN_MIN) & (g["passes"] >= MIN_PASSES)
             & g["position"].isin(CENTRAL_MID)].copy()
    pool = pool.sort_values("per90", ascending=False).reset_index(drop=True)
    if SUBJ not in set(pool["player"]):
        return None
    b = pool[pool["player"] == SUBJ].iloc[0]
    rank = int(pool.index[pool["player"] == SUBJ][0]) + 1
    # Barca-only rank: restrict the pool to players who appear most for Barca.
    # team isn't on the chains table; approximate "Barca CM" as pool CMs sharing
    # >=1 possession-team match with Busquets is overkill — instead rank Busquets
    # among the pool and report the leader, plus his per-90 level (the stable
    # quantity). Pool composition note kept in the writeup.
    return {"season": season, "busq_per90": round(b["per90"], 2),
            "busq_conn": int(b["conn"]), "rank": rank, "pool": len(pool),
            "leader_per90": round(pool.iloc[0]["per90"], 2)}


if CACHE.exists():
    res = pd.read_parquet(CACHE)
    print(f"loaded cached career table ({len(res)} seasons)")
else:
    rows = []
    for season in LA_LIGA:
        print(f"--- {season}")
        r = season_row(season)
        if r:
            rows.append(r)
    res = pd.DataFrame(rows)
    res.to_parquet(CACHE)

print("\n=== THIRD-MAN CONNECTORS per 90 across Busquets's club career ===")
print(res.to_string(index=False))
print(f"\nseasons ranked #1 of pool: {int((res['rank'] == 1).sum())}/{len(res)}; "
      f"top-3: {int((res['rank'] <= 3).sum())}/{len(res)}")
print(f"per-90 range {res['busq_per90'].min():.2f}-{res['busq_per90'].max():.2f}, "
      f"median {res['busq_per90'].median():.2f}")
print("DONE - season replication")
