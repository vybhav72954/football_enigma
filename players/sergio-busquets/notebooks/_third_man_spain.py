"""OUT-OF-CLUB (#11) — does the third-man signal survive away from Barca?

The strongest skeptic line on the crown is "system artifact": Busquets connects
because peak Barca's positional play manufactures third men for whoever holds the
pivot. The cleanest test is to look at him in a *different* ecosystem — Spain.

Pulls only Spain's own matches from Euro 2020 / WC 2018 / WC 2022 (open data,
attribution per root CLAUDE.md; ~15 matches), builds the same enriched chains,
and asks two things inside the Spain dataset:
  (a) is Busquets Spain's chief third-man connector?
  (b) do Spain receivers unlock MORE right after a Busquets feed than after
      another Spain player's feed (receiver uplift, out of club)?
Samples are small (a dozen matches) — treated as corroboration, not proof; the
honest read is whether the signal weakens to "ecosystem-amplified" or vanishes.

Run:  .venv/Scripts/python -X utf8 players/sergio-busquets/notebooks/_third_man_spain.py
"""

import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from _chains import ATTACK, CENTRAL_MID, match_chains
from football_enigma.data import statsbomb as sb_data
from football_enigma.data.statsbomb_ids import (
    BUSQUETS_PLAYER_NAME, EURO_2020, WORLD_CUP_2018, WORLD_CUP_2022,
)
from football_enigma.utils.paths import PROCESSED_DIR
from football_enigma.viz.charts import _short_name

SUBJ = BUSQUETS_PLAYER_NAME
COMPS = [("Euro 2020", EURO_2020), ("WC 2018", WORLD_CUP_2018), ("WC 2022", WORLD_CUP_2022)]
CHAINS = PROCESSED_DIR / "spain_chains.parquet"


def spain_minutes(evs):
    totals = {}
    for ev in evs:
        end = float(ev["minute"].max())
        starters = set()
        for t in ev.loc[ev["type"] == "Starting XI", "tactics"]:
            tt = json.loads(t) if isinstance(t, str) else t
            if isinstance(tt, dict):
                starters |= {p["player"]["name"] for p in tt.get("lineup", [])}
        on = {p: 0.0 for p in starters}
        off = {}
        for _, s in ev[ev["type"] == "Substitution"].iterrows():
            off[s["player"]] = float(s["minute"])
            rep = s.get("substitution_replacement")
            if isinstance(rep, str):
                on[rep] = float(s["minute"])
        for p, st in on.items():
            totals[p] = totals.get(p, 0.0) + max(off.get(p, end) - st, 0.0)
    return pd.Series(totals, name="minutes")


if CHAINS.exists():
    sp = pd.read_parquet(CHAINS)
    mins = pd.read_parquet(PROCESSED_DIR / "spain_minutes.parquet").set_index("player")["minutes"]
    print(f"loaded cached Spain chains ({len(sp)} passes)")
else:
    evs, frames = [], []
    for label, comp in COMPS:
        sm = sb_data.team_matches(comp, "Spain")
        print(f"{label}: {len(sm)} Spain matches")
        for mid in sm["match_id"].astype(int):
            evs.append(sb_data.load_events(mid))
    # one modal-position map across all Spain matches (avoids comp-wide pulls)
    posrows = pd.concat([e[["player", "position"]].dropna() for e in evs])
    pos_map = (posrows.groupby("player")["position"]
               .agg(lambda s: s.mode(dropna=True).iloc[0]).to_dict())
    mid_iter = []
    for label, comp in COMPS:
        for mid in sb_data.team_matches(comp, "Spain")["match_id"].astype(int):
            mid_iter.append(mid)
    for mid, ev in zip(mid_iter, evs):
        frames.append(match_chains(ev, mid, pos_map))
    sp = pd.concat(frames, ignore_index=True)
    sp.to_parquet(CHAINS)
    mins = spain_minutes(evs)
    mins.rename_axis("player").reset_index().to_parquet(PROCESSED_DIR / "spain_minutes.parquet")
    print(f"cached Spain chains ({len(sp)} passes)")

mins_d = dict(mins)
conn = (sp["u_primary"] == 0) & (sp["n_primary"] == 1)
g = sp.groupby("player").agg(passes=("u_primary", "size"), team=("team", "first")).reset_index()
g["conn"] = g["player"].map(sp[conn].groupby("player").size()).fillna(0).astype(int)
g["minutes"] = g["player"].map(mins_d)
g = g.dropna(subset=["minutes"])
g["per90"] = g["conn"] / (g["minutes"] / 90)
g["share"] = g["conn"] / g["passes"].clip(lower=1) * 100

# (a) is he Spain's chief connector? rank among everyone in Spain's matches with
#     enough involvement (small sample -> low thresholds, flagged loudly)
pool = g[(g["minutes"] >= 270) & (g["passes"] >= 80)].sort_values("per90", ascending=False).reset_index(drop=True)
spain = pool[pool["team"] == "Spain"].reset_index(drop=True)
rank_all = int(pool.index[pool["player"] == SUBJ][0]) + 1 if SUBJ in set(pool["player"]) else -1
rank_es = int(spain.index[spain["player"] == SUBJ][0]) + 1 if SUBJ in set(spain["player"]) else -1
b = g[g["player"] == SUBJ].iloc[0]
print(f"\n=== (a) THIRD-MAN CONNECTORS, Spain matches (n={sp['match_id'].nunique()} matches, SMALL SAMPLE) ===")
print(f"Busquets: {b['conn']:.0f} connectors, {b['per90']:.2f}/90, {b['share']:.1f}% of passes")
print(f"rank {rank_all} of {len(pool)} (all teams in Spain's games); rank {rank_es} of {len(spain)} among Spain players")
print("\nSpain players (third-man per 90):")
print(spain.assign(short=[_short_name(p) for p in spain["player"]])
      .head(8)[["short", "conn", "per90", "share"]].round(2).to_string(index=False))

# (b) receiver uplift, out of club
es = sp[(sp["u_primary"] == 0) & (sp["team"] == "Spain")].copy()
es["from_busq"] = es["player"] == SUBJ
busq, other = es[es["from_busq"]], es[~es["from_busq"]]
print(f"\n=== (b) RECEIVER UPLIFT (Spain) ===")
print(f"recipient next-pass unlock: after BUSQUETS feed {busq['n_primary'].mean() * 100:.1f}% "
      f"(n={len(busq)})  vs after other-Spain feed {other['n_primary'].mean() * 100:.1f}% (n={len(other)})")
rows = []
for rcpt, gg in es.groupby("recipient"):
    gb, go = gg[gg["from_busq"]], gg[~gg["from_busq"]]
    if len(gb) >= 15 and len(go) >= 15:
        rows.append((_short_name(rcpt), len(gb), gb["n_primary"].mean() * 100,
                     go["n_primary"].mean() * 100,
                     (gb["n_primary"].mean() - go["n_primary"].mean()) * 100))
if rows:
    upl = pd.DataFrame(rows, columns=["receiver", "n_busq", "after_busq", "after_other", "uplift"])
    print("per Spain receiver (>=15 feeds each side):")
    print(upl.sort_values("uplift", ascending=False).round(1).to_string(index=False))
else:
    print("(no Spain receiver clears the >=15-feeds-each-side bar — sample too thin per pair)")
print("\nDONE - Spain out-of-club")
