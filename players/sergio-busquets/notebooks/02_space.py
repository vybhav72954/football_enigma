"""Fig 02 — playing where the pressure isn't.

From StatsBomb 360 freeze-frames (La Liga 2020/21 + Euro 2020 + WC 2022), for
every framed pass we count opponents within 3 m of the passer. Forwards live
crowded and complete less; the deep pivots live in space and complete more.
Busquets sits at the extreme of that pivot corner: among 31 central
midfielders only ~0.46 opponents within 3 m and 4% of passes swarmed (2+
within 3 m), at 92% completion on by far the largest sample. The honest
caveat: low crowding is the deep-pivot *archetype* (Pjanic, Kroos, de Jong
share it) — Busquets is its platonic extreme, not a freak. And the box score
has no column for the defender who was never close enough to matter.

Run:  .venv/Scripts/python -X utf8 players/sergio-busquets/notebooks/02_space.py
"""

import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from football_enigma.data import statsbomb as sb_data
from football_enigma.data.statsbomb import decode_locations
from football_enigma.data.statsbomb_ids import (
    BUSQUETS_PLAYER_NAME, EURO_2020, LA_LIGA, WORLD_CUP_2022,
)
from football_enigma.metrics.packing import match_crowding
from football_enigma.utils.paths import PROCESSED_DIR, figures_dir
from football_enigma.viz.charts import crowd_scatter

SUBJ = BUSQUETS_PLAYER_NAME
FIGS = figures_dir("sergio-busquets")
MIN_FRAMED = 60

SETS = [(LA_LIGA["2020/2021"], "Barcelona"),
        (EURO_2020, "Spain"), (WORLD_CUP_2022, "Spain")]

agg = defaultdict(lambda: {"near": 0.0, "n": 0, "tight": 0, "comp": 0,
                           "pos": defaultdict(int)})
for comp, team in SETS:
    m = sb_data.load_matches(comp)
    m = m[(m["home_team"] == team) | (m["away_team"] == team)]
    for mid in m["match_id"].astype(int):
        ev = decode_locations(sb_data.load_events(mid), ("location",))
        cr = match_crowding(ev, sb_data.load_frames(mid), radius_m=3.0)
        if cr.empty:
            continue
        for p, ps in ev.dropna(subset=["player", "position"])[
            ["player", "position"]
        ].itertuples(index=False):
            agg[p]["pos"][ps] += 1
        for _, r in cr.iterrows():
            a = agg[r["player"]]
            a["near"] += r["near"]; a["n"] += 1
            a["tight"] += r["near"] >= 2; a["comp"] += bool(r["completed"])

rows = []
for p, a in agg.items():
    if a["n"] < MIN_FRAMED:
        continue
    pos = max(a["pos"], key=a["pos"].get) if a["pos"] else ""
    rows.append({"player": p, "framed": a["n"], "avg_near": a["near"] / a["n"],
                 "tight_share": a["tight"] / a["n"] * 100,
                 "comp": a["comp"] / a["n"] * 100, "pos": pos})
df = pd.DataFrame(rows)
df = df[~df["pos"].str.contains("Goalkeeper", na=False)]  # keepers trivially uncrowded
df.to_parquet(PROCESSED_DIR / "busquets_space.parquet")

fig, ax = crowd_scatter(
    df,
    x="avg_near",
    y="comp",
    subject=SUBJ,
    title="Playing where the pressure isn't",
    subtitle="outfield players, 60+ framed passes (La Liga 20/21, Euro 2020, "
             "WC 2022)  |  crowding vs completion",
    xlabel="Avg opponents within 3 m when passing  (left = more space)",
    ylabel="Pass completion %",
    annotate=["Frenkie de Jong", "Miralem Pjanić", "Luka Modrić"],
    source="StatsBomb 360",
)
fig.savefig(FIGS / "02_space.png")

b = df[df["player"] == SUBJ].iloc[0]
mids = df[df["pos"].str.contains("Midfield", na=False)]
print(f"Outfield players: {len(df)} | central-ish mids: {len(mids)}")
print(f"Busquets: framed {b['framed']:.0f} | avg within 3m {b['avg_near']:.2f} "
      f"| swarmed(2+) {b['tight_share']:.0f}% | comp {b['comp']:.0f}%")
print(f"  least-crowded rank among midfielders: "
      f"{(mids['avg_near'] < b['avg_near']).sum()+1}/{len(mids)}")
print("DONE — figure 02 saved")
