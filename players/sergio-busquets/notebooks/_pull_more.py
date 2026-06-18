"""Second cache warmer — the data that makes the invariance/absence thesis
obvious. Cache-first / resumable. Run after _pull_career.py.

  1. Early La Liga events 2008/09-2013/14 (Barca slate) -> extends the
     "constant" curve back to Busquets's 2008 debut (13 seasons total).
  2. La Liga 2020/21 360 frames (Barca matches) -> a full-season "bodies
     within 3m" comparison set for the swarmed-least map.
  3. World Cup 2022 events + 360 (Spain) -> his captain swan-song, with 360.

Run:  .venv/Scripts/python -X utf8 players/sergio-busquets/notebooks/_pull_more.py
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from football_enigma.data import statsbomb as sb_data
from football_enigma.data.statsbomb_ids import LA_LIGA, WORLD_CUP_2022

EARLY = ["2008/2009", "2009/2010", "2010/2011",
         "2011/2012", "2012/2013", "2013/2014"]


def barca_ids(comp):
    m = sb_data.load_matches(comp)
    b = m[(m["home_team"] == "Barcelona") | (m["away_team"] == "Barcelona")]
    return b["match_id"].astype(int).tolist()


# 1. early La Liga events
print(">>> early La Liga events", flush=True)
for s in EARLY:
    ids = barca_ids(LA_LIGA[s])
    for i, mid in enumerate(ids, 1):
        sb_data.load_events(mid)
        if i % 5 == 0 or i == len(ids):
            print(f"  {s}: {i}/{len(ids)}", flush=True)

# 2. La Liga 2020/21 360 frames (events already cached)
print(">>> La Liga 2020/21 360 frames", flush=True)
ids = barca_ids(LA_LIGA["2020/2021"])
for i, mid in enumerate(ids, 1):
    sb_data.load_frames(mid)
    if i % 5 == 0 or i == len(ids):
        print(f"  20/21 360: {i}/{len(ids)}", flush=True)

# 3. World Cup 2022 — Spain events + 360
print(">>> World Cup 2022 (Spain) events + 360", flush=True)
m = sb_data.load_matches(WORLD_CUP_2022)
spain = m[(m["home_team"] == "Spain") | (m["away_team"] == "Spain")]
for mid in spain["match_id"].astype(int):
    sb_data.load_events(mid)
    sb_data.load_frames(mid)
    print(f"  WC22 {mid} done", flush=True)

print("DONE", flush=True)
