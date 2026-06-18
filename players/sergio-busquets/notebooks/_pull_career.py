"""One-off cache warmer: fetch StatsBomb events for Busquets's Barca La Liga
seasons (2014/15-2020/21, ex-2015/16 already cached) so the career-trajectory
figures can be built offline. Cache-first, so safe to re-run / resume.

Run:  .venv/Scripts/python -X utf8 players/sergio-busquets/notebooks/_pull_career.py
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from football_enigma.data import statsbomb as sb_data
from football_enigma.data.statsbomb_ids import LA_LIGA

SEASONS = ["2014/2015", "2016/2017", "2017/2018",
           "2018/2019", "2019/2020", "2020/2021"]

done = 0
for s in SEASONS:
    m = sb_data.load_matches(LA_LIGA[s])
    ids = m["match_id"].astype(int).tolist()
    for i, mid in enumerate(ids, 1):
        sb_data.load_events(mid)  # caches to disk
        done += 1
        if i % 5 == 0 or i == len(ids):
            print(f"{s}: {i}/{len(ids)}  (total {done})", flush=True)
print(f"DONE — {done} matches cached")
