"""Bulk-scrape all 2014 World Cup matches from WhoScored (resumable).

One Chrome session, one match at a time, parquet written immediately per
match, failures logged and skipped — re-running picks up where it left off.

Run:  .venv/Scripts/python -X utf8 -u -m football_enigma.data.scrape_wc2014
"""

import sys
import time

import pandas as pd

from football_enigma.data.whoscored import WhoScored
from football_enigma.utils.paths import RAW_DIR

EVENTS_DIR = RAW_DIR / "whoscored" / "events"


def main() -> None:
    schedule = pd.read_parquet(
        RAW_DIR / "whoscored" / "schedule_INT-World_Cup_2014.parquet"
    )
    todo = [
        int(gid)
        for gid in schedule["game_id"]
        if not (EVENTS_DIR / f"{gid}.parquet").exists()
    ]
    print(f"{len(todo)} of {len(schedule)} matches to scrape", flush=True)
    if not todo:
        return

    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    ws = WhoScored(leagues="INT-World Cup", seasons="2014")
    failures = []
    for i, gid in enumerate(todo, 1):
        try:
            events = ws.read_events(match_id=gid).reset_index()
            events.to_parquet(EVENTS_DIR / f"{gid}.parquet")
            print(f"[{i}/{len(todo)}] {gid} ok ({len(events)} events)", flush=True)
        except Exception as exc:  # noqa: BLE001 - skip and continue the run
            failures.append(gid)
            print(f"[{i}/{len(todo)}] {gid} FAILED: {exc}", flush=True)
            time.sleep(10)
    print(f"DONE, failures: {failures}", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
