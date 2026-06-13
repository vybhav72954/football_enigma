"""Bulk-scrape Bayern's 2013/14 Bundesliga matches from WhoScored (resumable).

The Tier-2 "Act I" data: Kroos's last Bayern league season, scraped with the
same pipeline as WC 2014 so the metrics line up. One Chrome session, one
match at a time, parquet written immediately, failures logged and skipped —
re-running resumes. The Chrome process tree is killed on exit (uc-mode
chromedriver otherwise leaks a child that holds the stdout pipe; see
project CLAUDE.md §3).

Prove one match first, then run the batch:
  .venv/Scripts/python -X utf8 -u -m football_enigma.data.scrape_bayern1314 --limit 1
  .venv/Scripts/python -X utf8 -u -m football_enigma.data.scrape_bayern1314
"""

import argparse
import os
import sys
import time

import pandas as pd

from football_enigma.data.whoscored import WhoScored, load_schedule
from football_enigma.utils.paths import RAW_DIR

LEAGUE = "GER-Bundesliga"
SEASON = "1314"  # 2013/14; note soccerdata maps "2014" to 1415, not this season
TEAM_MATCH = "Bayern"  # substring match against home/away team names
EVENTS_DIR = RAW_DIR / "whoscored" / "events"


def bayern_schedule() -> pd.DataFrame:
    schedule = load_schedule(LEAGUE, SEASON)
    teams = schedule[["home_team", "away_team"]].astype(str)
    mask = teams.apply(
        lambda r: TEAM_MATCH in r["home_team"] or TEAM_MATCH in r["away_team"],
        axis=1,
    )
    return schedule[mask].reset_index(drop=True)


def _shutdown(ws: WhoScored) -> None:
    """Close the driver and reap any orphaned Chrome/chromedriver children."""
    try:
        ws._driver.quit()
    except Exception:  # noqa: BLE001 - best-effort teardown
        pass
    try:
        import psutil

        me = psutil.Process(os.getpid())
        for child in me.children(recursive=True):
            if "chrome" in child.name().lower():
                child.kill()
    except Exception:  # noqa: BLE001 - best-effort teardown
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None,
        help="scrape at most N matches (use 1 to prove the pipeline)",
    )
    args = parser.parse_args()

    schedule = bayern_schedule()
    todo = [
        int(gid)
        for gid in schedule["game_id"]
        if not (EVENTS_DIR / f"{gid}.parquet").exists()
    ]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(
        f"{len(schedule)} Bayern matches in {LEAGUE} {SEASON}; "
        f"{len(todo)} to scrape this run",
        flush=True,
    )
    if not todo:
        return

    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    ws = WhoScored(leagues=LEAGUE, seasons=SEASON)
    failures = []
    try:
        for i, gid in enumerate(todo, 1):
            try:
                events = ws.read_events(match_id=gid).reset_index()
                events.to_parquet(EVENTS_DIR / f"{gid}.parquet")
                print(
                    f"[{i}/{len(todo)}] {gid} ok ({len(events)} events)",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - skip and continue
                failures.append(gid)
                print(f"[{i}/{len(todo)}] {gid} FAILED: {exc}", flush=True)
                time.sleep(10)
    finally:
        _shutdown(ws)
    print(f"DONE, failures: {failures}", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
