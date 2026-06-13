"""WhoScored (Opta) event loader built on soccerdata, with two additions:

1. PatchedWhoScored fixes a Chrome quirk where raw JSON responses are
   wrapped in an HTML shell (``<html><head></head><body>{...}</body></html>``),
   which breaks soccerdata's ``json.load`` on cached files.
2. Loader helpers cache cleaned outputs to the repo's parquet store so the
   slow Selenium scrape never runs twice for the same data.

Heads-up: instantiating the reader launches a real Chrome window (Incapsula
anti-bot); scraping is slow by design. Be polite — never bulk-scrape what is
already cached.
"""

import re

import pandas as pd
import soccerdata as sd

from football_enigma.utils.paths import RAW_DIR

_WS_DIR = RAW_DIR / "whoscored"

_JSON_SHELL = re.compile(
    r"^<html><head></head><body>(?:<pre[^>]*>)?(.*?)(?:</pre>)?</body></html>$",
    re.DOTALL,
)


# Must be named exactly "WhoScored": soccerdata resolves a reader's valid
# leagues by cls.__name__ against its league_dict registry.
class WhoScored(sd.WhoScored):
    def _validate_page(self, url: str) -> str:
        page = super()._validate_page(url)
        match = _JSON_SHELL.match(page.strip())
        if match and match.group(1).lstrip()[:1] in "{[":
            return match.group(1)
        return page


def _reader(league: str, season: str | int, **kwargs) -> WhoScored:
    return WhoScored(leagues=league, seasons=season, **kwargs)


def load_schedule(league: str, season: str | int) -> pd.DataFrame:
    cache = _WS_DIR / f"schedule_{league.replace(' ', '_')}_{season}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    schedule = _reader(league, season).read_schedule().reset_index()
    cache.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_parquet(cache)
    return schedule


def load_events(league: str, season: str | int, match_ids: list[int]) -> pd.DataFrame:
    """Raw Opta match-centre events for the given WhoScored game ids."""
    cache = _WS_DIR / "events"
    cache.mkdir(parents=True, exist_ok=True)
    cached = {mid: cache / f"{mid}.parquet" for mid in match_ids}
    missing = [mid for mid, p in cached.items() if not p.exists()]
    if missing:
        ws = _reader(league, season)
        events = ws.read_events(match_id=missing).reset_index()
        for mid, group in events.groupby("game_id"):
            group.to_parquet(cache / f"{int(mid)}.parquet")
        # WhoScored ids may differ from requested ids if remapped; recheck
        still_missing = [mid for mid, p in cached.items() if not p.exists()]
        if still_missing:
            raise FileNotFoundError(
                f"No events cached for match ids {still_missing}"
            )
    return pd.concat(
        [pd.read_parquet(p) for p in cached.values()], ignore_index=True
    )
