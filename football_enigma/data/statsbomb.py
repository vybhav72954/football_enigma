"""Cached loaders for StatsBomb open data (events, matches, 360 frames).

Raw pulls are cached under data/raw/statsbomb/ so the analysis never
re-downloads. 360 freeze-frames are fetched as raw JSON from the open-data
repo because statsbombpy's frame parser is incompatible with pandas 3.
"""

import json
import warnings
from pathlib import Path

import pandas as pd
import requests

from football_enigma.data.statsbomb_ids import CompSeason
from football_enigma.utils.paths import RAW_DIR

_SB_DIR = RAW_DIR / "statsbomb"
_360_URL = (
    "https://raw.githubusercontent.com/statsbomb/open-data"
    "/master/data/three-sixty/{match_id}.json"
)


def load_matches(comp: CompSeason) -> pd.DataFrame:
    cache = _SB_DIR / f"matches_{comp.competition_id}_{comp.season_id}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    from statsbombpy import sb

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matches = sb.matches(
            competition_id=comp.competition_id, season_id=comp.season_id
        )
    # mixed-type object columns (e.g. two manager ids joined by a comma)
    # break pyarrow — stringify object columns before caching
    for col in matches.columns:
        if matches[col].dtype == object:
            matches[col] = matches[col].astype(str)
    cache.parent.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(cache)
    return matches


def load_events(match_id: int) -> pd.DataFrame:
    cache = _SB_DIR / "events" / f"{match_id}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    from statsbombpy import sb

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        events = sb.events(match_id=match_id)
    # Nested list/dict columns (location, freeze_frame, ...) break parquet;
    # serialise them to JSON strings and decode on read via decode_locations.
    events = events.map(
        lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(cache)
    return events


def decode_locations(events: pd.DataFrame, columns: tuple[str, ...] = ("location", "pass_end_location", "carry_end_location")) -> pd.DataFrame:
    events = events.copy()
    for col in columns:
        if col in events.columns:
            events[col] = events[col].map(
                lambda v: json.loads(v) if isinstance(v, str) else v
            )
    return events


def load_frames(match_id: int) -> list[dict]:
    """360 freeze-frames for one match as raw dicts (event_uuid,
    visible_area, freeze_frame). Returns [] if the match has no 360 data."""
    cache = _SB_DIR / "three-sixty" / f"{match_id}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    resp = requests.get(_360_URL.format(match_id=match_id), timeout=60)
    if resp.status_code == 404:
        frames = []
    else:
        resp.raise_for_status()
        frames = resp.json()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(frames), encoding="utf-8")
    return frames


def team_matches(comp: CompSeason, team: str) -> pd.DataFrame:
    matches = load_matches(comp)
    mask = (matches["home_team"] == team) | (matches["away_team"] == team)
    return matches[mask].sort_values("match_date").reset_index(drop=True)
