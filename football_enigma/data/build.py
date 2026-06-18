"""Assemble processed (canonical) datasets from the raw cache.

Raw match events are cached per match by data.statsbomb / data.whoscored;
these builders concatenate, convert to the canonical schema, and persist a
single parquet per competition to data/processed/.
"""

import pandas as pd

from football_enigma.data import statsbomb as sb_data
from football_enigma.data.schema import statsbomb_actions
from football_enigma.data.statsbomb_ids import CompSeason
from football_enigma.metrics.buildup import backline_receptions
from football_enigma.metrics.defending import defensive_profile
from football_enigma.utils.paths import PROCESSED_DIR

_DECODE_COLS = ("location", "pass_end_location", "carry_end_location")


def build_statsbomb_actions(comp: CompSeason, name: str, refresh: bool = False) -> pd.DataFrame:
    """Canonical actions for every match of a competition-season.

    Persists to data/processed/{name}_actions.parquet and returns it.
    """
    out = PROCESSED_DIR / f"{name}_actions.parquet"
    if out.exists() and not refresh:
        return pd.read_parquet(out)

    matches = sb_data.load_matches(comp)
    chunks = []
    for mid in matches["match_id"]:
        events = sb_data.decode_locations(
            sb_data.load_events(int(mid)), _DECODE_COLS
        )
        chunks.append(statsbomb_actions(events, int(mid)))
    actions = pd.concat(chunks, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    actions.to_parquet(out)
    return actions


def build_defensive_profile(
    comp: CompSeason, name: str, refresh: bool = False
) -> pd.DataFrame:
    """Per-player defensive-action totals across a competition-season.

    Sums metrics.defending.defensive_profile over every match. Persists to
    data/processed/{name}_defending.parquet.
    """
    out = PROCESSED_DIR / f"{name}_defending.parquet"
    if out.exists() and not refresh:
        return pd.read_parquet(out)

    chunks = [
        defensive_profile(sb_data.load_events(int(mid)))
        for mid in sb_data.load_matches(comp)["match_id"]
    ]
    totals = (
        pd.concat(chunks, ignore_index=True)
        .groupby("player", as_index=False)
        .sum(numeric_only=True)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    totals.to_parquet(out)
    return totals


def build_backline_receptions(
    comp: CompSeason, name: str, refresh: bool = False
) -> pd.DataFrame:
    """Per-player build-up receptions across a competition-season.

    Sums metrics.buildup.backline_receptions over every match (team taken as
    the player's modal team). Persists to data/processed/{name}_buildup.parquet.
    """
    out = PROCESSED_DIR / f"{name}_buildup.parquet"
    if out.exists() and not refresh:
        return pd.read_parquet(out)

    chunks = [
        backline_receptions(sb_data.load_events(int(mid)))
        for mid in sb_data.load_matches(comp)["match_id"]
    ]
    allrows = pd.concat(chunks, ignore_index=True)
    counts = allrows.groupby("player", as_index=False)[
        ["from_backline", "total_received"]
    ].sum()
    team = allrows.groupby("player")["team"].agg(lambda s: s.mode().iloc[0])
    totals = counts.merge(team.rename("team"), on="player")
    out.parent.mkdir(parents=True, exist_ok=True)
    totals.to_parquet(out)
    return totals


def build_positions(comp: CompSeason, name: str, refresh: bool = False) -> pd.DataFrame:
    """Modal on-pitch position per player across a competition-season.

    StatsBomb tags each event with the player's current position, but the
    canonical processed tables drop it; this recovers a single position per
    player (their most frequent one) from the raw events. Persists to
    data/processed/{name}_positions.parquet, with columns player, position.
    """
    out = PROCESSED_DIR / f"{name}_positions.parquet"
    if out.exists() and not refresh:
        return pd.read_parquet(out)

    chunks = []
    for mid in sb_data.load_matches(comp)["match_id"]:
        events = sb_data.load_events(int(mid))
        if "position" not in events.columns:
            continue
        rows = events[["player", "position"]].dropna()
        rows = rows.assign(
            position=rows["position"].map(
                lambda v: v.get("name") if isinstance(v, dict) else v
            )
        )
        chunks.append(rows)
    positions = (
        pd.concat(chunks, ignore_index=True)
        .groupby("player", as_index=False)["position"]
        .agg(lambda s: s.mode(dropna=True).iloc[0])
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    positions.to_parquet(out)
    return positions


def player_minutes(comp: CompSeason) -> pd.Series:
    """Approximate minutes per player from lineups/substitution events.

    Uses each match's maximum recorded minute as match length; players who
    started play from 0, substitutes from their entry minute, substituted-
    off players until their exit minute.
    """
    matches = sb_data.load_matches(comp)
    totals: dict[str, float] = {}
    for mid in matches["match_id"]:
        events = sb_data.load_events(int(mid))
        match_end = float(events["minute"].max())
        starters = set(
            events.loc[events["type"] == "Starting XI", "tactics"]
            .map(_lineup_players)
            .explode()
            .dropna()
        )
        subs = events[events["type"] == "Substitution"]
        on_at = {p: 0.0 for p in starters}
        off_at = {}
        for _, s in subs.iterrows():
            off_at[s["player"]] = float(s["minute"])
            replacement = s.get("substitution_replacement")
            if isinstance(replacement, str):
                on_at[replacement] = float(s["minute"])
        for player, start in on_at.items():
            end = off_at.get(player, match_end)
            totals[player] = totals.get(player, 0.0) + max(end - start, 0.0)
    return pd.Series(totals, name="minutes")


def _lineup_players(tactics) -> list[str]:
    import json

    if isinstance(tactics, str):
        tactics = json.loads(tactics)
    if not isinstance(tactics, dict):
        return []
    return [p["player"]["name"] for p in tactics.get("lineup", [])]
