"""Kroos's four international tournaments as one per-match table.

Shared by the aging-curve and WC2018 stress-test notebooks. Sources mix:
WC 2014 is Opta (full-event WhoScored scrape), WC 2018 / Euro 2020 /
Euro 2024 are StatsBomb (Germany's matches are cached; that is all Kroos
plays in). Cadence uses spell gaps (metrics.cadence.spell_gaps) so the two
providers' different event granularities stay comparable.

Persists to data/processed/:
  kroos_tournament_matches.parquet  — one row per Germany match
  kroos_tournament_spell_gaps.parquet — one row per Kroos involvement gap
"""

import glob

import pandas as pd

from football_enigma.data import statsbomb as sb_data
from football_enigma.data.schema import opta_passes, statsbomb_passes
from football_enigma.data.statsbomb_ids import (
    EURO_2020,
    EURO_2024,
    WORLD_CUP_2018,
)
from football_enigma.metrics.aggregates import flag_progressive, flag_switch
from football_enigma.metrics.cadence import spell_gaps
from football_enigma.utils.paths import PROCESSED_DIR, RAW_DIR

KROOS = "Toni Kroos"
TOURNAMENT_ORDER = ["WC 2014", "WC 2018", "Euro 2020", "Euro 2024"]

# the on-ball event set for StatsBomb spell cadence; Opta uses is_touch
_SB_ONBALL = [
    "Pass", "Carry", "Shot", "Dribble", "Ball Receipt*", "Ball Recovery",
    "Clearance", "Interception",
]

_MATCHES_PQ = PROCESSED_DIR / "kroos_tournament_matches.parquet"
_GAPS_PQ = PROCESSED_DIR / "kroos_tournament_spell_gaps.parquet"


def _match_row(tournament, match_id, opponent, result, minutes, passes,
               team_passes_on):
    kroos = passes[passes["player"] == KROOS].copy()
    kroos["progressive"] = flag_progressive(kroos)
    kroos["switch"] = flag_switch(kroos)
    return {
        "tournament": tournament,
        "match_id": match_id,
        "opponent": opponent,
        "result": result,
        "minutes": minutes,
        "passes": len(kroos),
        "completed": int(kroos["outcome"].sum()),
        "progressive": int(kroos["progressive"].sum()),
        "switches": int(kroos["switch"].sum()),
        "set_piece_switches": int(
            (kroos["switch"] & kroos["set_piece"]).sum()
        ),
        "team_passes_on": team_passes_on,
    }


def _wc2014():
    files = glob.glob(str(RAW_DIR / "whoscored" / "events" / "*.parquet"))
    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    # the events cache now also holds the Bayern 2013/14 scrape; isolate WC 2014
    raw = raw[raw["league"] == "INT-World Cup"]
    germany = raw[raw["team"] == "Germany"]
    rows, gap_chunks = [], []
    for gid, match in germany.groupby("game_id"):
        off = match.loc[
            (match["type"] == "SubstitutionOff") & (match["player"] == KROOS),
            "minute",
        ]
        end = float(match["expanded_minute"].max())
        off_minute = float(off.iloc[0]) if len(off) else end
        passes = opta_passes(match[match["type"] == "Pass"])
        on_pitch = passes["minute"] < off_minute
        label = str(match["game"].iloc[0])  # "2014-06-21 Germany-Ghana"
        opponent = label.split(" ", 1)[1].replace("Germany", "").strip("-")
        touches = match[match["is_touch"] & (match["player"] == KROOS)]
        gaps = spell_gaps(touches.rename(columns={"game_id": "match_id"}))
        gap_chunks.append(gaps.assign(tournament="WC 2014"))
        rows.append(_match_row(
            "WC 2014", int(gid), opponent, None,
            min(off_minute, end), passes, int(on_pitch.sum()),
        ))
    return rows, gap_chunks


def _statsbomb(tournament, comp):
    matches = sb_data.team_matches(comp, "Germany")
    rows, gap_chunks = [], []
    for _, m in matches.iterrows():
        mid = int(m["match_id"])
        events = sb_data.decode_locations(sb_data.load_events(mid))
        subs = events[events["type"] == "Substitution"]
        off = subs.loc[subs["player"] == KROOS, "minute"]
        end = float(events["minute"].max())
        off_minute = float(off.iloc[0]) if len(off) else end
        passes = statsbomb_passes(events, mid)
        passes = passes[passes["team"] == "Germany"]
        on_pitch = passes["minute"] < off_minute
        if m["home_team"] == "Germany":
            opponent, gf, ga = m["away_team"], m["home_score"], m["away_score"]
        else:
            opponent, gf, ga = m["home_team"], m["away_score"], m["home_score"]
        onball = events[
            events["type"].isin(_SB_ONBALL) & (events["player"] == KROOS)
        ]
        gaps = spell_gaps(onball.assign(match_id=mid)[
            ["match_id", "player", "period", "minute", "second"]
        ])
        gap_chunks.append(gaps.assign(tournament=tournament))
        rows.append(_match_row(
            tournament, mid, opponent, f"{int(gf)}–{int(ga)}",
            off_minute, passes, int(on_pitch.sum()),
        ))
    return rows, gap_chunks


def load(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-match table, spell-gap table), built from raw caches once."""
    if _MATCHES_PQ.exists() and _GAPS_PQ.exists() and not refresh:
        return pd.read_parquet(_MATCHES_PQ), pd.read_parquet(_GAPS_PQ)

    rows, gap_chunks = _wc2014()
    for tournament, comp in [
        ("WC 2018", WORLD_CUP_2018),
        ("Euro 2020", EURO_2020),
        ("Euro 2024", EURO_2024),
    ]:
        r, g = _statsbomb(tournament, comp)
        rows += r
        gap_chunks += g

    matches = pd.DataFrame(rows)
    gaps = pd.concat(gap_chunks, ignore_index=True)[
        ["tournament", "match_id", "gap"]
    ]
    matches.to_parquet(_MATCHES_PQ)
    gaps.to_parquet(_GAPS_PQ)
    return matches, gaps


def tournament_summary(matches: pd.DataFrame, gaps: pd.DataFrame) -> pd.DataFrame:
    """Per-tournament aggregates: raw per-90 and per-100-team-passes."""
    agg = matches.groupby("tournament", observed=True).agg(
        matches=("match_id", "size"),
        minutes=("minutes", "sum"),
        passes=("passes", "sum"),
        completed=("completed", "sum"),
        progressive=("progressive", "sum"),
        switches=("switches", "sum"),
        team_passes_on=("team_passes_on", "sum"),
    )
    p90 = agg["minutes"] / 90
    agg["comp_pct"] = agg["completed"] / agg["passes"] * 100
    for col in ("passes", "progressive", "switches"):
        agg[f"{col}_p90"] = agg[col] / p90
        agg[f"{col}_per100"] = agg[col] / agg["team_passes_on"] * 100
    med = gaps.groupby("tournament", observed=True)["gap"].median()
    agg["median_spell_gap"] = med
    return agg.reindex(TOURNAMENT_ORDER).reset_index()
