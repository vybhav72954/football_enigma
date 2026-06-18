"""Shared pass-chain builder for the third-man robustness battery.

One row per completed open-play pass, enriched with everything the
falsification tests need that the headline ``_third_man`` table lacks:

- ``feeder`` / ``hold_time`` — who fed this player and how long they held it
  (the low-touch / first-time-connector test).
- the recipient's immediate next pass: its identity (``nxt_recipient``), its
  geometry, and its unlock flag under *every* definition (``n_prog`` ... the
  definition-ladder test) plus a pointer (``nxt_idx``) so xT can be joined later.
- team-level "how fast does it become dangerous" flags (``tm2``/``tm3``/``u5s``/
  ``u10s`` — the time-to-progression test).
- per-pass self unlock flags under six definitions so "unlock" is never a single
  threshold.

A *connector* under definition D = a pass that is itself NOT a D-unlock (a simple
recycle) whose recipient's next pass IS a D-unlock. ``u_primary``/``n_primary``
reproduce the headline definition (progressive-by-distance OR final-third entry
OR shot assist). xT is added downstream (needs a fitted surface) and is the one
definition not built here.

Importable sibling (run-dir is on ``sys.path``); reused by the La Liga battery,
the career replication, and the Spain out-of-club test.
"""

import ast

import numpy as np
import pandas as pd

from football_enigma.data import statsbomb as sb_data
from football_enigma.data.build import build_positions
from football_enigma.data.statsbomb_ids import CompSeason
from football_enigma.utils.paths import PROCESSED_DIR

GOAL = (105.0, 34.0)
HALF_CENTRAL = 11.33  # central third half-width in metres (68 / 3 / 2)

CENTRAL_MID = {
    "Center Defensive Midfield", "Left Defensive Midfield",
    "Right Defensive Midfield", "Center Midfield",
    "Left Center Midfield", "Right Center Midfield",
}
ATTACK = {
    "Right Wing", "Left Wing", "Center Forward", "Right Center Forward",
    "Left Center Forward", "Secondary Striker", "Center Attacking Midfield",
    "Right Attacking Midfield", "Left Attacking Midfield",
}

_PERIOD_OFFSET = {1: 0, 2: 2700, 3: 5400, 4: 6300, 5: 7200}
_PASS_COLS = ["pass_outcome", "pass_recipient", "pass_end_location", "location",
              "pass_shot_assist", "pass_goal_assist", "duration", "timestamp",
              "period", "under_pressure"]


def _to_m(v):
    xy = ast.literal_eval(v) if isinstance(v, str) else v
    return xy[0] * 105 / 120, xy[1] * 68 / 80


def _is_loc(v):
    return isinstance(v, str) and v.startswith("[")


def _abs_t(period, ts):
    off = _PERIOD_OFFSET.get(int(period), 0) if pd.notna(period) else 0
    if not isinstance(ts, str):
        return np.nan
    hh, mm, ss = ts.split(":")
    return off + int(hh) * 3600 + int(mm) * 60 + float(ss)


def _third(x):
    return 0 if x < 35 else (1 if x < 70 else 2)


def match_chains(events: pd.DataFrame, match_id: int, pos_map: dict) -> pd.DataFrame:
    """Pass-level chain records for one match (see module docstring)."""
    ev = events
    for c in _PASS_COLS:
        if c not in ev:
            ev[c] = np.nan
    P = ev[(ev["type"] == "Pass") & ev["pass_outcome"].isna()
           & (ev["team"] == ev["possession_team"]) & ev["player"].notna()
           & ev["location"].apply(_is_loc)
           & ev["pass_end_location"].apply(_is_loc)].copy()
    if not len(P):
        return pd.DataFrame()

    s = np.array([_to_m(v) for v in P["location"]])
    e = np.array([_to_m(v) for v in P["pass_end_location"]])
    d0 = np.hypot(GOAL[0] - s[:, 0], GOAL[1] - s[:, 1])
    d1 = np.hypot(GOAL[0] - e[:, 0], GOAL[1] - e[:, 1])
    prog = (d1 <= 0.75 * d0) & ((d0 - d1) >= 5)
    fte = (s[:, 0] < 70) & (e[:, 0] >= 70)
    cfte = (e[:, 0] >= 70) & (np.abs(e[:, 1] - 34) <= HALF_CENTRAL)
    st3 = np.array([_third(x) for x in s[:, 0]])
    et3 = np.array([_third(x) for x in e[:, 0]])
    line = (et3 > st3) & ((e[:, 0] - s[:, 0]) >= 5)
    assist = ((P["pass_shot_assist"] == True)                        # noqa: E712
              | (P["pass_goal_assist"] == True)).to_numpy()          # noqa: E712
    advrecv = P["pass_recipient"].map(
        lambda r: pos_map.get(r) in ATTACK).fillna(False).to_numpy()
    primary = prog | fte | assist

    P = P.assign(
        ev_index=P["index"], match_id=match_id,
        x=s[:, 0], y=s[:, 1], ex=e[:, 0], ey=e[:, 1],
        dx=e[:, 0] - s[:, 0], dy=e[:, 1] - s[:, 1],
        length=np.hypot(e[:, 0] - s[:, 0], e[:, 1] - s[:, 1]),
        pressed=(P["under_pressure"] == True).astype(int),            # noqa: E712
        central3=(np.abs(s[:, 1] - 34) <= HALF_CENTRAL).astype(int),
        own=(s[:, 0] < 35).astype(int), final=(s[:, 0] >= 70).astype(int),
        u_prog=prog.astype(int), u_fte=fte.astype(int), u_cfte=cfte.astype(int),
        u_line=line.astype(int), u_advrecv=advrecv.astype(int),
        u_assist=assist.astype(int), u_primary=primary.astype(int),
        t=[_abs_t(p, ts) for p, ts in zip(P["period"], P["timestamp"])],
        dur=pd.to_numeric(P["duration"], errors="coerce").fillna(0.0).to_numpy(),
    )

    rows = []
    flagcols = ["u_prog", "u_fte", "u_cfte", "u_line", "u_advrecv", "u_assist",
                "u_primary"]
    for _, g in P.groupby("possession"):
        g = g.sort_values("ev_index")
        recs = g.to_dict("records")
        idx = np.array([r["ev_index"] for r in recs])
        ts = np.array([r["t"] for r in recs])
        prim = np.array([r["u_primary"] for r in recs])
        passers = [r["player"] for r in recs]
        # feeder / hold-time: running "who last passed to me, and when it arrived"
        last_to = {}
        for k, r in enumerate(recs):
            feeder, recv_t = last_to.get(r["player"], (None, np.nan))
            r["feeder"] = feeder
            r["hold_time"] = (r["t"] - recv_t) if pd.notna(recv_t) else np.nan
            last_to[r["pass_recipient"]] = (r["player"], r["t"] + r["dur"])
        # recipient's next pass + team-window unlocks
        for k, r in enumerate(recs):
            r["tm2"] = int(prim[k + 1:k + 3].any())
            r["tm3"] = int(prim[k + 1:k + 4].any())
            after = np.arange(k + 1, len(recs))
            r["u5s"] = int(any(prim[j] for j in after if ts[j] <= ts[k] + 5))
            r["u10s"] = int(any(prim[j] for j in after if ts[j] <= ts[k] + 10))
            nxt = next((recs[j] for j in after if passers[j] == r["pass_recipient"]),
                       None)
            if nxt is None:
                r["nxt_idx"] = -1
                r["nxt_recipient"] = None
                r["nxt_x"] = r["nxt_dx"] = r["nxt_len"] = r["nxt_t"] = np.nan
                for fc in flagcols:
                    r["n" + fc[1:]] = 0
            else:
                r["nxt_idx"] = nxt["ev_index"]
                r["nxt_recipient"] = nxt["pass_recipient"]
                r["nxt_x"], r["nxt_dx"] = nxt["x"], nxt["dx"]
                r["nxt_len"], r["nxt_t"] = nxt["length"], nxt["t"]
                for fc in flagcols:
                    r["n" + fc[1:]] = nxt[fc]
            rows.append(r)
    out = pd.DataFrame(rows)
    keep = ["match_id", "possession", "ev_index", "period", "t", "player",
            "pass_recipient", "feeder", "team", "x", "y", "ex", "ey", "dx", "dy",
            "length", "pressed", "central3", "own", "final", "hold_time",
            "u_prog", "u_fte", "u_cfte", "u_line", "u_advrecv", "u_assist",
            "u_primary", "nxt_idx", "nxt_recipient", "nxt_x", "nxt_dx", "nxt_len",
            "nxt_t", "n_prog", "n_fte", "n_cfte", "n_line", "n_advrecv",
            "n_assist", "n_primary", "tm2", "tm3", "u5s", "u10s"]
    return out[[c for c in keep if c in out.columns]].rename(
        columns={"pass_recipient": "recipient"})


def build_chains(comp: CompSeason, name: str, refresh: bool = False) -> pd.DataFrame:
    """Cache-first enriched pass chains for every match of a competition."""
    out = PROCESSED_DIR / f"{name}_chains_v2.parquet"
    if out.exists() and not refresh:
        return pd.read_parquet(out)
    m = sb_data.load_matches(comp)
    pos = build_positions(comp, name)
    pos_map = dict(zip(pos["player"], pos["position"]))
    frames = []
    for i, mid in enumerate(m["match_id"].astype(int)):
        frames.append(match_chains(sb_data.load_events(mid), mid, pos_map))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(m)} matches...")
    pc = pd.concat(frames, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    pc.to_parquet(out)
    return pc
