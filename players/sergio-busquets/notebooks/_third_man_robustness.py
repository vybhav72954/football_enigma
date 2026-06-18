"""ROBUSTNESS / FALSIFICATION battery for the third-man crown (La Liga 2015/16).

Reads the enriched ``_chains`` table and runs the nine cached falsification
tests, in the order the user prioritised. Each asks "is the #1 a definitional
trick?" from a different angle; honest cluster/non-dominance findings are kept.

  1. DEFINITION LADDER     - does #1 survive six definitions of "unlock"?
  2. MESSI REMOVAL         - drop every chain Messi touches; does he still lead?
  3. INIESTA REMOVAL       - same for Iniesta (and both together).
  4. RECIPIENT FIXED EFFECTS - linear-probability model: controlling for who
     plays the next pass (actor dummies) and where (zone), does "fed by
     Busquets" still add value?  (statsmodels absent -> numpy LPM + bootstrap CI)
  5. ZONE BASELINE         - his next-pass unlock % vs Barca-others, by zone.
  6. TIME-TO-PROGRESSION   - does it spike early (connector) or just eventually?
  7. DIRECTIONAL DISGUISE  - his backward/lateral passes -> forward next pass.
  8. LOW-TOUCH / FIRST-TIME- quick releases (hold-time) and the third man.
  9. NEGATIVE CONTROL      - metrics he should NOT top (the model is honest).

Caches two small tables for figures 08/09:
  laliga1516_thirdman_board.parquet  (per-90 board + over-expected)
  laliga1516_receiver_uplift.parquet (named-receiver uplift dumbbell)

Run:  .venv/Scripts/python -X utf8 players/sergio-busquets/notebooks/_third_man_robustness.py
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from _chains import CENTRAL_MID, build_chains
from football_enigma.data.build import build_positions
from football_enigma.data.statsbomb_ids import BUSQUETS_PLAYER_NAME, LA_LIGA
from football_enigma.metrics.xt import fit_xt
from football_enigma.utils.paths import PROCESSED_DIR
from football_enigma.viz.charts import _short_name

SUBJ = BUSQUETS_PLAYER_NAME
COMP = LA_LIGA["2015/2016"]
MIN_MIN, MIN_PASSES = 1500, 300
rng = np.random.default_rng(7)

pc = build_chains(COMP, "laliga1516")
print(f"chains: {len(pc)} passes")

# ---- pool metadata ----
agg = pd.read_parquet(PROCESSED_DIR / "laliga1516_player_agg.parquet")
team = pd.read_parquet(PROCESSED_DIR / "laliga1516_buildup.parquet")[["player", "team"]]
pos = build_positions(COMP, "laliga1516")
meta = agg[["player", "minutes"]].merge(team, on="player").merge(pos, on="player")
mins = dict(zip(meta["player"], meta["minutes"]))
tm_team = dict(zip(meta["player"], meta["team"]))
passes_per = pc.groupby("player").size()
POOL = set(meta[(meta["minutes"] >= MIN_MIN)
                & meta["position"].isin(CENTRAL_MID)
                & (meta["player"].map(passes_per).fillna(0) >= MIN_PASSES)]["player"])
print(f"CM pool: {len(POOL)}")


def name(sub):  # find the data name whose short form matches
    hits = [p for p in pc["player"].dropna().unique() if sub in p]
    return hits[0] if hits else None


MESSI, INIESTA = name("Messi"), name("Iniesta")

# ---- xT self-flag + recipient-next xT (the one definition built downstream) ----
_xm = fit_xt(pd.read_parquet(PROCESSED_DIR / "laliga1516_actions.parquet"))
xt = _xm.surface
r0, c0 = _xm.cell_of(pc["x"], pc["y"])
r1, c1 = _xm.cell_of(pc["ex"], pc["ey"])
pc["u_xt"] = (xt[r1, c1] - xt[r0, c0] > 0).astype(int)
nxt_xt = pc.set_index(["match_id", "ev_index"])["u_xt"]
key = list(zip(pc["match_id"], pc["nxt_idx"]))
pc["n_xt"] = [nxt_xt.get(k, 0) if k[1] != -1 else 0 for k in key]


def board(series, label, mincount=None, minn=1, n=8, quiet=False):
    s = series.dropna()
    s = s[[p in POOL for p in s.index]]
    if mincount is not None:
        s = s[[mincount.get(p, 0) >= minn for p in s.index]]
    s = s.sort_values(ascending=False)
    rank = list(s.index).index(SUBJ) + 1 if SUBJ in s.index else -1
    if not quiet:
        df = pd.DataFrame({"short": [_short_name(p) for p in s.index],
                           "team": [tm_team.get(p, "?") for p in s.index],
                           label: s.values})
        print(f"\n=== {label}: Busquets rank {rank} of {len(s)} ===")
        print(df.head(n).round(2).to_string(index=False))
    return rank, len(s)


def per90(mask_conn):
    cnt = pc[mask_conn].groupby("player").size()
    return (cnt / (pd.Series(mins) / 90)).dropna()


# ============================ 1. DEFINITION LADDER ============================
print("\n" + "#" * 70 + "\n# 1. DEFINITION LADDER  (connector = simple feed -> recipient's next pass unlocks)\n" + "#" * 70)
DEFS = [("primary (prog|fte|assist)", "primary"), ("progressive by distance", "prog"),
        ("enters final third", "fte"), ("enters CENTRAL final third", "cfte"),
        ("increases xT", "xt"), ("breaks a line (proxy)", "line"),
        ("reaches advanced teammate", "advrecv")]
ladder = []
for lbl, d in DEFS:
    conn = (pc[f"u_{d}"] == 0) & (pc[f"n_{d}"] == 1)
    p90 = per90(conn)
    r, npool = board(p90, f"per90 [{lbl}]", quiet=True)
    top = _short_name(p90[[p in POOL for p in p90.index]].idxmax())
    ladder.append((lbl, round(p90.get(SUBJ, np.nan), 2), r, npool, top))
lad = pd.DataFrame(ladder, columns=["unlock definition", "Busq/90", "rank", "of", "#1"])
print(lad.to_string(index=False))
print(f"-> Busquets is #1 under {sum(lad['rank'] == 1)}/{len(lad)} definitions; "
      f"worst rank {lad['rank'].max()}")

# =========================== 2 & 3. STAR REMOVAL =============================
print("\n" + "#" * 70 + "\n# 2 & 3. MESSI / INIESTA REMOVAL  (primary definition)\n" + "#" * 70)


def removal(label, drop_recipient=(), drop_anywhere=()):
    keep = pd.Series(True, index=pc.index)
    if drop_recipient:
        keep &= ~pc["recipient"].isin(drop_recipient)
    if drop_anywhere:
        keep &= ~pc["recipient"].isin(drop_anywhere) & ~pc["nxt_recipient"].isin(drop_anywhere)
    sub = pc[keep]
    conn = (sub["u_primary"] == 0) & (sub["n_primary"] == 1)
    cnt = sub[conn].groupby("player").size()
    p90 = (cnt / (pd.Series(mins) / 90)).dropna()
    p90 = p90[[p in POOL for p in p90.index]].sort_values(ascending=False)
    r = list(p90.index).index(SUBJ) + 1 if SUBJ in p90.index else -1
    barca = [p for p in p90.index if tm_team.get(p) == "Barcelona"]
    rb = barca.index(SUBJ) + 1 if SUBJ in barca else -1
    print(f"  {label:42s} Busq {p90.get(SUBJ, np.nan):5.2f}/90  rank {r}/{len(p90)}  "
          f"(Barca rank {rb}/{len(barca)})")


removal("baseline (no removal)")
removal("drop chains where Messi RECEIVES", drop_recipient=[MESSI])
removal("drop chains where Messi is ANYWHERE", drop_anywhere=[MESSI])
removal("drop chains where Iniesta RECEIVES", drop_recipient=[INIESTA])
removal("drop chains where Iniesta is ANYWHERE", drop_anywhere=[INIESTA])
removal("drop Messi AND Iniesta anywhere", drop_anywhere=[MESSI, INIESTA])

# ======================= 4. RECIPIENT FIXED EFFECTS (LPM) =====================
print("\n" + "#" * 70 + "\n# 4. RECIPIENT FIXED EFFECTS  (does 'fed by Busquets' survive actor+zone control?)\n" + "#" * 70)
# unit = every Barca pass that HAD a feed; y = does THIS pass unlock (u_primary).
# treatment = fed by Busquets; controls = actor dummies (who plays it) + zone + pressure + length.
fe = pc[(pc["team"] == "Barcelona") & pc["feeder"].notna()].copy()
fe["actor"] = fe["player"]
actors = fe["actor"].value_counts()
keep_actors = set(actors[actors >= 50].index)
fe = fe[fe["actor"].isin(keep_actors)]
fe["zone"] = (fe["x"].apply(lambda v: 0 if v < 35 else (1 if v < 70 else 2)) * 2
              + (fe["central3"] == 0).astype(int))
y = fe["u_primary"].to_numpy(float)
D_actor = pd.get_dummies(fe["actor"], prefix="a", drop_first=True).to_numpy(float)
D_zone = pd.get_dummies(fe["zone"], prefix="z", drop_first=True).to_numpy(float)
busq = (fe["feeder"] == SUBJ).to_numpy(float).reshape(-1, 1)
extra = np.column_stack([fe["pressed"].to_numpy(float),
                         (fe["length"].to_numpy(float) - fe["length"].mean()) / fe["length"].std()])
X = np.column_stack([np.ones(len(fe)), busq, extra, D_zone, D_actor])
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
naive = (fe.loc[fe["feeder"] == SUBJ, "u_primary"].mean()
         - fe.loc[fe["feeder"] != SUBJ, "u_primary"].mean()) * 100
# bootstrap CI over matches
coefs = []
mids = fe["match_id"].unique()
for _ in range(200):
    samp = fe[fe["match_id"].isin(rng.choice(mids, len(mids), replace=True))]
    if (samp["feeder"] == SUBJ).sum() < 20:
        continue
    yy = samp["u_primary"].to_numpy(float)
    Xb = np.column_stack([
        np.ones(len(samp)), (samp["feeder"] == SUBJ).to_numpy(float),
        samp["pressed"].to_numpy(float),
        (samp["length"].to_numpy(float) - samp["length"].mean()) / samp["length"].std(),
        pd.get_dummies(samp["zone"], prefix="z", drop_first=True).to_numpy(float),
        pd.get_dummies(samp["actor"], prefix="a", drop_first=True).to_numpy(float)])
    bb, *_ = np.linalg.lstsq(Xb, yy, rcond=None)
    coefs.append(bb[1] * 100)
lo, hi = np.percentile(coefs, [2.5, 97.5])
print(f"  unit = {len(fe)} Barca passes with a feed ({len(keep_actors)} actor FEs, 6 zones)")
print(f"  naive uplift (fed-by-Busquets vs not):          {naive:+.2f} pp")
print(f"  LPM coef on 'fed by Busquets' (actor+zone+press+len controlled): "
      f"{beta[1] * 100:+.2f} pp  [95% CI {lo:+.2f}, {hi:+.2f}]")
print("  -> the same actor, same zone, is more likely to unlock after a Busquets feed"
      if lo > 0 else "  -> effect not distinguishable from zero")

# ============================ 5. ZONE BASELINE ===============================
print("\n" + "#" * 70 + "\n# 5. ZONE BASELINE  (his next-pass unlock % vs Barca-others, by feed zone)\n" + "#" * 70)
ZN = {0: "Def-central", 1: "Def-wide", 2: "Mid-central", 3: "Mid-wide",
      4: "Final-central", 5: "Final-wide"}
simple_b = pc[(pc["u_primary"] == 0) & (pc["team"] == "Barcelona")].copy()
simple_b["zone"] = (simple_b["x"].apply(lambda v: 0 if v < 35 else (1 if v < 70 else 2)) * 2
                    + (simple_b["central3"] == 0).astype(int))
rows = []
for z, zl in ZN.items():
    zb = simple_b[simple_b["zone"] == z]
    b = zb[zb["player"] == SUBJ]
    o = zb[zb["player"] != SUBJ]
    if len(b) >= 15:
        rows.append((zl, len(b), b["n_primary"].mean() * 100,
                     o["n_primary"].mean() * 100,
                     (b["n_primary"].mean() - o["n_primary"].mean()) * 100))
zt = pd.DataFrame(rows, columns=["zone", "n_busq", "busq%", "barca_others%", "diff_pp"])
print(zt.round(1).to_string(index=False))

# ========================= 6. TIME-TO-PROGRESSION ============================
print("\n" + "#" * 70 + "\n# 6. TIME-TO-PROGRESSION  (spike early, or only eventually?)\n" + "#" * 70)
simple = pc[pc["u_primary"] == 0]
sb = simple[simple["player"] == SUBJ]
so = simple[(simple["team"] == "Barcelona") & (simple["player"] != SUBJ)]
sl = simple[simple["player"].isin(POOL)]
hz = [("recipient's next pass", "n_primary"), ("within 2 passes", "tm2"),
      ("within 3 passes", "tm3"), ("within 5 seconds", "u5s"),
      ("within 10 seconds", "u10s")]
print(f"  {'horizon':24s} {'Busq%':>7s} {'Barca-oth%':>11s} {'leagueCM%':>10s} {'gap vs oth':>11s}")
for lbl, col in hz:
    print(f"  {lbl:24s} {sb[col].mean() * 100:7.1f} {so[col].mean() * 100:11.1f} "
          f"{sl[col].mean() * 100:10.1f} {(sb[col].mean() - so[col].mean()) * 100:+11.1f}")

# ========================= 7. DIRECTIONAL DISGUISE ===========================
print("\n" + "#" * 70 + "\n# 7. DIRECTIONAL DISGUISE  (backward/lateral feed -> forward next pass)\n" + "#" * 70)
back = pc[(pc["dx"] <= 2) & pc["nxt_idx"].ne(-1)].copy()  # non-advancing passes
back["fwd_next"] = (back["nxt_dx"] > 2).astype(int)
bb, bo = back[back["player"] == SUBJ], back[(back["team"] == "Barcelona") & (back["player"] != SUBJ)]
bl = back[back["player"].isin(POOL)]
print(f"  share of his backward/lateral passes whose NEXT pass goes forward:")
print(f"    Busquets {bb['fwd_next'].mean() * 100:.1f}%  |  Barca-others {bo['fwd_next'].mean() * 100:.1f}%  "
      f"|  league CMs {bl['fwd_next'].mean() * 100:.1f}%")
print(f"  mean forward distance gained by that next pass (m):")
print(f"    Busquets {bb.loc[bb['fwd_next'] == 1, 'nxt_dx'].mean():.1f}  |  "
      f"Barca-others {bo.loc[bo['fwd_next'] == 1, 'nxt_dx'].mean():.1f}")
rback, _ = board(back.groupby("player")["fwd_next"].mean() * 100,
                 "backward->forward-next % (rank)", back.groupby("player").size(), minn=150)

# ======================= 8. LOW-TOUCH / FIRST-TIME ===========================
print("\n" + "#" * 70 + "\n# 8. LOW-TOUCH / FIRST-TIME  (quick release -> the third man early)\n" + "#" * 70)
print("  NOTE: hold_time approximated as (his release t) - (feed arrival t); event-clock resolution.")
sbh = sb.dropna(subset=["hold_time"])
buck = pd.cut(sbh["hold_time"], [-0.01, 1, 2, 1e9], labels=["<1s", "1-2s", ">2s"])
g = sbh.groupby(buck, observed=True)["n_primary"].agg(["mean", "size"])
g["mean"] = g["mean"] * 100
print(f"  Busquets simple passes by hold-time bucket -> recipient-next unlock %:")
print(g.rename(columns={"mean": "unlock%", "size": "n"}).round(1).to_string())
lcm = simple[simple["player"].isin(POOL)].dropna(subset=["hold_time"])
print(f"  median hold-time: Busquets {sbh['hold_time'].median():.2f}s vs league CMs "
      f"{lcm['hold_time'].median():.2f}s; his share released <=1s: "
      f"{(sbh['hold_time'] <= 1).mean() * 100:.0f}% vs {(lcm['hold_time'] <= 1).mean() * 100:.0f}%")

# ========================== 9. NEGATIVE CONTROL ==============================
print("\n" + "#" * 70 + "\n# 9. NEGATIVE CONTROL  (metrics he should NOT top - is the model honest?)\n" + "#" * 70)
deff = pd.read_parquet(PROCESSED_DIR / "laliga1516_defending.parquet")
am = agg.merge(deff, on="player", how="left")
am["mins"] = am["player"].map(mins)
for c in ["interceptions", "tackles"]:
    am[c + "_p90"] = am[c] / (am["mins"] / 90)
neg = [("progressive passes /90", "progressive_p90"), ("final-third entries /90", "final_third_p90"),
       ("switches /90", "switches_p90"), ("interceptions /90", "interceptions_p90"),
       ("tackles /90", "tackles_p90")]
nrows = []
for lbl, c in neg:
    s = am.set_index("player")[c]
    r, npool = board(s, lbl, quiet=True)
    nrows.append((lbl, round(s.get(SUBJ, np.nan), 2), r, npool))
print(pd.DataFrame(nrows, columns=["metric (should NOT be #1)", "Busq", "rank", "of"]).to_string(index=False))

# ============== caches for figures 08 (hero) + 09 (uplift dumbbell) ==========
conn = (pc["u_primary"] == 0) & (pc["n_primary"] == 1)
brd = pc.groupby("player").agg(passes=("u_primary", "size")).reset_index()
brd["conn"] = brd["player"].map(pc[conn].groupby("player").size()).fillna(0).astype(int)
brd = brd.merge(meta, on="player")
brd["per90"] = brd["conn"] / (brd["minutes"] / 90)
brd["share"] = brd["conn"] / brd["passes"].clip(lower=1) * 100
brd["short"] = [_short_name(p) for p in brd["player"]]
brd = brd[(brd["minutes"] >= MIN_MIN) & (brd["passes"] >= MIN_PASSES)
          & brd["position"].isin(CENTRAL_MID)].copy()
# over-expected (kills volume) — model P(simple -> third-man unlock) from geometry+pressure+zone
allsimple = pc[pc["u_primary"] == 0].copy()
F = ["x", "y", "dx", "length", "pressed", "central3", "own", "final"]
Xs = StandardScaler().fit_transform(allsimple[F].values)
allsimple["xtm"] = LogisticRegression(max_iter=1000).fit(Xs, allsimple["n_primary"]).predict_proba(Xs)[:, 1]
over = (allsimple.assign(over=allsimple["n_primary"] - allsimple["xtm"])
        .groupby("player")["over"].mean() * 100)
brd["over_exp"] = brd["player"].map(over)
brd.sort_values("per90", ascending=False).to_parquet(PROCESSED_DIR / "laliga1516_thirdman_board.parquet")

bsimple = allsimple[allsimple["team"] == "Barcelona"].copy()
bsimple["from_busq"] = bsimple["player"] == SUBJ
rows = []
for rcpt, gg in bsimple.groupby("recipient"):
    gb, go = gg[gg["from_busq"]], gg[~gg["from_busq"]]
    if len(gb) >= 40 and len(go) >= 40:
        rows.append((rcpt, _short_name(rcpt), len(gb), gb["n_primary"].mean() * 100,
                     go["n_primary"].mean() * 100,
                     (gb["n_primary"].mean() - go["n_primary"].mean()) * 100))
upl = pd.DataFrame(rows, columns=["player", "receiver", "n_busq", "after_busq", "after_other", "uplift"])
upl.sort_values("uplift", ascending=False).to_parquet(PROCESSED_DIR / "laliga1516_receiver_uplift.parquet")
print(f"\ncached board ({len(brd)} CMs) + receiver uplift ({len(upl)} receivers)")
print("DONE - robustness battery")
