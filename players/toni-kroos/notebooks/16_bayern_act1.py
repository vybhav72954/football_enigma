"""Act I — the receipts were in their own building (Bayern 2013/14).

Kroos's last Bundesliga season before Real Madrid bought him for €25m as a
player Bayern had decided was replaceable. The profile that made him the
metronome of the next decade was already on the pitch — measured here
against his *own* Bayern teammates, the strongest midfield in Europe.

In Pep's possession side everyone passed at high volume, so raw volume is
not the separator; progression and switches are. Both figures are scoped
to Bayern's squad only (team == "Bayern"), per-game over the player's
appearances, mirroring the WC 2014 tournament charts (fig 09 / 11).

Fig 22 — passes per game vs progressive passes per game, Bayern squad.
Fig 23 — switch leaderboard, Bayern squad.

Requires the full Bayern 2013/14 scrape (data.scrape_bayern1314).

Run:  .venv/Scripts/python -X utf8 players/toni-kroos/notebooks/16_bayern_act1.py
"""

import sys

import matplotlib

matplotlib.use("Agg")
sys.stdout.reconfigure(encoding="utf-8")

import glob

import matplotlib.pyplot as plt
import pandas as pd

from football_enigma.data.schema import opta_passes
from football_enigma.metrics.aggregates import flag_progressive, flag_switch
from football_enigma.utils.paths import PROCESSED_DIR, RAW_DIR, figures_dir
from football_enigma.viz.charts import crowd_scatter
from football_enigma.viz.theme import COLORS, apply_theme, credit

SUBJECT = "Toni Kroos"
MIN_MATCHES = 10  # rotation in a 34-game title season
FIGS = figures_dir("toni-kroos")
# Boateng sits just above Kroos (label it); Schweinsteiger is ~on top of him
# (named in the subtitle instead, to avoid an unreadable label collision)
TEAMMATES = ["Jérôme Boateng", "Philipp Lahm", "Thomas Müller", "Mario Götze"]

apply_theme()

files = glob.glob(str(RAW_DIR / "whoscored" / "events" / "*.parquet"))
raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
raw = raw[raw["league"] == "GER-Bundesliga"]  # exclude the WC 2014 cache
print(f"Bayern 2013/14: {raw['game_id'].nunique()} matches loaded")

passes = opta_passes(raw)
passes = passes[passes["team"] == "Bayern"].copy()
passes["progressive"] = flag_progressive(passes)
passes["switch"] = flag_switch(passes)

games = passes.groupby("player", observed=True)["match_id"].nunique()
agg = (
    passes.groupby("player", observed=True)
    .agg(
        passes=("outcome", "size"),
        comp_pct=("outcome", lambda s: s.mean() * 100),
        progressive=("progressive", "sum"),
        switches=("switch", "sum"),
    )
    .join(games.rename("games"))
)
agg = agg[agg["games"] >= MIN_MATCHES].reset_index()
agg["passes_pg"] = agg["passes"] / agg["games"]
agg["progressive_pg"] = agg["progressive"] / agg["games"]
agg["switches_pg"] = agg["switches"] / agg["games"]
agg.to_parquet(PROCESSED_DIR / "bayern1314_player_agg.parquet")

# ---------------------------------------------------------------- fig 22
fig, ax = crowd_scatter(
    agg,
    x="passes_pg",
    y="progressive_pg",
    subject=SUBJECT,
    title="The profile was already in the building",
    subtitle="Bayern 2013/14, squad only (10+ games) | a top-3 progressor, level with Schweinsteiger "
    "— the playmaker tools were there, if not yet singular",
    xlabel="Passes per game",
    ylabel="Progressive passes per game",
    annotate=TEAMMATES,
    source="Opta via WhoScored",
)
fig.savefig(FIGS / "22_bayern_volume_vs_progressive.png")

# ---------------------------------------------------------------- fig 23
top = agg.sort_values("switches", ascending=False).head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(10, 6.5))
colors = [
    COLORS["subject"] if p == SUBJECT else COLORS["field"] for p in top["player"]
]
bars = ax.barh(top["player"], top["switches"], color=colors, height=0.62)
for bar, (_, row) in zip(bars, top.iterrows()):
    ax.text(
        bar.get_width() + 0.6, bar.get_y() + bar.get_height() / 2,
        f"{int(row['switches'])}", va="center", fontsize=10,
        color=COLORS["text"] if row["player"] == SUBJECT else COLORS["muted"],
        fontweight="bold" if row["player"] == SUBJECT else "normal",
    )
ax.set_xlabel("Switches of play (completed passes ≥30 m across the pitch)")
ax.grid(axis="y", visible=False)
ax.set_title("Bayern switched constantly — and Kroos did it most",
             loc="left", pad=20)
ax.text(
    0, 1.02,
    "Bayern 2013/14, squad only (10+ games), total switches | the next names are ball-playing "
    "defenders, and Dante is level — switching ran through the whole side",
    transform=ax.transAxes, fontsize=9.5, color=COLORS["muted"], va="bottom",
)
credit(ax, "Opta via WhoScored")
fig.savefig(FIGS / "23_bayern_switch_leaderboard.png")

print("\n=== Bayern 2013/14 — top by progressive per game ===")
print(agg.sort_values("progressive_pg", ascending=False).head(8)[
    ["player", "games", "passes_pg", "progressive_pg", "switches"]
].round(1).to_string(index=False))
print("\n=== Kroos rank among Bayern squad ===")
for col in ("passes_pg", "progressive_pg", "switches_pg"):
    ranked = agg.sort_values(col, ascending=False).reset_index(drop=True)
    rank = ranked.index[ranked["player"] == SUBJECT][0] + 1
    print(f"  {col}: #{rank} of {len(agg)} ({ranked.loc[rank-1, col]:.1f})")

kroos_sw = passes[(passes["player"] == SUBJECT) & passes["switch"]]
print(f"\nKroos switches: {len(kroos_sw)} total, "
      f"{int(kroos_sw['set_piece'].sum())} from dead balls "
      f"(parallels the WC2014 open-play finding)")
print("DONE — figures 22–23 saved")
