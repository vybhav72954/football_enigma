"""WC 2014, tournament-wide: the three charts the video built by hand,
rebuilt from full Opta event data.

Fig 09 — passes per game vs progressive passes per game, players with 4+
         matches ("only one guy top right"; the De Rossi noise problem is
         solved by the 4-match filter, exactly as the video did)
Fig 10 — measuring silence at WC 2014 (median vs mean involvement gap)
Fig 11 — switch leaderboard (the "Gago chart"): top 10 by switches

Run after the full scrape:
  .venv/Scripts/python -X utf8 players/toni-kroos/notebooks/07_wc2014_tournament.py
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
from football_enigma.metrics.cadence import cadence_summary
from football_enigma.utils.paths import PROCESSED_DIR, RAW_DIR, figures_dir
from football_enigma.viz.charts import crowd_scatter
from football_enigma.viz.theme import COLORS, apply_theme, credit

SUBJECT = "Toni Kroos"
MIN_MATCHES = 4
FIGS = figures_dir("toni-kroos")

apply_theme()

files = glob.glob(str(RAW_DIR / "whoscored" / "events" / "*.parquet"))
raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
# the events cache now also holds the Bayern 2013/14 scrape; isolate WC 2014
raw = raw[raw["league"] == "INT-World Cup"]
print(f"{raw['game_id'].nunique()} World Cup 2014 matches")

passes = opta_passes(raw)
passes["progressive"] = flag_progressive(passes)
passes["switch"] = flag_switch(passes)

games_played = (
    passes.groupby("player", observed=True)["match_id"].nunique().rename("games")
)
agg = (
    passes.groupby("player", observed=True)
    .agg(
        passes=("outcome", "size"),
        comp_pct=("outcome", lambda s: s.mean() * 100),
        progressive=("progressive", "sum"),
        switches=("switch", "sum"),
    )
    .join(games_played)
)
agg = agg[agg["games"] >= MIN_MATCHES].reset_index()
agg["passes_pg"] = agg["passes"] / agg["games"]
agg["progressive_pg"] = agg["progressive"] / agg["games"]
agg.to_parquet(PROCESSED_DIR / "wc2014_player_agg.parquet")

# --------------------------------------------------------------- fig 9
fig, ax = crowd_scatter(
    agg,
    x="passes_pg",
    y="progressive_pg",
    subject=SUBJECT,
    title="World Cup 2014 — the sideways-passer myth, tested",
    subtitle=f"all players with {MIN_MATCHES}+ matches | volume vs forward aggression",
    xlabel="Passes per game",
    ylabel="Progressive passes per game",
    # Pirlo/Xavi (group-stage exits) fall below the 4-match filter; annotate
    # deep playmakers who actually qualify
    annotate=["Philipp Lahm", "Bastian Schweinsteiger", "Javier Mascherano"],
    source="Opta via WhoScored",
)
fig.savefig(FIGS / "09_wc2014_volume_vs_progressive.png")

# -------------------------------------------------------------- fig 10
# cadence is gaps between on-ball involvements: filter to Opta touch events
# (is_touch), not every logged event — fouls, aerials and offsides aren't
# involvements and would shrink the gaps (matches _tournaments._wc2014)
involved = raw[raw["is_touch"]].rename(columns={"game_id": "match_id"})
involved = involved.merge(games_played.reset_index(), on="player")
involved = involved[involved["games"] >= MIN_MATCHES]
cadence = cadence_summary(involved, min_involvements=150)
cadence.to_parquet(PROCESSED_DIR / "wc2014_cadence.parquet")

fig, ax = crowd_scatter(
    cadence,
    x="median_gap",
    y="mean_gap",
    subject=SUBJECT,
    title="Measuring silence — World Cup 2014",
    subtitle="median vs mean gap between on-ball actions (s); 4+ matches, 150+ involvements",
    xlabel="Median gap (s) — usual involvement rhythm",
    ylabel="Mean gap (s) — punished by disappearances",
    # Klose drops below the touch threshold under is_touch; van Persie is the
    # striker-up-right example (rare, decisive touches), Sneijder the metronome
    annotate=["Lionel Messi", "Wesley Sneijder", "Robin van Persie"],
    source="Opta via WhoScored",
)
fig.savefig(FIGS / "10_wc2014_measuring_silence.png")

# -------------------------------------------------------------- fig 11
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
ax.set_title("Who stretched the World Cup", loc="left", pad=20)
ax.text(
    0, 1.02, "World Cup 2014, whole tournament | players with 4+ matches",
    transform=ax.transAxes, fontsize=9.5, color=COLORS["muted"], va="bottom",
)
credit(ax, "Opta via WhoScored")
fig.savefig(FIGS / "11_wc2014_switches_leaderboard.png")

print("\n=== top by progressive per game ===")
print(agg.sort_values("progressive_pg", ascending=False).head(8)[
    ["player", "games", "passes_pg", "progressive_pg", "switches"]
].round(1).to_string(index=False))
print("\n=== cadence best (lowest mean gap) ===")
print(cadence.sort_values("mean_gap").head(8).round(1).to_string(index=False))
print("\nDONE — figures 09–11 saved")
