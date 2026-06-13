"""Euro 2024: time on ball — the tempo instrument.

For every completed receipt, the seconds until that player's next pass in
the same possession (metrics.tempo). The median is a player's tempo
signature; pairing it with how often he gets the ball separates the
metronome (constant receipts, instant release) from the ball-stopper.

Fig 18 — receipts per 90 vs median time on ball, whole tournament.

Run:  .venv/Scripts/python -X utf8 players/toni-kroos/notebooks/12_tempo.py
"""

import sys

import matplotlib

matplotlib.use("Agg")
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from football_enigma.data import statsbomb as sb_data
from football_enigma.data.build import player_minutes
from football_enigma.data.statsbomb_ids import EURO_2024
from football_enigma.metrics.tempo import receipt_to_release, tempo_summary
from football_enigma.utils.paths import PROCESSED_DIR, figures_dir
from football_enigma.viz.charts import crowd_scatter
from football_enigma.viz.theme import apply_theme

SUBJECT = "Toni Kroos"
MIN_MINUTES = 270
MIN_RECEIPTS = 100
FIGS = figures_dir("toni-kroos")

apply_theme()

matches = sb_data.load_matches(EURO_2024)
holds = pd.concat(
    [
        receipt_to_release(sb_data.load_events(int(mid)), int(mid))
        for mid in matches["match_id"]
    ],
    ignore_index=True,
)
tempo = tempo_summary(holds, min_receipts=MIN_RECEIPTS)

minutes = player_minutes(EURO_2024)
tempo = tempo.merge(minutes.rename("minutes"), left_on="player",
                    right_index=True)
tempo = tempo[tempo["minutes"] >= MIN_MINUTES].copy()
tempo["receipts_p90"] = tempo["receipts"] / (tempo["minutes"] / 90)
tempo.to_parquet(PROCESSED_DIR / "euro2024_tempo.parquet")

fig, ax = crowd_scatter(
    tempo,
    x="median_hold",
    y="receipts_p90",
    subject=SUBJECT,
    title="Tempo — who gives the ball back fastest",
    subtitle="Euro 2024, 270+ min & 100+ receipts | median seconds from receiving to releasing "
    "(passes in the same possession) vs how often the ball arrives",
    xlabel="Median time on ball (s) — left = faster release",
    ylabel="Receipts resolved into a pass, per 90",
    annotate=["Rodrigo Hernández Cascante", "Granit Xhaka",
              "Jamal Musiala", "Lamine Yamal Nasraoui Ebana"],
)
fig.savefig(FIGS / "18_tempo.png")

print("=== fastest median release (qualified) ===")
print(tempo.sort_values("median_hold").head(10)[
    ["player", "median_hold", "mean_hold", "receipts", "receipts_p90"]
].round(2).to_string(index=False))
kroos = tempo[tempo["player"] == SUBJECT]
print("\nKroos:", kroos.round(2).to_string(index=False))
print("DONE — figure 18 saved")
