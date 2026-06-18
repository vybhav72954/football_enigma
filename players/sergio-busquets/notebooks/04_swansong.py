"""Fig 04 — the last pass for Spain (WC 2022 R16, Morocco 0–0 Spain, pens).

Busquets's final match for his country, as an event map: every pass, the
last one highlighted. He passed at 88% to the very end, and his literal final
ball was a completed pass to Rodri — the man who would inherit the holding
role. Then the metronome's international career closed on the one act the
game almost never asked of him: a penalty, saved. The player defined by never
losing the ball, gone on the only moment of pure isolated pressure.

Run:  .venv/Scripts/python -X utf8 players/sergio-busquets/notebooks/04_swansong.py
"""

import sys

import matplotlib

matplotlib.use("Agg")
sys.stdout.reconfigure(encoding="utf-8")

from football_enigma.data import statsbomb as sb_data
from football_enigma.data.schema import statsbomb_passes
from football_enigma.data.statsbomb_ids import BUSQUETS_PLAYER_NAME
from football_enigma.utils.paths import figures_dir
from football_enigma.viz.pitch import draw, meter_pitch, pass_arrows
from football_enigma.viz.theme import COLORS, apply_theme, credit

SUBJ = BUSQUETS_PLAYER_NAME
MATCH_ID = 3869220  # Morocco - Spain, R16, 6 Dec 2022 (Spain out on pens)
FIGS = figures_dir("sergio-busquets")

apply_theme()
FIGS.mkdir(parents=True, exist_ok=True)

events = sb_data.decode_locations(sb_data.load_events(MATCH_ID))
raw = events[(events["type"] == "Pass") & (events["player"] == SUBJ)].sort_values(
    "index"
)
passes = statsbomb_passes(events.loc[raw.index], MATCH_ID)
completed, missed = passes[passes["outcome"]], passes[~passes["outcome"]]
last = passes.iloc[[-1]]
last_minute = int(last["minute"].iloc[0])

pitch = meter_pitch(pad_top=14)
fig, ax = draw(pitch, figsize=(11, 7.5))
pass_arrows(pitch, ax, completed, color=COLORS["subject"], alpha=0.32, width=1.3)
pass_arrows(pitch, ax, missed, color=COLORS["field"], alpha=0.3, width=1.1)
pass_arrows(pitch, ax, last, color=COLORS["subject"], alpha=1.0, width=3.0)
ax.annotate(
    f"the last pass — {last_minute}', completed, to Rodri",
    (float(last["end_x"].iloc[0]), float(last["end_y"].iloc[0])),
    xytext=(-18, -38), textcoords="offset points", ha="right",
    fontsize=11, fontweight="bold", color=COLORS["subject"],
    arrowprops=dict(arrowstyle="-", color=COLORS["grid"], lw=0.8),
)

fig.subplots_adjust(top=0.8)
fig.text(0.06, 0.95, "The last pass for Spain — Morocco 0–0 Spain, World Cup "
         "2022 (last 16)", color=COLORS["text"], fontsize=15,
         fontweight="bold", ha="left", va="top")
fig.text(0.06, 0.905,
         f"every Busquets pass in his final international: {len(completed)} "
         f"completed (amber), {len(missed)} incomplete (grey), "
         f"{len(completed) / len(passes) * 100:.0f}% accuracy",
         fontsize=10, color=COLORS["muted"], ha="left", va="top")
fig.text(0.06, 0.87,
         "Spain went out on penalties; Busquets's was saved — the man who "
         "never lost the ball, gone on the one act of pure pressure",
         fontsize=9.5, color=COLORS["muted"], ha="left", va="top", style="italic")
credit(ax, "StatsBomb")
fig.savefig(FIGS / "04_swansong.png")

print(f"{len(passes)} passes, {len(completed)} completed "
      f"({len(completed) / len(passes) * 100:.1f}%)")
print(f"last pass: minute {last_minute}, completed="
      f"{bool(last['outcome'].iloc[0])}")
print("DONE — figure 04 saved")
