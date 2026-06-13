"""The last pass — Euro 2024 quarter-final, Germany 1–2 Spain (AET).

Kroos's final professional match, as an event map: every pass he played,
with the literal last pass of his career highlighted. The emotional close
of the post.

Run:  .venv/Scripts/python -X utf8 players/toni-kroos/notebooks/13_last_pass.py
"""

import sys

import matplotlib

matplotlib.use("Agg")
sys.stdout.reconfigure(encoding="utf-8")

from football_enigma.data import statsbomb as sb_data
from football_enigma.data.schema import statsbomb_passes
from football_enigma.utils.paths import figures_dir
from football_enigma.viz.pitch import draw, meter_pitch, pass_arrows
from football_enigma.viz.theme import COLORS, apply_theme, credit

SUBJECT = "Toni Kroos"
MATCH_ID = 3942226  # Spain - Germany, Stuttgart, 5 July 2024
FIGS = figures_dir("toni-kroos")

apply_theme()

events = sb_data.decode_locations(sb_data.load_events(MATCH_ID))
# StatsBomb event `index` is match chronology; the last pass row is the
# last pass of his career
kroos_raw = events[
    (events["type"] == "Pass") & (events["player"] == SUBJECT)
].sort_values("index")
final = kroos_raw.iloc[-1]
if isinstance(final["pass_recipient"], str):
    last_desc = f"to {final['pass_recipient']}"
else:
    restart = (
        f"a {final['pass_type'].lower()} "
        if isinstance(final["pass_type"], str) else ""
    )
    last_desc = f"{restart}into the box, never arrived"

passes = statsbomb_passes(events.loc[kroos_raw.index], MATCH_ID)
completed, missed = passes[passes["outcome"]], passes[~passes["outcome"]]
last = passes.iloc[[-1]]
last_minute = int(last["minute"].iloc[0])

pitch = meter_pitch(pad_top=14)
fig, ax = draw(pitch, figsize=(11, 7.5))
pass_arrows(pitch, ax, completed, color=COLORS["subject"], alpha=0.35,
            width=1.3)
pass_arrows(pitch, ax, missed, color=COLORS["field"], alpha=0.3, width=1.1)
pass_arrows(pitch, ax, last, color=COLORS["subject"], alpha=1.0, width=3.0)
ax.annotate(
    f"the last pass — {last_minute}', {last_desc}",
    (float(last["end_x"].iloc[0]), float(last["end_y"].iloc[0])),
    xytext=(-18, -36), textcoords="offset points", ha="right",
    fontsize=11, fontweight="bold", color=COLORS["subject"],
    arrowprops=dict(arrowstyle="-", color=COLORS["grid"], lw=0.8),
)

fig.subplots_adjust(top=0.82)
fig.text(
    0.06, 0.95, "The last pass — Germany 1–2 Spain, Euro 2024 quarter-final",
    color=COLORS["text"], fontsize=15, fontweight="bold", ha="left", va="top",
)
fig.text(
    0.06, 0.905,
    f"every Kroos pass in his final professional match: {len(completed)} completed (amber), "
    f"{len(missed)} incomplete (grey) | Stuttgart, 5 July 2024",
    fontsize=10, color=COLORS["muted"], ha="left", va="top",
)
credit(ax, "StatsBomb")
fig.savefig(FIGS / "19_last_pass.png")

print(f"{len(passes)} passes, {len(completed)} completed "
      f"({len(completed) / len(passes) * 100:.1f}%)")
print(f"last pass: minute {last_minute}, {last_desc}, "
      f"completed={bool(last['outcome'].iloc[0])}")
last_completed = kroos_raw[kroos_raw["pass_outcome"].isna()].iloc[-1]
print(f"last completed pass: minute {int(last_completed['minute'])}, "
      f"to {last_completed['pass_recipient']}")
print("DONE — figure 19 saved")
