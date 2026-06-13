"""The price of a metronome — Kroos's €25m against the midfield market.

The one figure in this series not built from event data: public, reported
transfer fees (Transfermarkt). The argument is deliberately conservative —
the like-for-like case is Kroos vs James Rodríguez, signed by the *same
club in the same summer* for 3x the fee, which controls for transfer
inflation. The later signings carry their year on the bar because fees
rose across the board; they show the deep-midfield role being repriced,
not a clean Kroos-only comparison.

Fig 21 — fees sorted high to low; the two 2014 Real Madrid signings land
together at the floor, James at €75m directly above Kroos at €25m.

Source: players/toni-kroos/source/transfer_fees.csv (reported figures).

Run:  .venv/Scripts/python -X utf8 players/toni-kroos/notebooks/15_money_context.py
"""

import sys

import matplotlib

matplotlib.use("Agg")
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import pandas as pd

from football_enigma.utils.paths import PLAYERS_DIR, figures_dir
from football_enigma.viz.theme import COLORS, apply_theme, credit

SUBJECT = "Toni Kroos"
ANCHOR = "James Rodriguez"  # same club, same summer — the honest contrast
FIGS = figures_dir("toni-kroos")

apply_theme()

fees = pd.read_csv(
    PLAYERS_DIR / "toni-kroos" / "source" / "transfer_fees.csv", comment="#"
)
fees = fees.sort_values("fee_m", ascending=False).reset_index(drop=True)
order = fees.iloc[::-1]  # barh plots bottom-up; cheapest (Kroos) at the floor


def bar_color(player):
    if player == SUBJECT:
        return COLORS["subject"]
    if player == ANCHOR:
        return COLORS["accent2"]
    return COLORS["field"]


fig, ax = plt.subplots(figsize=(10, 7))
colors = [bar_color(p) for p in order["player"]]
labels = [
    p.replace("James Rodriguez", "James Rodríguez")
     .replace("Aurelien Tchouameni", "Aurélien Tchouaméni")
     .replace("Moises Caicedo", "Moisés Caicedo")
    for p in order["player"]
]
bars = ax.barh(labels, order["fee_m"], color=colors, height=0.66)
for bar, (_, row) in zip(bars, order.iterrows()):
    emphasis = row["player"] in (SUBJECT, ANCHOR)
    ax.text(
        bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
        f"€{int(row['fee_m'])}m  ’{str(row['year'])[2:]}",
        va="center", fontsize=10,
        color=COLORS["text"] if emphasis else COLORS["muted"],
        fontweight="bold" if emphasis else "normal",
    )

ax.set_xlim(0, fees["fee_m"].max() * 1.15)
ax.set_xlabel("Reported transfer fee (€m)")
ax.grid(axis="y", visible=False)
ax.set_title("Priced like a departure, not a cornerstone", loc="left", pad=30)
ax.text(
    0, 1.025,
    "Real Madrid paid 3× more for James Rodríguez the same summer | later "
    "fees carry their year — the midfield market rose for everyone",
    transform=ax.transAxes, fontsize=9.5, color=COLORS["muted"], va="bottom",
)

# bracket the two 2014 Madrid signings — the controlled comparison
kroos_y = labels.index("Toni Kroos")
james_y = labels.index("James Rodríguez")
ax.annotate(
    "same club,\nsame summer,\n3× the fee",
    xy=(75, (kroos_y + james_y) / 2), xytext=(96, (kroos_y + james_y) / 2),
    fontsize=9.5, color=COLORS["subject"], va="center", ha="left",
    fontweight="bold",
)

credit(ax, "Reported fees, Transfermarkt")
fig.savefig(FIGS / "21_money_context.png")

print(fees[["player", "year", "fee_m", "from_club", "to_club"]].to_string(index=False))
print(f"\nKroos €25m vs James €75m — same club, same summer, 3.0× the fee.")
print("DONE — figure 21 saved")
