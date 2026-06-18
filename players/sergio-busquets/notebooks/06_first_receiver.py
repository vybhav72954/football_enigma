"""Fig 06 — the outlet the defenders look for.

For every completed pass in La Liga 2015/16 we ask whether it left a team's own
back line (a goalkeeper or centre-back passing to a non-defender) and credit the
receiver. Busquets is the 4th-most-trusted such outlet in the whole league — and
the three players above him are fullbacks (Alba, Alves, Marcelo), who stand right
next to the defenders. Among central organisers only one player matches him:
Toni Kroos, the subject of Post #1, sitting on the same spot. The release valve
is a role with no column in the box score; it is also the role Barcelona built
their decade around.

Honesty: this is "near the top, not the top" again (the fullbacks lead on raw
volume, and his *share* of receptions from the back line ranks only 17th — he
receives everywhere, not just from defenders). The finding is the company he
keeps, not a solo record.

Run:  .venv/Scripts/python -X utf8 players/sergio-busquets/notebooks/06_first_receiver.py
"""

import sys

import matplotlib

matplotlib.use("Agg")
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from football_enigma.data.build import (
    build_backline_receptions,
    player_minutes,
)
from football_enigma.data.statsbomb_ids import BUSQUETS_PLAYER_NAME, LA_LIGA
from football_enigma.utils.paths import figures_dir
from football_enigma.viz.charts import crowd_scatter

SUBJ = BUSQUETS_PLAYER_NAME
COMP = LA_LIGA["2015/2016"]
FIGS = figures_dir("sergio-busquets")
MIN_MINUTES = 1500

bu = build_backline_receptions(COMP, "laliga1516").set_index("player")
bu = bu.join(player_minutes(COMP).rename("minutes"), how="inner")
bu = bu[bu["minutes"] >= MIN_MINUTES].copy()
per90 = bu["minutes"] / 90
bu["from_backline_p90"] = bu["from_backline"] / per90
bu["backline_share"] = bu["from_backline"] / bu["total_received"].clip(lower=1) * 100
bu = bu[bu["from_backline_p90"] >= 4].reset_index()  # players the back line uses at all

fig, ax = crowd_scatter(
    bu,
    x="from_backline_p90",
    y="backline_share",
    subject=SUBJ,
    title="The outlet the defenders look for",
    subtitle="receptions out of a team's own back line, La Liga 2015/16 "
             "(1500+ min)",
    xlabel="Passes received from own GK/centre-backs per 90  (right = bigger outlet)",
    ylabel="Share of all his receptions from the back line (%)",
    annotate=["Toni Kroos", "Jordi Alba Ramos", "Andrés Iniesta Luján"],
    source="StatsBomb",
)
ax.set_xlim(3.5, 22)  # room for the subject label near the right edge
fig.savefig(FIGS / "06_first_receiver.png")

top = bu.sort_values("from_backline_p90", ascending=False).head(8)
print("=== most-trusted outlets out of the back line (per 90) ===")
print(top[["player", "from_backline_p90", "backline_share"]].round(1).to_string(index=False))
b = bu[bu["player"] == SUBJ].iloc[0]
rank = int((bu["from_backline_p90"] > b["from_backline_p90"]).sum() + 1)
print(f"\nBusquets: {b['from_backline_p90']:.1f}/90 from the back line "
      f"(rank {rank}/{len(bu)}), {b['backline_share']:.0f}% of his receptions")
print("DONE — figure 06 saved")
