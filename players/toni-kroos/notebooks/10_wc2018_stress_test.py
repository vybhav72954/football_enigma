"""WC 2018: the stress test. Germany went out in the group stage — the
worst German World Cup in 80 years. If the Kroos profile were a team-
dominance artefact, it should collapse here, in the failing team.

Fig 16 — every Kroos tournament match as one dot, four metrics; the three
WC 2018 matches highlighted with their scorelines. The metrics hold (and
the possession-adjusted panel answers "Germany chased games, so he saw
more ball": his progressive share of team passes was the highest of his
four tournaments).

Honesty note (printed + subtitle): three group matches is a thin sample,
and that is the point — it is all the 2018 team gave him.

Run:  .venv/Scripts/python -X utf8 players/toni-kroos/notebooks/10_wc2018_stress_test.py
"""

import sys

import matplotlib

matplotlib.use("Agg")
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import numpy as np

from _tournaments import TOURNAMENT_ORDER, load
from football_enigma.utils.paths import figures_dir
from football_enigma.viz.theme import COLORS, apply_theme, credit

FIGS = figures_dir("toni-kroos")
STRESS = "WC 2018"

apply_theme()

matches, _ = load()
per90 = matches["minutes"] / 90
matches["progressive_p90"] = matches["progressive"] / per90
matches["switches_p90"] = matches["switches"] / per90
matches["comp_pct"] = matches["completed"] / matches["passes"] * 100
matches["progressive_per100"] = (
    matches["progressive"] / matches["team_passes_on"] * 100
)

PANELS = [
    ("progressive_p90", "Progressive passes per 90"),
    ("switches_p90", "Switches per 90"),
    ("comp_pct", "Completion %"),
    ("progressive_per100", "Progressive per 100 team passes"),
]

rng = np.random.default_rng(7)
matches["xpos"] = (
    matches["tournament"].map(TOURNAMENT_ORDER.index)
    + rng.uniform(-0.12, 0.12, len(matches))
)

fig, axes = plt.subplots(1, len(PANELS), figsize=(4.2 * len(PANELS), 4.8))
for ax, (col, label) in zip(axes, PANELS):
    crowd = matches[matches["tournament"] != STRESS]
    stress = matches[matches["tournament"] == STRESS]
    ax.scatter(crowd["xpos"], crowd[col], s=46, color=COLORS["field"],
               alpha=0.6, edgecolors="none")
    ax.scatter(stress["xpos"], stress[col], s=90, color=COLORS["subject"],
               zorder=5, edgecolors=COLORS["text"], linewidths=1.0)
    # tournament medians as quiet dashes
    med = matches.groupby("tournament", observed=True)[col].median()
    for i, t in enumerate(TOURNAMENT_ORDER):
        ax.hlines(med[t], i - 0.2, i + 0.2, color=COLORS["muted"],
                  lw=1.4, alpha=0.8, zorder=2)
    if col == "progressive_p90":
        # stagger labels so near-equal values don't overlap
        for offset, (_, row) in zip(
            [(8, 5), (8, -11), (8, -3)],
            stress.sort_values(col, ascending=False).iterrows(),
        ):
            ax.annotate(
                f"{row['result']} {row['opponent']}",
                (row["xpos"], row[col]), xytext=offset,
                textcoords="offset points", fontsize=8.5,
                color=COLORS["subject"],
            )
    ax.set_xticks(range(4), [t.replace(" ", "\n") for t in TOURNAMENT_ORDER])
    ax.set_xlim(-0.5, 3.5)
    ax.set_title(label, loc="left", fontsize=11.5, pad=8)
    ax.grid(axis="x", visible=False)

fig.suptitle("The team failed. The signal didn't — WC 2018 as stress test",
             x=0.025, ha="left", fontsize=15, fontweight="bold")
fig.text(
    0.025, 0.9,
    "one dot per Kroos tournament match, dash = tournament median | amber: the three matches of "
    "Germany's group-stage exit — a thin sample, but every match the 2018 team gave him",
    fontsize=9.5, color=COLORS["muted"],
)
credit(axes[0], "Opta via WhoScored (2014); StatsBomb (2018–2024)")
fig.tight_layout(rect=(0, 0.02, 1, 0.87))
fig.savefig(FIGS / "16_wc2018_stress_test.png")

print(matches[matches["tournament"] == STRESS][
    ["opponent", "result", "minutes", "passes", "comp_pct",
     "progressive_p90", "switches_p90", "progressive_per100"]
].round(2).to_string(index=False))
print("\nCaveats: 3 matches, group stage only; possession inflated by"
      " chasing games — which the per-100-team-passes panel controls for.")
print("DONE — figure 16 saved")
