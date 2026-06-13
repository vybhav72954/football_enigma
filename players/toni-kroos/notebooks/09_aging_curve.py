"""The aging curve: same instruments, same player, ten years apart.

Kroos's four international tournaments measured with identical
definitions — WC 2014 (age 24) to Euro 2024 (age 34).

Fig 14 — raw per-90: pass volume, progressive passes, switches, and
         involvement cadence (median spell gap) across the decade.
Fig 15 — possession-adjusted versions (per 100 Germany passes while he
         was on the pitch): the answer to "his numbers are just German
         possession dominance".

Run:  .venv/Scripts/python -X utf8 players/toni-kroos/notebooks/09_aging_curve.py
"""

import sys

import matplotlib

matplotlib.use("Agg")
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt

from _tournaments import TOURNAMENT_ORDER, load, tournament_summary
from football_enigma.utils.paths import PROCESSED_DIR, figures_dir
from football_enigma.viz.theme import COLORS, apply_theme, credit

FIGS = figures_dir("toni-kroos")
SOURCES = "Opta via WhoScored (2014); StatsBomb (2018–2024)"

apply_theme()

matches, gaps = load()
summary = tournament_summary(matches, gaps)
summary.to_parquet(PROCESSED_DIR / "kroos_tournament_summary.parquet")


def aging_panels(panels, title, subtitle, filename):
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.6))
    x = range(len(TOURNAMENT_ORDER))
    for ax, (col, label, fmt) in zip(axes, panels):
        vals = summary[col]
        ax.plot(x, vals, color=COLORS["subject"], lw=2.4, zorder=3,
                marker="o", markersize=8, markeredgecolor=COLORS["text"],
                markeredgewidth=1.0)
        for xi, v in zip(x, vals):
            ax.annotate(fmt.format(v), (xi, v), xytext=(0, 11),
                        textcoords="offset points", ha="center", fontsize=10,
                        fontweight="bold", color=COLORS["text"])
        ax.set_xticks(list(x), [t.replace(" ", "\n") for t in TOURNAMENT_ORDER])
        ax.set_title(label, loc="left", fontsize=11.5, pad=8)
        ax.set_ylim(0, vals.max() * 1.25)
        ax.grid(axis="x", visible=False)
    fig.suptitle(title, x=0.025, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.025, 0.905, subtitle, fontsize=9.5, color=COLORS["muted"])
    credit(axes[0], SOURCES)
    fig.tight_layout(rect=(0, 0.02, 1, 0.88))
    fig.savefig(FIGS / filename)
    return fig


aging_panels(
    [
        ("passes_p90", "Passes per 90", "{:.0f}"),
        ("progressive_p90", "Progressive passes per 90", "{:.1f}"),
        ("switches_p90", "Switches per 90", "{:.1f}"),
        ("median_spell_gap", "Median gap between involvements (s)", "{:.0f}"),
    ],
    "Ten years, one profile — Kroos at four tournaments",
    "Germany matches only, identical metric definitions | cadence merges events <5 s apart "
    "into one involvement, so Opta and StatsBomb event logs compare fairly",
    "14_aging_curve.png",
)

aging_panels(
    [
        ("passes_per100", "Share of Germany passes (%)", "{:.1f}"),
        ("progressive_per100", "Progressive per 100 team passes", "{:.2f}"),
        ("switches_per100", "Switches per 100 team passes", "{:.2f}"),
    ],
    "Not a possession artefact — the same curve, possession-adjusted",
    "per 100 Germany passes attempted while Kroos was on the pitch | "
    "controls for how much ball each Germany side had",
    "15_aging_curve_possession_adjusted.png",
)

print(summary.round(2).to_string(index=False))
print("\nDONE — figures 14–15 saved")
