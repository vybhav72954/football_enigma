"""Fig 01 (headline) — the constant who never cracks.

Busquets's completion under pressure across THIRTEEN Barca seasons
(2008/09-2020/21, age 20-32). After a two-season rookie ramp it locks at
~85% and never moves — through Xavi, Henry, Eto'o, Villa, Neymar,
Mascherano, Iniesta and Suarez all leaving, and four-plus managers. The
team was rebuilt around him again and again; the number stayed put. You
cannot build a leaderboard out of a constant.

Run:  .venv/Scripts/python -X utf8 players/sergio-busquets/notebooks/01_the_constant.py
"""

import sys

import matplotlib

matplotlib.use("Agg")
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from football_enigma.data import statsbomb as sb_data
from football_enigma.data.schema import statsbomb_passes
from football_enigma.data.statsbomb import decode_locations
from football_enigma.data.statsbomb_ids import BUSQUETS_PLAYER_NAME, LA_LIGA
from football_enigma.metrics.pressure import pressure_split
from football_enigma.utils.paths import PROCESSED_DIR, figures_dir
from football_enigma.viz.theme import COLORS, apply_theme, credit

SUBJ = BUSQUETS_PLAYER_NAME
FIGS = figures_dir("sergio-busquets")
SEASONS = ["2008/2009", "2009/2010", "2010/2011", "2011/2012", "2012/2013",
           "2013/2014", "2014/2015", "2015/2016", "2016/2017", "2017/2018",
           "2018/2019", "2019/2020", "2020/2021"]

# greats who left, plotted between the season they last appear and the next
DEPARTURES = {
    "2014/2015": "Xavi leaves",
    "2016/2017": "Neymar leaves",
    "2017/2018": "Iniesta &\nMascherano leave",
    "2019/2020": "Suárez leaves",
}

apply_theme()
FIGS.mkdir(parents=True, exist_ok=True)

rows = []
for s in SEASONS:
    m = sb_data.load_matches(LA_LIGA[s])
    barca = m[(m["home_team"] == "Barcelona") | (m["away_team"] == "Barcelona")]
    bp = []
    for mid in barca["match_id"].astype(int):
        ev = decode_locations(sb_data.load_events(mid))
        bp.append(statsbomb_passes(ev, mid))
    BP = pd.concat(bp, ignore_index=True)
    ps = pressure_split(BP, min_pressured=40)
    busq = ps[ps["player"] == SUBJ]["comp_pct_pressured"]
    rows.append({
        "season": s,
        "busquets": float(busq.iloc[0]) if len(busq) else float("nan"),
        "peer_median": float(ps["comp_pct_pressured"].median()),
        "peer_q25": float(ps["comp_pct_pressured"].quantile(0.25)),
        "peer_q75": float(ps["comp_pct_pressured"].quantile(0.75)),
        "peers": len(ps),
    })
traj = pd.DataFrame(rows)
traj.to_parquet(PROCESSED_DIR / "busquets_constant.parquet")

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(11, 6.5))
x = range(len(SEASONS))
labels = [s[2:4] + "/" + s[7:9] for s in SEASONS]  # 08/09 ...

# peer context: ball-dominant players in the same matches
ax.fill_between(x, traj["peer_q25"], traj["peer_q75"], color=COLORS["field"],
                alpha=0.18, linewidth=0, label="other ball-dominant players (same matches)")
ax.plot(x, traj["peer_median"], color=COLORS["field"], lw=1.4, alpha=0.8,
        linestyle="--", label="their median")

ax.plot(x, traj["busquets"], color=COLORS["subject"], lw=3, marker="o",
        markersize=7, markeredgecolor=COLORS["background"], markeredgewidth=1.2,
        label="Busquets", zorder=5)

# departure markers: a dotted line at each summer a great left, labelled
# vertically in the empty lower band so the labels never collide
for s, txt in DEPARTURES.items():
    xi = SEASONS.index(s) + 0.5
    ax.axvline(xi, color=COLORS["negative"], lw=1, alpha=0.4, linestyle=":")
    ax.text(xi, 61, txt.replace("\n", " "), color=COLORS["negative"],
            fontsize=8.5, ha="center", va="bottom", rotation=90, alpha=0.9)

ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_ylim(60, 100)
ax.set_ylabel("Completion % under pressure")
ax.set_xlabel("La Liga season")
ax.set_title("Thirteen seasons, one number", loc="left", pad=30)
ax.text(0, 1.025,
        "Busquets's completion under pressure, 2008/09–2020/21 (age 20–32)  |  "
        "the team kept changing; he didn't",
        transform=ax.transAxes, fontsize=10, color=COLORS["muted"], va="bottom")
ax.legend(loc="lower left", framealpha=0, fontsize=9, labelcolor=COLORS["muted"])
credit(ax, "StatsBomb")
fig.savefig(FIGS / "01_the_constant.png")

print(traj.round(1).to_string(index=False))
locked = traj[traj["season"] >= "2010/2011"]["busquets"]
print(f"\nLocked stretch 2010/11–2020/21: min {locked.min():.1f}  max "
      f"{locked.max():.1f}  range {locked.max()-locked.min():.1f}pp over "
      f"{len(locked)} seasons")
print("DONE — figure 01 saved")
