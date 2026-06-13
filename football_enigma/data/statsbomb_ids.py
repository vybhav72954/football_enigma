"""StatsBomb open-data competition/season IDs relevant to the Kroos analysis.

Verified against statsbomb/open-data competitions.json on 2026-06-12.
has_360 marks competitions with 360 freeze-frame data (defender positions
at event moments) — required for packing/line-breaking metrics.
"""

from typing import NamedTuple


class CompSeason(NamedTuple):
    competition_id: int
    season_id: int
    has_360: bool


EURO_2024 = CompSeason(55, 282, True)
EURO_2020 = CompSeason(55, 43, True)
WORLD_CUP_2018 = CompSeason(43, 3, False)

CL_2015_16 = CompSeason(16, 27, False)  # final: Real Madrid - Atlético
CL_2016_17 = CompSeason(16, 2, False)  # final: Juventus - Real Madrid
CL_2017_18 = CompSeason(16, 1, False)  # final: Real Madrid - Liverpool

# StatsBomb open data: 2015/2016 is the COMPLETE 380-match La Liga season
# (all 20 teams, Real Madrid included) — Kroos has full-season minutes here,
# which is why it anchors the prime-Madrid analysis. Other seasons listed are
# largely Barcelona's matches (= El Clásico only, for Kroos).
LA_LIGA = {
    "2014/2015": CompSeason(11, 26, False),
    "2015/2016": CompSeason(11, 27, False),
    "2016/2017": CompSeason(11, 2, False),
    "2017/2018": CompSeason(11, 1, False),
    "2018/2019": CompSeason(11, 4, False),
    "2019/2020": CompSeason(11, 42, False),
    "2020/2021": CompSeason(11, 90, True),
}

KROOS_PLAYER_NAME = "Toni Kroos"
