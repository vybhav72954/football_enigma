"""Regenerate every Busquets figure from cached data, in order.

Usage:  .venv/Scripts/python -X utf8 players/sergio-busquets/run_all.py
Assumes the StatsBomb cache is populated (the _pull_*.py helpers warm it).
"""

import subprocess
import sys
from pathlib import Path

NOTEBOOKS = [
    "01_the_constant.py",    # invariance: 13 seasons, one number
    "02_space.py",           # plays where the pressure isn't (360)
    "03_radar.py",           # fundamentals maxed, glamour quiet
    "04_swansong.py",        # the last pass for Spain (WC 2022)
    "05_never_first.py",     # always near the top, never the top
    "06_first_receiver.py",  # the outlet the defenders look for
    "07_territory.py",       # two men, one midfield (heatmap pair)
]

here = Path(__file__).parent / "notebooks"
for nb in NOTEBOOKS:
    print(f"--- {nb}")
    result = subprocess.run(
        [sys.executable, "-X", "utf8", str(here / nb)],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        sys.exit(f"{nb} failed")
print("ALL FIGURES REGENERATED")
