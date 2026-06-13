"""Regenerate every Kroos figure from cached data, in order.

Usage:  .venv/Scripts/python -X utf8 players/toni-kroos/run_all.py
Assumes data/raw is populated (loaders fetch anything missing).
"""

import subprocess
import sys
from pathlib import Path

NOTEBOOKS = [
    "01_euro2024.py",
    "02_cadence_quarterback.py",
    "03_wc2014_switches.py",
    "04_cl_finals.py",
    "05_packing_figure.py",
    "06_radar.py",
    "07_wc2014_tournament.py",
    "08_laliga1516.py",
    "09_aging_curve.py",
    "10_wc2018_stress_test.py",
    "11_switch_anatomy.py",
    "12_tempo.py",
    "13_last_pass.py",
    "14_pass_network.py",
    "15_money_context.py",
    "16_bayern_act1.py",
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
