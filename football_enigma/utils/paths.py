"""Repo-anchored paths so notebooks and scripts resolve the same locations
regardless of their own working directory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PLAYERS_DIR = ROOT / "players"


def player_dir(slug: str) -> Path:
    return PLAYERS_DIR / slug


def figures_dir(slug: str) -> Path:
    return PLAYERS_DIR / slug / "figures"
