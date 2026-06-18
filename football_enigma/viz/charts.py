"""Signature chart shapes for the series.

The recurring visual argument of Football Enigma is one player against the
crowd: a grey field of peers and a single amber outlier. `crowd_scatter`
is that chart; everything else is variation.
"""

import warnings

import matplotlib.pyplot as plt
import pandas as pd

from football_enigma.viz.theme import COLORS, credit, highlight_point

_NAME_OVERRIDES = {
    "Bruno Miguel Borges Fernandes": "Bruno Fernandes",
    "Lamine Yamal Nasraoui Ebana": "Lamine Yamal",
    "Marcelo Vieira da Silva Júnior": "Marcelo",
    "Carlos Henrique Casimiro": "Casemiro",
    "Kléper Laveran Lima Ferreira": "Pepe",
    "Rodrigo Hernández Cascante": "Rodri",
    "Sergio Busquets i Burgos": "Sergio Busquets",
    "Andrés Iniesta Luján": "Andrés Iniesta",
    "Francisco Román Alarcón Suárez": "Isco",
    "Gerard Piqué Bernabéu": "Gerard Piqué",
    "Javier Alejandro Mascherano": "Javier Mascherano",
}


def _short_name(name: str) -> str:
    return _NAME_OVERRIDES.get(name, name)


def crowd_scatter(
    data: pd.DataFrame,
    x: str,
    y: str,
    subject: str,
    player_col: str = "player",
    title: str = "",
    subtitle: str = "",
    xlabel: str = "",
    ylabel: str = "",
    source: str = "StatsBomb",
    annotate: list[str] | None = None,
    figsize: tuple[float, float] = (10, 7),
) -> tuple[plt.Figure, plt.Axes]:
    """Grey crowd, amber subject, optional named comparison points."""
    fig, ax = plt.subplots(figsize=figsize)
    crowd = data[data[player_col] != subject]
    ax.scatter(
        crowd[x], crowd[y], s=42, color=COLORS["field"], alpha=0.55,
        edgecolors="none",
    )
    texts = []
    for name in annotate or []:
        row = data[data[player_col] == name]
        if row.empty:
            # a requested comparison point isn't in the data (wrong name string,
            # or filtered out upstream) — surface it instead of silently dropping
            warnings.warn(
                f"crowd_scatter: annotate name {name!r} not found in data; "
                "annotation skipped",
                stacklevel=2,
            )
            continue
        px, py = float(row[x].iloc[0]), float(row[y].iloc[0])
        texts.append(
            ax.text(px, py, _short_name(name), fontsize=9, color=COLORS["muted"])
        )
        ax.scatter([px], [py], s=60, color=COLORS["accent2"], alpha=0.9,
                   edgecolors="none", zorder=4)
    if texts:
        from adjustText import adjust_text

        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color=COLORS["grid"], lw=0.8),
        )

    row = data[data[player_col] == subject]
    if not row.empty:
        highlight_point(
            ax, float(row[x].iloc[0]), float(row[y].iloc[0]), _short_name(subject)
        )

    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title, loc="left", pad=30 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.025, subtitle, transform=ax.transAxes,
                fontsize=10, color=COLORS["muted"], va="bottom")
    credit(ax, source)
    return fig, ax
