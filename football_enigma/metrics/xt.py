"""Expected Threat (xT) — grid model after Karun Singh (2018).

The pitch is divided into a grid of cells. For each cell we estimate, from
event data:

- shoot_prob: P(next action is a shot | ball in cell)
- move_prob:  P(next action is a move | ball in cell)  (= 1 - shoot_prob)
- goal_prob:  P(goal | shot from cell)
- transition: P(move ends in cell j | successful move starts in cell i)

The xT surface is the fixed point of

    xT(c) = shoot_prob(c) * goal_prob(c)
          + move_prob(c) * sum_j transition(c, j) * xT(j)

A successful move action (pass/carry/dribble) is then valued as
xT(end cell) - xT(start cell).

All coordinates are canonical meters on a 105x68 pitch (see data.schema).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from football_enigma.data.schema import PITCH_LENGTH, PITCH_WIDTH

MOVE_TYPES = ("pass", "carry", "dribble")


@dataclass
class XTModel:
    surface: np.ndarray  # shape (n_rows, n_cols), xT value per cell
    n_cols: int = 16
    n_rows: int = 12

    def cell_of(self, x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        col = np.clip(
            (pd.to_numeric(x, errors="coerce") / PITCH_LENGTH * self.n_cols)
            .to_numpy(),
            0,
            self.n_cols - 1e-9,
        ).astype(int)
        row = np.clip(
            (pd.to_numeric(y, errors="coerce") / PITCH_WIDTH * self.n_rows)
            .to_numpy(),
            0,
            self.n_rows - 1e-9,
        ).astype(int)
        return row, col

    def value(self, actions: pd.DataFrame) -> pd.Series:
        """xT added by each action: surface[end] - surface[start] for
        successful moves; 0 for unsuccessful or non-move actions."""
        is_move = actions["type"].isin(MOVE_TYPES) & actions["outcome"].astype(bool)
        valid = (
            is_move
            & actions[["x", "y", "end_x", "end_y"]].notna().all(axis=1)
        )
        out = pd.Series(0.0, index=actions.index, name="xt_value")
        if not valid.any():
            return out
        sub = actions[valid]
        r0, c0 = self.cell_of(sub["x"], sub["y"])
        r1, c1 = self.cell_of(sub["end_x"], sub["end_y"])
        out.loc[valid] = self.surface[r1, c1] - self.surface[r0, c0]
        return out


def fit_xt(
    actions: pd.DataFrame,
    n_cols: int = 16,
    n_rows: int = 12,
    n_iter: int = 100,
) -> XTModel:
    """Fit an xT surface from canonical actions.

    `actions` must contain moves (type in pass/carry/dribble) and shots
    (type == 'shot') with `outcome` True for completed moves / goals.
    """
    model = XTModel(surface=np.zeros((n_rows, n_cols)), n_cols=n_cols, n_rows=n_rows)

    acts = actions[actions[["x", "y"]].notna().all(axis=1)]
    moves = acts[acts["type"].isin(MOVE_TYPES)]
    shots = acts[acts["type"] == "shot"]

    move_count = np.zeros((n_rows, n_cols))
    shot_count = np.zeros((n_rows, n_cols))
    goal_count = np.zeros((n_rows, n_cols))

    r, c = model.cell_of(moves["x"], moves["y"])
    np.add.at(move_count, (r, c), 1)
    r, c = model.cell_of(shots["x"], shots["y"])
    np.add.at(shot_count, (r, c), 1)
    goals = shots[shots["outcome"].astype(bool)]
    r, c = model.cell_of(goals["x"], goals["y"])
    np.add.at(goal_count, (r, c), 1)

    total = move_count + shot_count
    with np.errstate(divide="ignore", invalid="ignore"):
        shoot_prob = np.where(total > 0, shot_count / total, 0.0)
        move_prob = np.where(total > 0, move_count / total, 0.0)
        goal_prob = np.where(shot_count > 0, goal_count / shot_count, 0.0)

    # Transition matrix from successful, fully-located moves
    ok = moves[
        moves["outcome"].astype(bool)
        & moves[["end_x", "end_y"]].notna().all(axis=1)
    ]
    n_cells = n_rows * n_cols
    transition = np.zeros((n_cells, n_cells))
    r0, c0 = model.cell_of(ok["x"], ok["y"])
    r1, c1 = model.cell_of(ok["end_x"], ok["end_y"])
    np.add.at(transition, (r0 * n_cols + c0, r1 * n_cols + c1), 1)
    row_sums = transition.sum(axis=1, keepdims=True)
    transition = np.divide(
        transition, row_sums, out=np.zeros_like(transition), where=row_sums > 0
    )

    xt = np.zeros(n_cells)
    shoot_flat = shoot_prob.ravel()
    move_flat = move_prob.ravel()
    goal_flat = goal_prob.ravel()
    for _ in range(n_iter):
        xt = shoot_flat * goal_flat + move_flat * (transition @ xt)

    model.surface = xt.reshape(n_rows, n_cols)
    return model
