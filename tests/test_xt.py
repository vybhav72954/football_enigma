import numpy as np
import pandas as pd
import pytest

from football_enigma.data.schema import PITCH_LENGTH, PITCH_WIDTH
from football_enigma.metrics.xt import MOVE_TYPES, XTModel, fit_xt


def synthetic_actions(n: int = 5000, seed: int = 7) -> pd.DataFrame:
    """Random walk of moves plus shots that get likelier near the goal."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, PITCH_LENGTH, n)
    y = rng.uniform(0, PITCH_WIDTH, n)
    end_x = np.clip(x + rng.normal(8, 15, n), 0, PITCH_LENGTH)
    end_y = np.clip(y + rng.normal(0, 10, n), 0, PITCH_WIDTH)
    shot_p = np.where(x > 0.75 * PITCH_LENGTH, 0.25, 0.01)
    is_shot = rng.uniform(size=n) < shot_p
    # goal odds rise sharply with proximity to goal, like real football
    goal = is_shot & (rng.uniform(size=n) < 0.5 * (x / PITCH_LENGTH) ** 4)
    return pd.DataFrame(
        {
            "type": np.where(is_shot, "shot", "pass"),
            "x": x,
            "y": y,
            "end_x": np.where(is_shot, np.nan, end_x),
            "end_y": np.where(is_shot, np.nan, end_y),
            "outcome": np.where(is_shot, goal, rng.uniform(size=n) < 0.7),
        }
    )


@pytest.fixture(scope="module")
def model() -> XTModel:
    return fit_xt(synthetic_actions())


def test_surface_is_valid_probability(model):
    assert model.surface.shape == (12, 16)
    assert np.all(model.surface >= 0)
    assert np.all(model.surface <= 1)
    assert model.surface.max() > 0


def test_surface_increases_towards_goal(model):
    # Mean xT in the attacking quarter must exceed the defensive quarter
    assert model.surface[:, -4:].mean() > model.surface[:, :4].mean()


def test_value_zero_for_failed_or_non_move(model):
    actions = pd.DataFrame(
        {
            "type": ["pass", "shot"],
            "x": [50.0, 90.0],
            "y": [34.0, 34.0],
            "end_x": [80.0, np.nan],
            "end_y": [34.0, np.nan],
            "outcome": [False, True],
        }
    )
    values = model.value(actions)
    assert (values == 0).all()


def test_value_positive_for_big_forward_pass(model):
    actions = pd.DataFrame(
        {
            "type": ["pass"],
            "x": [20.0],
            "y": [34.0],
            "end_x": [95.0],
            "end_y": [34.0],
            "outcome": [True],
        }
    )
    assert model.value(actions).iloc[0] > 0


def test_cell_of_clips_to_grid(model):
    r, c = model.cell_of(
        pd.Series([0.0, PITCH_LENGTH]), pd.Series([0.0, PITCH_WIDTH])
    )
    assert r.min() >= 0 and r.max() <= model.n_rows - 1
    assert c.min() >= 0 and c.max() <= model.n_cols - 1


def test_move_types_constant():
    assert set(MOVE_TYPES) == {"pass", "carry", "dribble"}
