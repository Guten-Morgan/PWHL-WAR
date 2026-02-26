"""
Tests for pwhl_war.coord_xg.CoordXGModel.

Covers:
  1. train() returns a fitted model with coef_ attribute
  2. predict() values are in [0, 1]
  3. player_xg_season() returns DataFrame with player, season, coord_xG columns
  4. brier_score() is below 0.35 on training data
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from pwhl_war.coord_xg import CoordXGModel


@pytest.fixture
def synthetic_pbp():
    """
    20-row synthetic PBP fixture.
    Shots close to net (small dist) are more likely to be goals.
    """
    rng = np.random.default_rng(7)
    n = 40
    # Half close-in shots (higher goal probability), half long-range
    x_close = rng.uniform(5, 20, n // 2)
    x_far   = rng.uniform(40, 70, n // 2)
    x = np.concatenate([x_close, x_far])
    y = rng.uniform(-15, 15, n)
    strength = rng.choice(["EV", "PP", "SH"], n)

    # Goal more likely when close
    goal_prob = np.where(x < 25, 0.25, 0.05)
    event = np.where(rng.random(n) < goal_prob, "Goal", "Shot")

    return pd.DataFrame({
        "x":        x,
        "y":        y,
        "strength": strength,
        "event":    event,
        "player":   [f"P{i % 10}" for i in range(n)],
        "season":   ["2024-25"] * n,
        "game_id":  rng.integers(100, 120, n),
    })


def test_train_returns_fitted_model(synthetic_pbp):
    """train() should return self with _model having coef_ attribute."""
    model = CoordXGModel()
    result = model.train(synthetic_pbp)
    assert result is model, "train() should return self"
    assert model._is_fitted, "Model should be marked as fitted"
    assert hasattr(model._model, "coef_"), "Fitted model should have coef_"


def test_predict_values_in_unit_interval(synthetic_pbp):
    """predict() should return values in [0, 1] for all rows."""
    model = CoordXGModel().train(synthetic_pbp)
    preds = model.predict(synthetic_pbp)
    assert preds.shape == (len(synthetic_pbp),), "Wrong shape"
    assert (preds >= 0.0).all() and (preds <= 1.0).all(), (
        f"Predictions out of [0,1]: min={preds.min():.4f} max={preds.max():.4f}"
    )


def test_player_xg_season_schema(synthetic_pbp):
    """player_xg_season() must return player, season, coord_xG columns."""
    model = CoordXGModel().train(synthetic_pbp)
    result = model.player_xg_season(synthetic_pbp)
    for col in ["player", "season", "coord_xG"]:
        assert col in result.columns, f"Missing column: {col}"
    assert len(result) > 0, "Should have at least one player"


def test_brier_score_below_ceiling(synthetic_pbp):
    """Brier score on training data should be < 0.35."""
    model = CoordXGModel().train(synthetic_pbp)
    bs = model.brier_score(synthetic_pbp)
    assert bs < 0.35, f"Brier score too high: {bs:.4f} (ceiling 0.35)"
