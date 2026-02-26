"""
Tests for XGWar._compute_gpw() (box_war.py).

Covers:
  1. Correct gpw from a fixture schedule DataFrame with known GF/GA
  2. Returns 6.0 on empty DataFrame
  3. Returns 6.0 on None input
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from pwhl_war.box_war import XGWar


@pytest.fixture
def model():
    return XGWar()


def test_gpw_from_schedule(model):
    """
    Known fixture: 4 games, total goals = 3+2+4+1+2+3+1+4 = 20
    n_team_game_rows = 8 (each game has 2 team rows)
    avg_per_team_per_game = 20 / (2 * 4) = 2.5
    gpw = 2 * 2.5 = 5.0
    """
    sched = pd.DataFrame({
        "GF": [3, 2, 4, 1],
        "GA": [2, 3, 1, 4],
        "game_status": ["Final"] * 4,
        "SeasonStage": ["Regular"] * 4,
    })
    gpw = model._compute_gpw(sched)
    assert abs(gpw - 5.0) < 0.001, f"Expected gpw=5.0, got {gpw}"


def test_gpw_empty_df_returns_fallback(model):
    """Empty DataFrame should fall back to 6.0."""
    gpw = model._compute_gpw(pd.DataFrame())
    assert gpw == 6.0


def test_gpw_none_returns_fallback(model):
    """None input should fall back to 6.0."""
    gpw = model._compute_gpw(None)
    assert gpw == 6.0
