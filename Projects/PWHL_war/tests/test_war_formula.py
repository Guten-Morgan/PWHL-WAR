"""
Tests for the core WAR formula in XGWar (box_war.py).

Covers:
  1. oWAR equals expected value given known o_xG60, toi_min, gpw
  2. dWAR equals expected value
  3. WAR == oWAR + dWAR identity holds for all rows
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from pwhl_war.box_war import XGWar


def _make_game_data(n_players=10, n_games=5, seed=0):
    """Build a minimal game_data DataFrame."""
    rng = np.random.default_rng(seed)
    records = []
    teams = ["A"] * (n_players // 2) + ["B"] * (n_players - n_players // 2)
    for i in range(n_players):
        for _ in range(n_games):
            records.append({
                "PlayerID": i + 1,
                "Name": f"Player{i}",
                "Team": teams[i],
                "position": "F",
                "TOI": rng.uniform(12, 20),
                "EV_ixG": rng.uniform(0.05, 0.4),
                "PP_ixG": rng.uniform(0, 0.1),
                "SH_ixG": 0.0,
                "plusMinus": rng.integers(-3, 4),
                "PIM": 0,
                "EV_G": 0, "EV_A1": 0, "EV_A2": 0, "EV_Shots": 1,
                "PP_G": 0, "PP_A1": 0, "PP_A2": 0, "PP_Shots": 0,
                "SH_G": 0, "SH_A1": 0, "SH_A2": 0, "SH_Shots": 0,
                "EN_G": 0,
                "hits": 0,
            })
    return pd.DataFrame(records)


def _make_schedule():
    return pd.DataFrame({
        "GF": [3, 2, 4, 1, 5],
        "GA": [2, 3, 1, 4, 2],
        "game_status": ["Final"] * 5,
        "SeasonStage": ["Regular"] * 5,
    })


@pytest.fixture
def fitted_model():
    gd = _make_game_data()
    sched = _make_schedule()
    m = XGWar(min_toi_min=5.0)
    m.fit(gd, schedule_df=sched)
    return m


def test_owar_formula(fitted_model):
    """oWAR = oGAR / gpw = (o_xG60 - o_repl) * toi_min/60 / gpw for each player."""
    df = fitted_model.get_war(min_toi=0.0)
    gpw = fitted_model.goals_per_win_
    o_repl = fitted_model.o_replacement_val60_

    expected_oWAR = ((df["o_xG60"] - o_repl) * df["toi_min"] / 60 / gpw).round(3)
    pd.testing.assert_series_equal(
        df["oWAR"].reset_index(drop=True),
        expected_oWAR.reset_index(drop=True),
        check_names=False,
    )


def test_dwar_formula(fitted_model):
    """dWAR = dGAR / gpw.  dGAR is first rounded to 3dp, then divided by gpw."""
    df = fitted_model.get_war(min_toi=0.0)
    gpw = fitted_model.goals_per_win_
    d_repl = fitted_model.d_replacement_val60_

    # The model rounds dGAR to 3dp before dividing by gpw — match that path.
    expected_dGAR = ((df["d_value60"] - d_repl) * df["toi_min"] / 60).round(3)
    expected_dWAR = (expected_dGAR / gpw).round(3)
    pd.testing.assert_series_equal(
        df["dWAR"].reset_index(drop=True),
        expected_dWAR.reset_index(drop=True),
        check_names=False,
    )


def test_war_equals_owar_plus_dwar(fitted_model):
    """WAR == oWAR + dWAR for all rows (within floating-point rounding)."""
    df = fitted_model.get_war(min_toi=0.0)
    np.testing.assert_array_almost_equal(
        df["WAR"].values,
        (df["oWAR"] + df["dWAR"]).round(3).values,
        decimal=3,
    )
