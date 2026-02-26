"""
Tests for box_war.XGWar.get_war() output column schema (Phase 4).

Verifies that the output uses lowercase snake_case canonical column names
to match xga_war.XGAWar output schema.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from pwhl_war.box_war import XGWar


@pytest.fixture
def minimal_game_data():
    """Minimal game_data DataFrame sufficient to fit XGWar."""
    rng = np.random.default_rng(42)
    n = 60  # 60 player-games across 15 players × 4 games
    players = [f"P{i}" for i in range(15)]
    teams   = ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B",
               "C", "C", "C", "C", "C"]
    records = []
    for i, (pid, team) in enumerate(zip(players, teams)):
        for game in range(4):
            records.append({
                "PlayerID": i + 1,
                "Name": pid,
                "Team": team,
                "position": "F",
                "TOI": rng.uniform(10, 25),
                "EV_ixG": rng.uniform(0, 0.3),
                "PP_ixG": rng.uniform(0, 0.1),
                "SH_ixG": 0.0,
                "plusMinus": rng.integers(-2, 3),
                "PIM": 0,
                "EV_G": 0, "EV_A1": 0, "EV_A2": 0, "EV_Shots": 1,
                "PP_G": 0, "PP_A1": 0, "PP_A2": 0, "PP_Shots": 0,
                "SH_G": 0, "SH_A1": 0, "SH_A2": 0, "SH_Shots": 0,
                "EN_G": 0,
                "hits": 0,
            })
    return pd.DataFrame(records)


@pytest.fixture
def minimal_schedule():
    """Minimal schedule DataFrame for GPW computation."""
    return pd.DataFrame({
        "GF": [3, 2, 4, 1, 3, 2],
        "GA": [2, 3, 1, 4, 2, 3],
        "game_status": ["Final"] * 6,
        "SeasonStage": ["Regular"] * 6,
    })


def test_get_war_has_canonical_columns(minimal_game_data, minimal_schedule):
    """get_war() must return team, player_id, name, pos, gp columns."""
    model = XGWar(min_toi_min=10.0)
    model.fit(minimal_game_data, schedule_df=minimal_schedule)
    result = model.get_war(min_toi=0.0)
    for col in ["team", "player_id", "name", "pos", "gp"]:
        assert col in result.columns, f"Missing canonical column: {col}"


def test_get_war_no_uppercase_legacy_columns(minimal_game_data, minimal_schedule):
    """get_war() must NOT return the old uppercase column names."""
    model = XGWar(min_toi_min=10.0)
    model.fit(minimal_game_data, schedule_df=minimal_schedule)
    result = model.get_war(min_toi=0.0)
    for old_col in ["Team", "PlayerID", "Name", "position", "GP"]:
        assert old_col not in result.columns, f"Old column still present: {old_col}"
