"""
test_game_scope.py
------------------
Regular-season game selection from the sportsdataverse schedule master.

Behaviours covered:
 1. Playoff games are excluded; regular-season games are kept.
 2. Only the requested season is returned ("2025-26" -> season column 2026).
 3. Ids come back sorted and unique.
 4. A schedule without the required columns raises ValueError.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pwhl_war.scope import regular_season_game_ids


@pytest.fixture
def schedule():
    return pd.DataFrame({
        "game_id":   [329, 328, 338, 340, 105, 195],
        "season":    [2026, 2026, 2026, 2026, 2025, 2025],
        "game_type": ["regular", "regular", "playoffs", "playoffs", "regular", "playoffs"],
    })


def test_regular_season_ids_exclude_playoffs(schedule):
    assert regular_season_game_ids(schedule, "2025-26") == [328, 329]


def test_only_requested_season_returned(schedule):
    assert regular_season_game_ids(schedule, "2024-25") == [105]


def test_ids_sorted_and_unique(schedule):
    dup = pd.concat([schedule, schedule.iloc[[0]]], ignore_index=True)
    ids = regular_season_game_ids(dup, "2025-26")
    assert ids == sorted(set(ids))


def test_missing_columns_raise(schedule):
    with pytest.raises(ValueError, match="game_type"):
        regular_season_game_ids(schedule.drop(columns=["game_type"]), "2025-26")
