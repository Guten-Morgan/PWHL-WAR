"""
test_onice_loader.py
--------------------
Loading and cleaning the sportsdataverse PWHL files (pwhl_war/onice.py).

Behaviours covered:
 1. A goal recorded as both a `shot` row (goal=True) and a `goal` row becomes one event.
 2. A `goal` row with no matching shot row is kept.
 3. Non-goal rows are untouched.
 4. load_season reads cached files and returns pbp / shifts / box.
 5. load_season raises ValueError naming a missing required column.
 6. Season labels map to the end-year file suffix (2025-26 -> 2026).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pwhl_war.onice import (
    REQUIRED_BOX, REQUIRED_PBP, REQUIRED_SHIFTS, dedupe_goals, file_year, load_season,
)


def _pbp(rows):
    base = {"game_id": 1, "period_of_game": "1", "time_of_period": "5:00", "player_id": 10,
            "goal": False}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_dedupe_goals_keeps_one_row_per_goal():
    pbp = _pbp([
        {"event": "shot", "goal": True},
        {"event": "goal", "goal": True},
    ])
    out = dedupe_goals(pbp)
    assert len(out) == 1
    assert out.iloc[0]["event"] == "shot"


def test_unmatched_goal_row_is_kept():
    pbp = _pbp([
        {"event": "shot", "goal": True},
        {"event": "goal", "goal": True},
        {"event": "goal", "goal": True, "time_of_period": "9:00", "player_id": 11},
    ])
    out = dedupe_goals(pbp)
    assert len(out) == 2
    assert (out["event"] == "goal").sum() == 1


def test_non_goal_rows_untouched():
    pbp = _pbp([
        {"event": "shot", "goal": False, "time_of_period": "1:00"},
        {"event": "blocked_shot", "time_of_period": "2:00"},
        {"event": "hit", "time_of_period": "3:00"},
    ])
    out = dedupe_goals(pbp)
    pd.testing.assert_frame_equal(out.reset_index(drop=True), pbp.reset_index(drop=True))


def test_same_player_goals_at_different_times_not_collapsed():
    pbp = _pbp([
        {"event": "shot", "goal": True, "time_of_period": "5:00"},
        {"event": "goal", "goal": True, "time_of_period": "5:00"},
        {"event": "shot", "goal": True, "time_of_period": "15:00"},
        {"event": "goal", "goal": True, "time_of_period": "15:00"},
    ])
    out = dedupe_goals(pbp)
    assert len(out) == 2


def _write_cache(tmp_path, drop=None):
    """Minimal valid cache for 2025-26 (files suffixed 2026)."""
    frames = {
        f"play_by_play_2026.parquet": pd.DataFrame({c: [1] for c in REQUIRED_PBP}),
        f"shifts_2026.parquet":       pd.DataFrame({c: [1] for c in REQUIRED_SHIFTS}),
        f"skater_box_2026.parquet":   pd.DataFrame({c: [1] for c in REQUIRED_BOX}),
    }
    for name, df in frames.items():
        if drop and drop[0] == name:
            df = df.drop(columns=[drop[1]])
        df.to_parquet(tmp_path / name)


def test_load_season_returns_three_frames(tmp_path):
    _write_cache(tmp_path)
    data = load_season("2025-26", cache_dir=tmp_path)
    assert set(data) == {"pbp", "shifts", "box"}
    assert all(len(df) == 1 for df in data.values())


def test_load_rejects_missing_required_columns(tmp_path):
    _write_cache(tmp_path, drop=("shifts_2026.parquet", "end_s"))
    with pytest.raises(ValueError, match="end_s"):
        load_season("2025-26", cache_dir=tmp_path)


def test_file_year_is_season_end_year():
    assert file_year("2023-24") == 2024
    assert file_year("2025-26") == 2026
    with pytest.raises(ValueError):
        file_year("2025")
