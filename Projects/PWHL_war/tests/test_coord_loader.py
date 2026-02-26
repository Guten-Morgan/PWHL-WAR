"""
Tests for pwhl_war.coord_loader.CoordLoader.

Covers:
  1. x, y, xG are float columns in output
  2. Season format mapping '2024-25' -> '2024/2025' is applied in API call
  3. Blocked shots (null xG) are excluded when drop_blocked=True
"""
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from pwhl_war.coord_loader import CoordLoader


FIXTURE_ROWS = [
    {"Event": "Shot", "Strength": "EV", "Player 1": "Alice Smith",
     "GameID": 101, "Season": "2024/2025", "x": 45.0, "y": 12.3, "xG": 0.12},
    {"Event": "Goal", "Strength": "PP", "Player 1": "Bob Jones",
     "GameID": 101, "Season": "2024/2025", "x": 10.0, "y": 0.0,  "xG": 0.45},
    {"Event": "Shot", "Strength": "EV", "Player 1": "Carol Lee",
     "GameID": 102, "Season": "2024/2025", "x": 30.0, "y": -8.0, "xG": None},  # blocked
    {"Event": "Shot", "Strength": "SH", "Player 1": "Dana Fox",
     "GameID": 102, "Season": "2024/2025", "x": 55.0, "y": 5.0,  "xG": 0.08},
    {"Event": "Block", "Strength": "EV", "Player 1": "Eve Park",
     "GameID": 102, "Season": "2024/2025", "x": None, "y": None,  "xG": None},
]


def _make_mock_response(rows):
    mock_resp = MagicMock()
    mock_resp.json.return_value = rows
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


def test_output_columns_are_float(tmp_path):
    """x, y, xG should be float dtype in the returned DataFrame."""
    loader = CoordLoader(delay=0.0)
    mock_resp = _make_mock_response(FIXTURE_ROWS)

    with patch.object(loader._session, "get", return_value=mock_resp):
        df = loader.fetch_pbp(["2024-25"])

    assert pd.api.types.is_float_dtype(df["x"]), "x is not float"
    assert pd.api.types.is_float_dtype(df["y"]), "y is not float"
    assert pd.api.types.is_float_dtype(df["xG"]), "xG is not float"


def test_season_format_mapping():
    """The API call should use '2024/2025' when season='2024-25' is requested."""
    loader = CoordLoader(delay=0.0)
    mock_resp = _make_mock_response(FIXTURE_ROWS)
    call_args = []

    def capture_get(url, **kwargs):
        call_args.append(kwargs.get("params", {}))
        return mock_resp

    with patch.object(loader._session, "get", side_effect=capture_get):
        loader.fetch_pbp(["2024-25"])

    assert call_args, "No API call was made"
    assert call_args[0].get("season") == "2024/2025", (
        f"Expected season='2024/2025', got {call_args[0].get('season')!r}"
    )


def test_drop_blocked_excludes_null_xg():
    """Rows with xG=null should be excluded when drop_blocked=True."""
    loader = CoordLoader(delay=0.0)
    mock_resp = _make_mock_response(FIXTURE_ROWS)

    with patch.object(loader._session, "get", return_value=mock_resp):
        df = loader.fetch_pbp(["2024-25"], drop_blocked=True)

    # FIXTURE_ROWS has 3 Shot/Goal with non-null xG; 1 Shot with null xG;
    # 1 Block (excluded by event filter). So we expect 3 rows.
    assert df["xG"].notna().all(), "Null xG rows not excluded"
    assert len(df) == 3, f"Expected 3 non-null xG rows, got {len(df)}"


def test_drop_blocked_false_keeps_null_xg():
    """When drop_blocked=False, rows with null xG should be retained."""
    loader = CoordLoader(delay=0.0)
    mock_resp = _make_mock_response(FIXTURE_ROWS)

    with patch.object(loader._session, "get", return_value=mock_resp):
        df = loader.fetch_pbp(["2024-25"], drop_blocked=False)

    # Shot+Goal events = 4 (1 has null xG), Block excluded by event filter
    assert len(df) == 4, f"Expected 4 Shot/Goal rows, got {len(df)}"
