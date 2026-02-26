"""
Tests for pwhl_war.csv_loader.

Covers:
  1. _load_csv() raises ValueError when file content starts with '<' (HTML)
  2. SEASON_CODES reverse-maps '2024-25' → '20242025' correctly
"""
import sys
import io
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from pwhl_war.csv_loader import PWHLCsvLoader
from pwhl_war.constants   import SEASON_CODES


def test_html_guard_raises_value_error(tmp_path):
    """
    If the cached file starts with '<', _load_csv() (called via _download())
    should raise ValueError with a descriptive message.
    """
    loader = PWHLCsvLoader(cache_dir=tmp_path)

    html_content = b"<html><body>Access Denied</body></html>"

    mock_resp = MagicMock()
    mock_resp.content = html_content
    mock_resp.raise_for_status = MagicMock()

    with patch.object(loader._session, "get", return_value=mock_resp):
        with pytest.raises(ValueError, match="HTML"):
            loader._load_csv("game_data")


def test_season_codes_reverse_map():
    """SEASON_CODES must reverse-map '2024-25' → '20242025'."""
    reverse = {v: k for k, v in SEASON_CODES.items()}
    assert reverse.get("2024-25") == "20242025"
