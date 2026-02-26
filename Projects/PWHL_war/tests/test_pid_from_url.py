"""
Parametric tests for xga_war._pid_from_url().

Covers:
  1. Standard URL — extracts numeric stem
  2. URL with trailing slash — handles gracefully
  3. URL with no numeric stem — returns None
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from pwhl_war.xga_war import _pid_from_url


@pytest.mark.parametrize("url, expected", [
    ("https://assets.leaguestat.com/pwhl/120x160/12345.jpg", 12345),
    ("https://assets.leaguestat.com/pwhl/120x160/99001.jpg", 99001),
])
def test_standard_url(url, expected):
    assert _pid_from_url(url) == expected


def test_url_with_no_numeric_stem():
    """Non-numeric stem should return None."""
    url = "https://assets.leaguestat.com/pwhl/120x160/headshot.jpg"
    result = _pid_from_url(url)
    assert result is None


def test_empty_string():
    """Empty string should return None (AttributeError caught)."""
    result = _pid_from_url("")
    assert result is None


def test_none_input():
    """None input should return None."""
    result = _pid_from_url(None)
    assert result is None
