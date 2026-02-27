"""
Tests for validate_war.check_gpw_constants().

Covers:
  1. A mock standings dict where derived gpw differs from the constant by >0.3
     should trigger a warning print containing the season key.
"""
import sys
import io
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from validate_war import check_gpw_constants
from pwhl_war.constants import GPW


def test_check_gpw_constants_warns_on_large_discrepancy(capsys):
    """
    If derived gpw deviates from the inline constant by >0.3, a warning
    containing the season key should be printed to stdout.

    We manufacture a discrepancy for '2024-25':
      inline constant: 5.049
      we inject a standings df where derived gpw = 5.049 + 0.4 = 5.449
      That requires: 2 * (total_goals / n_team_games) = 5.449
      => total_goals / n_team_games = 2.7245
      Use n_team_games=6 => total_goals = ~16.3 (round to 16 for int GF/GA)
    """
    # Build a minimal standings DataFrame that forces derived gpw >> 5.349
    # Formula: derived = 2 * GF.sum() / GP.sum()  (GA is ignored)
    # With GP.sum()=6, GF.sum()=17 => derived = 2*17/6 = 5.667
    # diff = |5.667 - 5.049| = 0.618 > 0.3 — triggers warning
    sched = pd.DataFrame({
        "GF": [3, 3, 3, 3, 3, 2],
        "GA": [2, 2, 2, 1, 1, 2],
        "GP": [1, 1, 1, 1, 1, 1],
    })
    # derived gpw = 2 * GF.sum() / GP.sum() = 2 * 17 / 6 = 5.667, diff >> 0.3

    standings_by_season = {"2024-25": sched}
    check_gpw_constants(standings_by_season)
    captured = capsys.readouterr()
    assert "2024-25" in captured.out, (
        f"Expected '2024-25' in warning output, got: {captured.out!r}"
    )
    assert "WARNING" in captured.out or "gpw_check" in captured.out


def test_check_gpw_constants_no_warning_on_small_discrepancy(capsys):
    """
    If the discrepancy is <= 0.3, no warning should be printed.
    """
    # inline for 2024-25 is 5.049; construct df giving derived gpw ~5.1
    # 2 * (total / n) = 5.1 => total/n = 2.55 => use n=6 teams, each GP=1
    # total_goals = 2.55 * 6 = 15.3 => 8 GF + 7 GA = 15 rows / 6
    # Formula: derived = 2 * GF.sum() / GP.sum()  (GA is ignored)
    # inline for 2024-25 is 5.049; want derived ≈ 5.0 (diff=0.049 < 0.3)
    # 2 * GF.sum() / GP.sum() = 5.0 => GF.sum() = 5.0*6/2 = 15
    sched3 = pd.DataFrame({
        "GF": [3, 3, 3, 2, 2, 2],
        "GA": [2, 2, 2, 2, 2, 2],
        "GP": [1, 1, 1, 1, 1, 1],
    })
    # derived = 2 * 15 / 6 = 5.0, diff = |5.0 - 5.049| = 0.049 < 0.3 => no WARNING
    standings_by_season = {"2024-25": sched3}
    check_gpw_constants(standings_by_season)
    captured = capsys.readouterr()
    # derived=5.0, diff=|5.0-5.049|=0.049 < 0.3 => no WARNING
    assert "WARNING" not in captured.out
