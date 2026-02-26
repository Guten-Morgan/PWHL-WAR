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
    # Build a minimal standings DataFrame that forces derived gpw ~5.45
    # GF/GA per row are per-team per-game; GP counts games per team.
    # We set 6 team-season rows with GP=1 each, GF+GA summing to ~16
    sched = pd.DataFrame({
        "GF": [3, 3, 3, 2, 2, 3],
        "GA": [2, 2, 2, 1, 1, 2],
        "GP": [1, 1, 1, 1, 1, 1],
    })
    # derived gpw = 2 * (GF.sum() + GA.sum()) / GP.sum()
    # = 2 * (16 + 10) / 6 = 2 * 26/6 = 8.667 — way above 5.049, diff >> 0.3

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
    sched = pd.DataFrame({
        "GF": [3, 3, 2, 2, 2, 2],
        "GA": [2, 2, 2, 2, 2, 2],
        "GP": [1, 1, 1, 1, 1, 1],
    })
    # derived = 2*(14+12)/6 = 2*26/6 = 8.667 -- still large
    # Let me make it small: use very close values
    # inline 5.049, want derived 5.0 (diff=0.049 < 0.3)
    # 2*(total)/6 = 5.0 => total=15 => e.g. 3+2=5 per game, 3 games: GF=[3,2,2], GA=[2,2,2]
    # That's 3+2+2+2+2+2=13 total, 2*13/6=4.33 -- nope
    # Simpler: derive exactly -- GP.sum()=6, want total_goals=15
    # GF sum + GA sum = 15 => e.g. GF=[3,2,2,2,2,2]=13 GA=[1,0,0,0,0,1]=2 total=15
    sched2 = pd.DataFrame({
        "GF": [3, 2, 2, 2, 2, 2],
        "GA": [2, 2, 2, 2, 2, 2],
        "GP": [1, 1, 1, 1, 1, 1],
    })
    # derived = 2*(13+12)/6 = 50/6 = 8.33 -- still not 5.0
    # The formula in check_gpw_constants uses GF.sum() + GA.sum() / GP.sum()
    # We need: 2*(GF_sum+GA_sum) / GP_sum = ~5.05
    # With GP.sum()=6: (GF+GA)_sum = 5.05*6/2 = 15.15
    # GF=[2,3,2,2,2,2]=13, GA=[0,0,1,0,1,0]=2 => total=15, gpw=5.0 diff=0.049
    sched3 = pd.DataFrame({
        "GF": [2, 3, 2, 2, 2, 2],
        "GA": [0, 0, 1, 0, 1, 0],
        "GP": [1, 1, 1, 1, 1, 1],
    })
    standings_by_season = {"2024-25": sched3}
    check_gpw_constants(standings_by_season)
    captured = capsys.readouterr()
    # diff should be |5.0 - 5.049| = 0.049 < 0.3 => no WARNING
    assert "WARNING" not in captured.out
