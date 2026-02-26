"""
Integration tests for Phase 9: coordinate xG integration into WAR models.

Covers:
  1. XGAWar._aggregate_games() uses API-native xG field, not XG_MAP, for
     individual ixG accumulation.  Shot with API xG=0.200 but a quality
     label that XG_MAP would map to 0.128 should produce ixG=0.200.
  2. XGWar.fit() with an optional pbp_df produces the canonical output
     column schema (same columns as baseline fit without pbp_df).
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from pwhl_war.xga_war import XGAWar, PWHLApiLoader
from pwhl_war.box_war import XGWar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_shot_pbp(pid: int, xg_native: float, quality_label: str) -> list:
    """Single shot event with explicit API xG and a quality label."""
    return [
        {
            "event": "shot",
            "details": {
                "shotQuality": quality_label,
                "xG": xg_native,
                "shooterTeamId": "10",
                "shooter": {"id": pid},
            },
        }
    ]


def _make_summary_with_skater(pid: int) -> dict:
    """Single-skater game summary so the player appears in aggregated records."""
    return {
        "homeTeam": {
            "skaters": [
                {
                    "playerImageURL": (
                        f"https://assets.leaguestat.com/pwhl/120x160/{pid}.jpg"
                    ),
                    "name": f"Player {pid}",
                    "position": "F",
                    "stats": {"toi": "20:00"},
                }
            ]
        },
        "visitingTeam": {"skaters": []},
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def game_data_df():
    """Minimal game_data_df for XGWar with 3 players above the TOI floor."""
    return pd.DataFrame(
        {
            "PlayerID":  [1, 2, 3],
            "Name":      ["Alice Smith", "Bob Jones", "Carol Lee"],
            "Team":      ["BOS", "MIN", "BOS"],
            "position":  ["F", "D", "F"],
            "TOI":       [200.0, 180.0, 160.0],
            "EV_G":      [3, 1, 2],
            "EV_A1":     [2, 1, 1],
            "EV_A2":     [1, 0, 1],
            "EV_Shots":  [30, 20, 25],
            "EV_ixG":    [2.1, 1.2, 1.8],
            "PP_G":      [1, 0, 0],
            "PP_A1":     [1, 0, 0],
            "PP_A2":     [0, 0, 0],
            "PP_Shots":  [10, 0, 0],
            "PP_ixG":    [0.5, 0.0, 0.0],
            "SH_G":      [0, 0, 0],
            "SH_A1":     [0, 0, 0],
            "SH_A2":     [0, 0, 0],
            "SH_Shots":  [0, 0, 0],
            "SH_ixG":    [0.0, 0.0, 0.0],
            "EN_G":      [0, 0, 0],
            "PIM":       [2, 4, 0],
            "plusMinus": [5, -2, 3],
            "hits":      [10, 15, 8],
        }
    )


@pytest.fixture
def pbp_df():
    """Minimal PBP DataFrame matching the players in game_data_df."""
    rng = np.random.default_rng(42)
    n = 36
    players = ["Alice Smith"] * 14 + ["Bob Jones"] * 10 + ["Carol Lee"] * 12
    x = rng.uniform(5, 60, n)
    y = rng.uniform(-15, 15, n)
    event = np.where(rng.random(n) < 0.15, "Goal", "Shot")
    return pd.DataFrame(
        {
            "player":   players,
            "season":   ["2024/2025"] * n,
            "event":    event,
            "x":        x,
            "y":        y,
            "strength": rng.choice(["EV", "PP", "SH"], n),
            "xG":       rng.uniform(0.02, 0.30, n),
            "game_id":  rng.integers(100, 120, n),
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_xga_war_uses_api_native_xg():
    """
    XGAWar._aggregate_games() must use the API-native xG field, not XG_MAP.

    A shot with quality label "Quality on net" (XG_MAP → 0.128) but API
    xG = 0.200 should produce player ixG = 0.200 after removing XG_MAP.
    """
    pid = 99
    api_xg = 0.200  # API says 0.200; XG_MAP["Quality on net"] = 0.128

    pbp   = _make_shot_pbp(pid, api_xg, quality_label="Quality on net")
    summ  = _make_summary_with_skater(pid)

    mock_loader = MagicMock(spec=PWHLApiLoader)
    mock_loader.get_pbp.return_value    = pbp
    mock_loader.get_summary.return_value = summ

    model   = XGAWar()
    records = model._aggregate_games(
        [1], mock_loader,
        {1: {"home_abbr": "HOM", "away_abbr": "VIS"}},
    )

    rec = next((r for r in records if r["player_id"] == pid), None)
    assert rec is not None, f"Player {pid} should appear in aggregated records"
    assert abs(rec["ixG"] - api_xg) < 0.001, (
        f"ixG={rec['ixG']:.4f}: expected {api_xg} (API-native xG). "
        "XG_MAP lookup would give 0.128 — verify XG_MAP has been removed."
    )


def test_xg_war_schema_with_coord_xg(game_data_df, pbp_df):
    """
    XGWar.fit() with pbp_df must produce the canonical output column schema.

    All columns present in a baseline fit (no pbp_df) must also appear
    when coord-model xG replaces the CSV ixG sum.
    """
    expected_cols = [
        "name", "player_id", "team", "pos", "gp", "toi_min",
        "o_xG60", "d_value60", "oWAR", "dWAR", "WAR",
    ]

    model  = XGWar(goals_per_win=5.0)
    result = model.fit(game_data_df, pbp_df=pbp_df).get_war()

    for col in expected_cols:
        assert col in result.columns, f"Missing column after coord xG integration: {col}"
