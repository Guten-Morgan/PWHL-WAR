"""
Tests for pwhl_war.stats_utils.compute_defensive_value60()

Uses a 5-player fixture DataFrame with known pm60, o_xG60, Team, toi_min
values to characterise the 4-step OLS → team-adjust → league-adjust → scale
pipeline and guard against regressions.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from pwhl_war.stats_utils import compute_defensive_value60


@pytest.fixture
def fixture_df():
    """
    5-player fixture with controlled pm60, o_xG60, Team, toi_min values.
    Two teams (A, B), all players above 50-min TOI threshold.
    """
    return pd.DataFrame({
        "player_id": [1, 2, 3, 4, 5],
        "pm60":      [0.5, -0.3, 0.8, -0.6, 0.1],
        "o_xG60":    [0.9, 0.6, 1.1, 0.7, 0.8],
        "Team":      ["A", "A", "B", "B", "B"],
        "toi_min":   [200.0, 180.0, 220.0, 160.0, 190.0],
    })


def test_returns_d_value60_column(fixture_df):
    qual_mask = fixture_df["toi_min"] >= 50.0
    result = compute_defensive_value60(
        fixture_df, qual_mask,
        raw_col="pm60", off_col="o_xG60",
        team_col="Team", defense_weight=0.36, sign=1,
    )
    assert "d_value60" in result.columns


def test_d_value60_is_league_mean_zero(fixture_df):
    """TOI-weighted league mean of d_value60 should be ~0."""
    qual_mask = fixture_df["toi_min"] >= 50.0
    result = compute_defensive_value60(
        fixture_df, qual_mask,
        raw_col="pm60", off_col="o_xG60",
        team_col="Team", defense_weight=0.36, sign=1,
    )
    qual = result[qual_mask]
    w_mean = np.average(qual["d_value60"], weights=qual["toi_min"])
    assert abs(w_mean) < 1e-8, f"Weighted league mean not ~0: {w_mean}"


def test_sign_inverts_direction(fixture_df):
    """sign=-1 should flip the sign of d_value60 relative to sign=+1."""
    qual_mask = fixture_df["toi_min"] >= 50.0
    res_pos = compute_defensive_value60(
        fixture_df, qual_mask,
        raw_col="pm60", off_col="o_xG60",
        team_col="Team", defense_weight=0.36, sign=1,
    )
    res_neg = compute_defensive_value60(
        fixture_df, qual_mask,
        raw_col="pm60", off_col="o_xG60",
        team_col="Team", defense_weight=0.36, sign=-1,
    )
    np.testing.assert_array_almost_equal(
        res_pos["d_value60"].values,
        -res_neg["d_value60"].values,
        decimal=6,
    )


def test_does_not_mutate_input(fixture_df):
    """Original DataFrame must not be modified in place."""
    original_pm60 = fixture_df["pm60"].copy()
    qual_mask = fixture_df["toi_min"] >= 50.0
    compute_defensive_value60(
        fixture_df, qual_mask,
        raw_col="pm60", off_col="o_xG60",
        team_col="Team", defense_weight=0.36, sign=1,
    )
    pd.testing.assert_series_equal(fixture_df["pm60"], original_pm60)


# ── Phase 10 diagnostic tests ─────────────────────────────────────────────────

@pytest.fixture
def three_team_df():
    """
    3-team 6-player fixture with clear between-team pm60 signal.
    Team A (high pm60), Team B (medium), Team C (low).
    Equal TOI and equal o_xG60 so OLS residual ≈ raw pm60 (slope near 0).
    """
    return pd.DataFrame({
        "player_id": [1, 2, 3, 4, 5, 6],
        "pm60":      [2.0, 1.0, 0.2, -0.2, -1.0, -2.0],
        "o_xG60":    [0.5, 0.5, 0.5,  0.5,  0.5,  0.5],
        "Team":      ["A", "A", "B",  "B",  "C",  "C"],
        "toi_min":   [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
    })


def test_team_centering_zeros_toi_weighted_team_sum(three_team_df):
    """
    With team_adjust=True, the TOI-weighted sum of d_value60 within each
    team must be ~0.  This is the structural property that zeroes out
    between-team defensive signal (Step 2 subtracts each team's mean).
    """
    qual_mask = three_team_df["toi_min"] >= 50.0
    result = compute_defensive_value60(
        three_team_df, qual_mask,
        raw_col="pm60", off_col="o_xG60",
        team_col="Team", defense_weight=0.36, sign=1,
        team_adjust=True,
    )
    for team in ["A", "B", "C"]:
        mask = result["Team"] == team
        toi_wt_sum = (result.loc[mask, "d_value60"] * result.loc[mask, "toi_min"]).sum()
        assert abs(toi_wt_sum) < 1e-8, (
            f"Team {team}: TOI-weighted d_value60 sum = {toi_wt_sum} (expected ~0)"
        )


def test_no_team_centering_preserves_between_team_signal(three_team_df):
    """
    With team_adjust=False, between-team variance in d_value60 must be
    non-zero — the raw pm60 signal (team A > B > C) is preserved.
    """
    qual_mask = three_team_df["toi_min"] >= 50.0
    result = compute_defensive_value60(
        three_team_df, qual_mask,
        raw_col="pm60", off_col="o_xG60",
        team_col="Team", defense_weight=0.36, sign=1,
        team_adjust=False,
    )
    team_means = result.groupby("Team")["d_value60"].mean()
    assert team_means.var() > 1e-6, (
        f"Expected non-zero between-team variance with team_adjust=False, "
        f"got team means: {team_means.to_dict()}"
    )
