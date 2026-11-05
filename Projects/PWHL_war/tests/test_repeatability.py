"""
test_repeatability.py
---------------------
The comparison harness in validate_onice.py.

Behaviours covered:
 1. split_half_games alternates whole games (sorted order), never splits a game.
 2. fa_candidate output depends only on the games asked for (onice_xga,
    toi_share_ownxg and naive).
 3. The on-ice candidate credits actual on-ice defenders; the TOI-share control
    spreads the same xG by ice-time share (same total, different players).
 4. season_to_season_rows applies the 5v5 minutes cut to both seasons.
 5. bootstrap_diff is zero for identical candidates and deterministic for a seed.
 6. spearman_brown corrects a half-sample correlation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from validate_onice import (
    SeasonData, bootstrap_diff, fa_candidate, season_to_season_rows,
    spearman_brown, split_half_games,
)


# ---------------------------------------------------------------- split half
def test_split_half_alternates_games_not_stints():
    a, b = split_half_games([212, 210, 211, 214, 213])
    assert a == [210, 212, 214] and b == [211, 213]
    assert set(a).isdisjoint(b) and set(a) | set(b) == {210, 211, 212, 213, 214}


# ----------------------------------------------------------- synthetic season
def _data():
    """Two games, teams HOM / AWY. Home skaters 1-5 (+6 plays only game 2), away 11-15."""
    home, away = (1, 2, 3, 4, 5), (11, 12, 13, 14, 15)
    shots = pd.DataFrame([
        # game 1: home shoots xg .10 (credited to away 11-15), away shoots xg .20 (credited to home 1-5)
        {"game_id": 1, "team_id": 1, "home_team_id": 1, "xg": 0.10, "home_skaters": home, "away_skaters": away,
         "n_home": 5, "n_away": 5, "rebuild_ok": True},
        {"game_id": 1, "team_id": 2, "home_team_id": 1, "xg": 0.20, "home_skaters": home, "away_skaters": away,
         "n_home": 5, "n_away": 5, "rebuild_ok": True},
        # game 2: home shoots xg .40
        {"game_id": 2, "team_id": 1, "home_team_id": 1, "xg": 0.40, "home_skaters": (6, 2, 3, 4, 5),
         "away_skaters": away, "n_home": 5, "n_away": 5, "rebuild_ok": True},
    ])
    rows = []
    for gid in (1, 2):
        for pid in (home if gid == 1 else (6, 2, 3, 4, 5)):
            rows.append({"GameID": gid, "Team": "HOM", "PlayerID": pid, "TOI": 20.0})
        for pid in away:
            rows.append({"GameID": gid, "Team": "AWY", "PlayerID": pid, "TOI": 20.0})
    # a sixth home skater with half the minutes in game 1 only changes shares, not totals
    sh = []
    for gid in (1, 2):
        for pid in (home if gid == 1 else (6, 2, 3, 4, 5)):
            sh.append({"game_id": gid, "period": 1, "player_id": pid, "home": 1, "start_s": 1200, "end_s": 0})
        for pid in away:
            sh.append({"game_id": gid, "period": 1, "player_id": pid, "home": 0, "start_s": 1200, "end_s": 0})
    sched_map = {g: {"home_abbr": "HOM", "away_abbr": "AWY", "home_id": "1", "away_id": "2"} for g in (1, 2)}
    return SeasonData(season="2025-26", game_data=pd.DataFrame(rows), shots=shots,
                      shifts=pd.DataFrame(sh), sched_map=sched_map, coord_df=None,
                      skater_ids=set(range(1, 16)))


@pytest.mark.parametrize("candidate", ["onice_xga", "toi_share_ownxg", "naive"])
def test_candidate_uses_only_given_games(candidate):
    data = _data()
    only_g1 = fa_candidate(candidate, data, [1]).set_index("player_id")
    trimmed = SeasonData(**{**data.__dict__,
                            "game_data": data.game_data[data.game_data.GameID == 1],
                            "shots": data.shots[data.shots.game_id == 1],
                            "shifts": data.shifts[data.shifts.game_id == 1]})
    same = fa_candidate(candidate, trimmed, [1]).set_index("player_id")
    pd.testing.assert_frame_equal(only_g1.sort_index(), same.sort_index())
    assert 6 not in only_g1.index          # player 6 only plays game 2


def test_onice_credits_actual_defenders_and_conserves_total():
    fa = fa_candidate("onice_xga", _data(), [1, 2]).set_index("player_id")
    # away skaters 11-15 defend against .10 + .40; home 1-5 defend against .20 (game 1 only)
    assert fa.loc[11, "xGA"] == pytest.approx(0.50)
    assert fa.loc[1, "xGA"] == pytest.approx(0.20)
    assert fa.loc[6, "xGA"] == pytest.approx(0.0)
    assert fa["xGA"].sum() == pytest.approx(5 * 0.70)


def test_toi_share_control_same_total_spread_by_ice_time():
    data = _data()
    # make player 1 play half the minutes of the others in game 1
    data.game_data.loc[(data.game_data.GameID == 1) & (data.game_data.PlayerID == 1), "TOI"] = 10.0
    fa = fa_candidate("toi_share_ownxg", data, [1]).set_index("player_id")
    # home team xGA in game 1 = .20 split by TOI: player 1 has 10 of 90 minutes
    assert fa.loc[1, "xGA"] == pytest.approx(0.20 * 10 / 90)
    assert fa.loc[2, "xGA"] == pytest.approx(0.20 * 20 / 90)
    assert fa["xGA"].sum() == pytest.approx(0.10 + 0.20)      # both teams' xGA, nothing created or lost


def test_naive_gives_team_rate_times_minutes():
    fa = fa_candidate("naive", _data(), [1]).set_index("player_id")
    assert fa.loc[1, "xGA"] == pytest.approx(fa.loc[2, "xGA"])
    assert fa.loc[11, "xGA"] == pytest.approx(0.10 / 5)


# -------------------------------------------------------- season to season
def test_season_to_season_applies_minutes_cut_in_both_seasons():
    mk = lambda ids, toi: pd.DataFrame({"player_id": ids, "toi_5v5": toi, "d_full": [float(i) for i in ids]})
    full = {"2023-24": mk([1, 2, 3], [150, 150, 50]), "2024-25": mk([1, 2, 3], [150, 50, 150]),
            "2025-26": mk([1], [150])}
    rows = season_to_season_rows(full, min_toi=100)
    assert sorted(rows["player_id"]) == [1, 1]            # player 1 passes in all; 2 and 3 fail once
    assert set(rows["transition"]) == {"2023-24>2024-25", "2024-25>2025-26"}


# ------------------------------------------------------------------ bootstrap
def _rows(seed, n=60):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame({"key": np.arange(n), "player_id": np.arange(n), "x": x, "y": x + rng.normal(size=n)})


def test_bootstrap_diff_zero_for_identical_candidates():
    r = _rows(1)
    out = bootstrap_diff(r, r.copy(), n_boot=200, seed=42)
    assert out["diff"] == pytest.approx(0.0)
    assert out["lo"] == pytest.approx(0.0) and out["hi"] == pytest.approx(0.0)


def test_bootstrap_diff_deterministic_and_detects_better_candidate():
    a = _rows(1)
    b = a.assign(y=np.random.default_rng(7).normal(size=len(a)))     # b has no signal
    o1 = bootstrap_diff(a, b, n_boot=300, seed=42)
    o2 = bootstrap_diff(a, b, n_boot=300, seed=42)
    assert o1 == o2
    assert o1["diff"] > 0 and o1["lo"] > 0


def test_spearman_brown():
    assert spearman_brown(0.5) == pytest.approx(2 * 0.5 / 1.5)
