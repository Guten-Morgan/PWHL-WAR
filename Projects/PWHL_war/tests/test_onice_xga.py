"""
test_onice_xga.py
-----------------
Per-player on-ice xGA (Option 1) from rebuilt on-ice sets.

Behaviours covered:
 1. The xG model returns probabilities in [0, 1] and never uses outcome-leaking fields.
 2. Each 5v5 shot's xG is credited to all five defending skaters (total = 5 x xG).
 3. Shots that are not 5v5 by the rebuild (or not rebuild_ok) are not credited.
 4. five_on_five_minutes counts only seconds with exactly five skaters per side.
 5. build_fa_frame returns one row per player_id; traded players are summed.
 6. Players with ice time but no shots against appear with xGA = 0.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pwhl_war.onice import (
    LEAKY_FIELDS, build_fa_frame, credit_xga, five_on_five_minutes, fit_onice_xg, score_shots,
)


def _train_shots(n=400, seed=3):
    rng = np.random.default_rng(seed)
    dist = rng.uniform(3, 70, n)
    ang = rng.uniform(0, 80, n)
    p = np.clip(0.30 - dist / 300, 0.02, 0.3)
    return pd.DataFrame({
        "shot_distance": dist, "shot_angle": ang,
        "event_type": rng.choice(["Wrist", "Snap", "Slap", "Tip", "Backhand", "Default"], n),
        "power_play": rng.choice([None, "1", "0"], n), "short_handed": rng.choice([None, "0", "1"], n),
        "goal": rng.random(n) < p,
        # leaky / outcome fields present in the real data; must not become features
        "shot_quality": np.where(rng.random(n) < p, "Quality goal", "Non quality on net"),
        "scoring_chance": rng.random(n) < 0.3,
    })


def test_xg_probabilities_in_unit_interval_and_no_outcome_columns():
    shots = _train_shots()
    model = fit_onice_xg(shots)
    xg = score_shots(model, shots)
    assert len(xg) == len(shots)
    assert ((xg >= 0) & (xg <= 1)).all()
    assert not set(LEAKY_FIELDS) & set(model.feature_names)
    for leaky in ("shot_quality", "goal", "scoring_chance"):
        assert leaky in LEAKY_FIELDS
        assert not any(leaky in f for f in model.feature_names)


def test_closer_shots_score_higher():
    shots = _train_shots(n=2000)
    model = fit_onice_xg(shots)
    near = shots.iloc[[0]].assign(shot_distance=5.0, shot_angle=10.0)
    far = near.assign(shot_distance=60.0)
    assert score_shots(model, near).iloc[0] > score_shots(model, far).iloc[0]


def _shot_row(home_shooting=True, xg=0.10, n_home=5, n_away=5, ok=True,
              home=(1, 2, 3, 4, 5), away=(11, 12, 13, 14, 15)):
    return {"team_id": 1 if home_shooting else 2, "home_team_id": 1, "xg": xg,
            "home_skaters": tuple(home), "away_skaters": tuple(away),
            "n_home": n_home, "n_away": n_away, "rebuild_ok": ok}


def test_player_xga_sums_to_defending_team_xg_times_five():
    shots = pd.DataFrame([_shot_row(home_shooting=True, xg=0.10)])
    xga = credit_xga(shots)
    assert set(xga.index) == {11, 12, 13, 14, 15}
    assert xga.sum() == pytest.approx(0.50)
    assert np.allclose(xga.to_numpy(), 0.10)


def test_away_shot_credits_home_defenders():
    shots = pd.DataFrame([_shot_row(home_shooting=False, xg=0.20)])
    assert set(credit_xga(shots).index) == {1, 2, 3, 4, 5}


def test_non_5v5_or_unrebuilt_shots_not_credited():
    shots = pd.DataFrame([
        _shot_row(n_home=5, n_away=4, away=(11, 12, 13, 14)),
        _shot_row(n_home=6, home=(1, 2, 3, 4, 5, 6)),
        _shot_row(ok=False),
    ])
    assert credit_xga(shots).empty


def _shift(pid, home, start_s, end_s, game=1, period=1):
    return {"game_id": game, "period": period, "player_id": pid, "home": home,
            "start_s": start_s, "end_s": end_s}


def test_five_on_five_minutes_counts_only_exact_5v5_seconds():
    rows = [_shift(p, 1, 600, 540) for p in (1, 2, 3, 4, 5)]
    rows += [_shift(p, 0, 600, 540) for p in (11, 12, 13, 14, 15)]
    mins = five_on_five_minutes(pd.DataFrame(rows))
    assert mins.loc[1] == pytest.approx(1.0)       # 60 s
    assert mins.loc[15] == pytest.approx(1.0)


def test_extra_skater_overlap_excluded_from_5v5_minutes():
    rows = [_shift(p, 1, 600, 540) for p in (1, 2, 3, 4, 5)]
    rows += [_shift(p, 0, 600, 540) for p in (11, 12, 13, 14, 15)]
    rows += [_shift(6, 1, 570, 540)]               # 6th home skater for the last 30 s
    mins = five_on_five_minutes(pd.DataFrame(rows))
    assert mins.loc[1] == pytest.approx(0.5)       # only the first 30 s are 5v5
    assert mins.loc[6] == 0 if 6 in mins.index else True


def test_nan_end_shift_ignored_in_minutes():
    rows = [_shift(p, 1, 600, 540) for p in (1, 2, 3, 4, 5)]
    rows += [_shift(p, 0, 600, 540) for p in (11, 12, 13, 14, 15)]
    rows += [{**_shift(99, 1, 600, np.nan)}]
    mins = five_on_five_minutes(pd.DataFrame(rows))
    assert 99 not in mins.index
    assert mins.loc[1] == pytest.approx(1.0)


def test_fa_df_one_row_per_player_with_traded_player_collapsed():
    # player 11 defends for two different teams in two games
    xga = pd.Series({11: 0.30, 12: 0.10})
    toi = pd.Series({11: 12.0, 12: 6.0})
    shots = pd.DataFrame([_shot_row(home_shooting=True, xg=0.20, away=(11, 12, 13, 14, 15)),
                          _shot_row(home_shooting=False, xg=0.10, home=(11, 22, 23, 24, 25))])
    credited = credit_xga(shots)
    assert credited.loc[11] == pytest.approx(0.30)
    fa = build_fa_frame(credited, pd.Series({11: 12.0, 12: 6.0, 13: 3.0, 14: 3.0, 15: 3.0, 22: 3.0, 23: 3.0, 24: 3.0, 25: 3.0}))
    assert fa["player_id"].is_unique
    assert list(fa.columns) == ["player_id", "xGA", "toi_min"]
    assert fa.set_index("player_id").loc[11, "xGA"] == pytest.approx(0.30)


def test_player_with_ice_time_but_no_shots_has_zero_xga():
    fa = build_fa_frame(pd.Series({11: 0.5}), pd.Series({11: 10.0, 99: 4.0}))
    row = fa.set_index("player_id").loc[99]
    assert row["xGA"] == 0.0 and row["toi_min"] == 4.0
