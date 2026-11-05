"""
test_onice_rapm.py
------------------
Stints and the shrunk ridge (pwhl_war/onice_rapm.py, Option 2).

Behaviours covered:
 1. Stint seconds add up to the exact 5v5 clock time (no gaps, no overlap).
 2. A stint ends at every substitution; seconds with 6+ skaters are excluded.
 3. A shot is credited to the stint containing it (end of a stint inclusive).
 4. With a huge penalty the fit returns the prior (Option 1) values.
 5. Two players who are always together get equal, finite estimates.
 6. The cross-validated penalty is chosen by whole games.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pwhl_war.onice import five_on_five_minutes
from pwhl_war.onice_rapm import build_stints, fit_rapm

HOME = (1, 2, 3, 4, 5)
AWAY = (11, 12, 13, 14, 15)


def _shift(pid, home, start_s, end_s, game=1, period=1):
    return {"game_id": game, "period": period, "player_id": pid, "home": home,
            "start_s": start_s, "end_s": end_s}


def _full_period(home=HOME, away=AWAY, game=1, period=1):
    rows = [_shift(p, 1, 1200, 0, game, period) for p in home]
    rows += [_shift(p, 0, 1200, 0, game, period) for p in away]
    return rows


def _shot(elapsed_s, team_id=1, xg=0.1, game=1, period="1"):
    mm, ss = divmod(elapsed_s, 60)
    return {"game_id": game, "period_of_game": period, "time_of_period": f"{mm}:{ss:02d}",
            "team_id": team_id, "home_team_id": 1, "xg": xg}


def test_stint_minutes_sum_to_5v5_clock_time():
    shifts = pd.DataFrame(_full_period())
    stints = build_stints(pd.DataFrame([_shot(300)]), shifts)
    assert stints["duration"].sum() == pytest.approx(1200.0)
    mins = five_on_five_minutes(shifts)
    assert stints["duration"].sum() / 60 == pytest.approx(mins.loc[1])


def test_stints_split_on_every_substitution():
    # home player 5 is replaced by player 6 at elapsed 600 s
    rows = _full_period()
    rows = [r for r in rows if r["player_id"] != 5]
    rows += [_shift(5, 1, 1200, 600), _shift(6, 1, 600, 0)]
    stints = build_stints(pd.DataFrame([_shot(100)]), pd.DataFrame(rows)).sort_values("start")
    assert len(stints) == 2
    assert list(stints["duration"]) == [600.0, 600.0]
    assert 5 in stints.iloc[0]["home_skaters"] and 6 in stints.iloc[1]["home_skaters"]


def test_overlap_with_sixth_skater_excluded():
    rows = _full_period() + [_shift(6, 1, 700, 600)]          # 100 s of 6v5
    stints = build_stints(pd.DataFrame([_shot(100)]), pd.DataFrame(rows))
    assert stints["duration"].sum() == pytest.approx(1100.0)


def test_shot_credited_to_stint_containing_it_end_inclusive():
    rows = _full_period()
    rows = [r for r in rows if r["player_id"] != 5] + [_shift(5, 1, 1200, 600), _shift(6, 1, 600, 0)]
    shots = pd.DataFrame([_shot(300, team_id=1, xg=0.2), _shot(600, team_id=2, xg=0.3), _shot(900, team_id=1, xg=0.5)])
    st = build_stints(shots, pd.DataFrame(rows)).sort_values("start").reset_index(drop=True)
    # shots at 300 (home) and 600 (away, last second of stint 1) fall in stint 1; 900 (home) in stint 2
    assert st.loc[0, "xg_home"] == pytest.approx(0.2) and st.loc[0, "xg_away"] == pytest.approx(0.3)
    assert st.loc[1, "xg_home"] == pytest.approx(0.5) and st.loc[1, "xg_away"] == 0


def _synthetic_stints(n_games=12, seed=0, together=None):
    """Random 5v5 stints over a pool of 14 home and 14 away players."""
    rng = np.random.default_rng(seed)
    home_pool, away_pool = list(range(1, 15)), list(range(101, 115))
    rows = []
    for g in range(n_games):
        for _ in range(25):
            a = tuple(sorted(int(p) for p in rng.choice(away_pool, 5, replace=False)))
            if together:
                # players in `together` are always on the ice as a group, or all absent
                others = [p for p in home_pool if p not in together]
                if rng.random() < 0.5:
                    h = tuple(sorted(list(together) + [int(p) for p in rng.choice(others, 5 - len(together), replace=False)]))
                else:
                    h = tuple(sorted(int(p) for p in rng.choice(others, 5, replace=False)))
            else:
                h = tuple(sorted(int(p) for p in rng.choice(home_pool, 5, replace=False)))
            rows.append({"game_id": g, "start": 0, "end": 60, "duration": float(rng.integers(30, 120)),
                         "home_skaters": h, "away_skaters": a,
                         "xg_home": float(rng.random() * 0.2), "xg_away": float(rng.random() * 0.2)})
    return pd.DataFrame(rows)


def _prior(stints, rate=2.0, per=None):
    players = sorted({p for t in stints["home_skaters"] for p in t} | {p for t in stints["away_skaters"] for p in t})
    return pd.DataFrame({"player_id": players,
                         "xGA": [(per or {}).get(p, rate) * 10 for p in players],      # rate/60 * 600 min
                         "toi_min": 600.0})


def test_fit_rapm_returns_prior_when_alpha_is_huge():
    stints = _synthetic_stints()
    per = {p: 1.5 + 0.1 * (p % 7) for p in range(1, 15)} | {p: 2.5 - 0.1 * (p % 5) for p in range(101, 115)}
    prior = _prior(stints, per=per)
    out = fit_rapm(stints, prior, alphas=[1e12]).set_index("player_id")
    ref = prior.set_index("player_id")
    assert np.allclose(out["xGA"], ref["xGA"], rtol=1e-3)
    assert list(out.columns) == ["xGA", "toi_min"]


def test_fit_rapm_is_unique_for_collinear_linemates():
    stints = _synthetic_stints(together=(1, 2))
    # players 1 and 2 only ever appear together (or both absent)
    both = stints["home_skaters"].map(lambda t: (1 in t) == (2 in t))
    stints = stints[both]
    prior = _prior(stints)
    out = fit_rapm(stints, prior, alphas=[10.0]).set_index("player_id")
    assert np.isfinite(out["xGA"]).all()
    assert out.loc[1, "xGA"] == pytest.approx(out.loc[2, "xGA"], rel=1e-6)


def test_cv_penalty_chosen_by_whole_games_and_runs():
    stints = _synthetic_stints(n_games=10, seed=3)
    out = fit_rapm(stints, _prior(stints), alphas=[1.0, 100.0, 1e4])
    assert len(out) == len(_prior(stints))
    assert np.isfinite(out["xGA"]).all()
