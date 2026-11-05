"""
test_onice_shifts.py
--------------------
Rebuilding the skaters on the ice from shift records (pwhl_war/onice.py).

Shift columns start_s / end_s are CLOCK-REMAINING seconds in the period;
events carry elapsed time (time_of_period "mm:ss"). A shift with
start_s=600, end_s=540 covers elapsed (600, 660]: a player is on the ice at
the second the shift ends (the goal-boundary case) and off at the second it
starts.

Behaviours covered:
 1. Shift ending at the event second -> player on the ice.
 2. Shift starting at the event second -> player not yet on the ice.
 3. Goalies (ids not in skater_ids) are excluded from skater sets.
 4. Shifts with NaN end_s are ignored and counted.
 5. Home / away split follows the shift's `home` flag.
 6. Events outside periods 1-3 are flagged rebuild_ok=False.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from pwhl_war.onice import attach_onice, skaters_on_ice


def _shifts(rows):
    base = {"game_id": 1, "period": 1, "home": 1, "start_s": 600, "end_s": 540}
    return pd.DataFrame([{**base, **r} for r in rows])


def _events(*times, period="1"):
    return pd.DataFrame({
        "game_id": 1,
        "period_of_game": [period] * len(times),
        "time_of_period": list(times),
    })


def test_player_on_ice_when_shift_ends_at_event_second():
    shifts = _shifts([{"player_id": 7}])
    # event at elapsed 11:00 = 660 s -> clock remaining 540 = end_s
    assert skaters_on_ice(shifts, 1, 1, 540)["home"] == {7}


def test_player_off_ice_when_shift_starts_at_event_second():
    shifts = _shifts([{"player_id": 7}])
    # event at elapsed 10:00 = 600 s -> clock remaining 600 = start_s
    assert skaters_on_ice(shifts, 1, 1, 600)["home"] == set()


def test_player_on_ice_strictly_inside_shift():
    shifts = _shifts([{"player_id": 7}])
    assert skaters_on_ice(shifts, 1, 1, 570)["home"] == {7}


def test_goalies_excluded_from_skater_sets():
    shifts = _shifts([{"player_id": 7}, {"player_id": 99}])
    out = skaters_on_ice(shifts, 1, 1, 570, skater_ids={7})
    assert out["home"] == {7}


def test_nan_end_s_shift_is_dropped_and_counted():
    shifts = _shifts([{"player_id": 7}, {"player_id": 8, "end_s": float("nan")}])
    assert skaters_on_ice(shifts, 1, 1, 570)["home"] == {7}
    events = _events("10:30")
    out = attach_onice(events, shifts)
    assert out.attrs["shifts_dropped_nan_end"] == 1


def test_home_and_away_split():
    shifts = _shifts([{"player_id": 7, "home": 1}, {"player_id": 8, "home": 0}])
    out = skaters_on_ice(shifts, 1, 1, 570)
    assert out == {"home": {7}, "away": {8}}


def test_other_game_or_period_ignored():
    shifts = _shifts([{"player_id": 7, "game_id": 2}, {"player_id": 8, "period": 2}])
    assert skaters_on_ice(shifts, 1, 1, 570) == {"home": set(), "away": set()}


def test_attach_onice_columns_and_boundaries():
    shifts = _shifts([{"player_id": 7}, {"player_id": 8, "home": 0}])
    events = _events("10:00", "10:30", "11:00", "11:01")
    out = attach_onice(events, shifts)
    assert list(out["n_home"]) == [0, 1, 1, 0]
    assert list(out["n_away"]) == [0, 1, 1, 0]
    assert out.loc[1, "home_skaters"] == (7,)
    assert out.loc[2, "away_skaters"] == (8,)
    assert list(out["rebuild_ok"]) == [False, True, True, False]


def test_overtime_period_flagged_not_ok():
    shifts = _shifts([{"player_id": 7, "period": 4}])
    out = attach_onice(_events("10:30", period="OT1"), shifts)
    assert not out.loc[0, "rebuild_ok"]
    out4 = attach_onice(_events("10:30", period="4"), shifts)
    assert not out4.loc[0, "rebuild_ok"]
