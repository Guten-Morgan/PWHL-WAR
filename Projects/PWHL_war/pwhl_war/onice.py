"""
onice.py
--------
On-ice defensive data from the sportsdataverse `fastRhockey-pwhl-data` repo:
play-by-play, shift records and skater box scores, one parquet per season.

Files are suffixed with the season END year (2025-26 -> 2026). They are
downloaded once into pwhl_war/data/raw/sdv/ (git-ignored) and treated as
untrusted data: only read as parquet, columns checked before use.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BASE_URL = "https://raw.githubusercontent.com/sportsdataverse/fastRhockey-pwhl-data/main/pwhl"
SDV_DIR  = Path(__file__).parent / "data" / "raw" / "sdv"

_FILES = {
    "pbp":    ("pbp/parquet",        "play_by_play_{y}.parquet"),
    "shifts": ("shifts/parquet",     "shifts_{y}.parquet"),
    "box":    ("skater_box/parquet", "skater_box_{y}.parquet"),
}

_ON_ICE_CREDIT = [f"{side}_player_{n}_id" for side in ("plus", "minus")
                  for n in ("one", "two", "three", "four", "five")]

REQUIRED_PBP = [
    "game_id", "event", "period_of_game", "time_of_period", "goal", "player_id",
    "goalie_id", "team_id", "home_team_id", "away_team_id", "game_date",
    "shot_distance", "shot_angle", "event_type", "empty_net", "power_play",
    "short_handed", "strength_state", "strength_state_valid",
] + _ON_ICE_CREDIT
REQUIRED_SHIFTS = ["game_id", "player_id", "home", "period", "start_s", "end_s"]
REQUIRED_BOX    = ["game_id", "player_id", "position", "team_id", "toi"]

_REQUIRED = {"pbp": REQUIRED_PBP, "shifts": REQUIRED_SHIFTS, "box": REQUIRED_BOX}

# A goal appears twice in the play-by-play: a `shot` row with goal=True and a
# `goal` row. They share this key.
_GOAL_KEY = ["game_id", "period_of_game", "time_of_period", "player_id"]


def file_year(season: str) -> int:
    """Season label "2025-26" -> file suffix 2026."""
    try:
        start, end = season.split("-")
        return int(start) + 1 if len(end) == 2 else int(end)
    except ValueError as exc:
        raise ValueError(f"season must look like '2025-26', got {season!r}") from exc


def _cached_path(kind: str, year: int, cache_dir: Path, refresh: bool) -> Path:
    folder, pattern = _FILES[kind]
    name = pattern.format(y=year)
    path = Path(cache_dir) / name
    if refresh or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{BASE_URL}/{folder}/{name}"
        log.info("Downloading %s", url)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    return path


def load_season(season: str, cache_dir: Path = SDV_DIR, refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Return {"pbp", "shifts", "box"} for a season, downloading once into cache_dir.

    Raises ValueError naming any required column that is missing.
    """
    year = file_year(season)
    out: dict[str, pd.DataFrame] = {}
    for kind in _FILES:
        df = pd.read_parquet(_cached_path(kind, year, cache_dir, refresh))
        missing = [c for c in _REQUIRED[kind] if c not in df.columns]
        if missing:
            raise ValueError(f"{kind} {season}: missing required column(s): {', '.join(missing)}")
        out[kind] = df
    return out


PERIOD_SECONDS = 1200   # regulation periods 1-3; overtime is not rebuilt


def _usable(shifts: pd.DataFrame, skater_ids) -> pd.DataFrame:
    """Shifts with a known end, restricted to skaters when ids are given."""
    out = shifts.dropna(subset=["end_s"])
    if skater_ids is not None:
        out = out[out["player_id"].isin(skater_ids)]
    return out


def skaters_on_ice(shifts: pd.DataFrame, game_id: int, period: int, clock: float,
                   skater_ids=None) -> dict[str, set[int]]:
    """Skaters on the ice at one moment.

    `clock` is seconds REMAINING in the period. A shift (start_s, end_s are
    clock-remaining, start_s > end_s) covers the player when
    end_s <= clock < start_s: on the ice at the second the shift ends (the
    shift feed ends shifts at the goal's clock second), off at the second it
    starts. Shifts with NaN end_s are ignored. When `skater_ids` is given,
    other ids (goalies) are excluded.
    """
    s = _usable(shifts, skater_ids)
    s = s[(s["game_id"] == game_id) & (s["period"] == period)
          & (s["end_s"] <= clock) & (clock < s["start_s"])]
    return {
        "home": {int(p) for p in s.loc[s["home"] == 1, "player_id"]},
        "away": {int(p) for p in s.loc[s["home"] != 1, "player_id"]},
    }


def _elapsed_seconds(time_of_period: pd.Series) -> pd.Series:
    """"mm:ss" elapsed time in the period -> seconds (NaN when unparseable)."""
    parts = time_of_period.astype("string").str.extract(r"^\s*(\d+):(\d{2})\s*$")
    return pd.to_numeric(parts[0]) * 60 + pd.to_numeric(parts[1])


def attach_onice(events: pd.DataFrame, shifts: pd.DataFrame, skater_ids=None) -> pd.DataFrame:
    """Add home_skaters / away_skaters (sorted tuples), n_home, n_away, rebuild_ok.

    Events carry elapsed time in `time_of_period` and the period in
    `period_of_game` (a string; "OT1" etc. are not numeric). Only periods 1-3
    are rebuilt; others get empty sets and rebuild_ok=False. rebuild_ok is also
    False when either side has no skater on the ice. The number of shifts
    dropped for a missing end is stored in attrs["shifts_dropped_nan_end"].
    """
    out = events.copy()
    period = pd.to_numeric(out["period_of_game"], errors="coerce")
    clock = PERIOD_SECONDS - _elapsed_seconds(out["time_of_period"])
    regulation = period.isin([1, 2, 3]) & clock.notna()

    usable = _usable(shifts, skater_ids)
    by_group = {k: g for k, g in usable.groupby(["game_id", "period"])}

    home_sets, away_sets = [], []
    for gid, per, clk, ok in zip(out["game_id"], period, clock, regulation):
        g = by_group.get((gid, per)) if ok else None
        if g is None:
            home_sets.append(()); away_sets.append(())
            continue
        on = g[(g["end_s"] <= clk) & (clk < g["start_s"])]
        home_sets.append(tuple(sorted(int(p) for p in on.loc[on["home"] == 1, "player_id"])))
        away_sets.append(tuple(sorted(int(p) for p in on.loc[on["home"] != 1, "player_id"])))

    out["home_skaters"] = home_sets
    out["away_skaters"] = away_sets
    out["n_home"] = [len(t) for t in home_sets]
    out["n_away"] = [len(t) for t in away_sets]
    out["rebuild_ok"] = regulation.to_numpy() & (out["n_home"] > 0).to_numpy() & (out["n_away"] > 0).to_numpy()
    out.attrs["shifts_dropped_nan_end"] = int(shifts["end_s"].isna().sum())
    return out


def dedupe_goals(pbp: pd.DataFrame) -> pd.DataFrame:
    """One event per goal.

    Drops each `goal` row that has a matching `shot` row with goal=True (same
    game, period, clock and player). A `goal` row with no shot twin is kept.
    Rows of every other kind are untouched.
    """
    is_shot_goal = (pbp["event"] == "shot") & (pbp["goal"].astype(bool))
    shot_keys = pd.MultiIndex.from_frame(pbp.loc[is_shot_goal, _GOAL_KEY])
    row_keys = pd.MultiIndex.from_frame(pbp[_GOAL_KEY])
    is_twin = (pbp["event"] == "goal") & row_keys.isin(shot_keys)
    return pbp.loc[~is_twin]


# ---------------------------------------------------------------------------
# Option 1: per-player on-ice xGA at five-on-five
# ---------------------------------------------------------------------------

# Fields that encode or leak the outcome; never model features.
LEAKY_FIELDS = ("shot_quality", "goal", "scoring_chance")

_XG_TYPES = ("Wrist", "Snap", "Slap", "Tip", "Backhand", "Default")


class OnIceXG:
    """Logistic regression on shot geometry, shot type and special-teams flags.

    Same family as CoordXGModel (log distance, angle, interaction) but fit on
    this dataset's own `shot_distance` / `shot_angle`, since the hockey-statistics
    API xG has no on-ice players to join to.
    """

    def __init__(self):
        self.feature_names = (["log_dist", "angle", "is_pp", "is_sh", "log_dist_x_angle"]
                              + [f"type_{t}" for t in _XG_TYPES])
        self._model = None

    def _features(self, shots: pd.DataFrame) -> pd.DataFrame:
        log_dist = np.log(shots["shot_distance"].astype(float).clip(lower=1.0))
        angle = shots["shot_angle"].astype(float) / 90.0
        def flag(col, value):
            # a missing flag means "not on a power play / not that shot type", not unknown
            return (shots[col].astype("string") == value).fillna(False).astype(float)

        X = pd.DataFrame({
            "log_dist": log_dist,
            "angle": angle,
            "is_pp": flag("power_play", "1"),
            "is_sh": flag("short_handed", "1"),
            "log_dist_x_angle": log_dist * angle,
        }, index=shots.index)
        for t in _XG_TYPES:
            X[f"type_{t}"] = flag("event_type", t)
        return X[self.feature_names]

    def fit(self, shots: pd.DataFrame) -> "OnIceXG":
        from sklearn.linear_model import LogisticRegression
        X = self._features(shots)
        keep = X.notna().all(axis=1)
        self._model = LogisticRegression(max_iter=1000).fit(X[keep], shots.loc[keep, "goal"].astype(int))
        return self

    def predict(self, shots: pd.DataFrame) -> pd.Series:
        X = self._features(shots)
        ok = X.notna().all(axis=1)
        out = pd.Series(np.nan, index=shots.index, name="xg")
        out[ok] = self._model.predict_proba(X[ok])[:, 1]
        return out


def fit_onice_xg(train_shots: pd.DataFrame) -> OnIceXG:
    """Fit the xG model on shots on goal (columns: shot_distance, shot_angle,
    event_type, power_play, short_handed, goal)."""
    return OnIceXG().fit(train_shots)


def score_shots(model: OnIceXG, shots: pd.DataFrame) -> pd.Series:
    """xG per shot, in [0, 1]; NaN where distance or angle is missing."""
    return model.predict(shots)


def credit_xga(shots: pd.DataFrame) -> pd.Series:
    """Credit each 5v5 shot's xG to all five defending skaters.

    `shots` needs team_id (shooting team), home_team_id, xg, home_skaters,
    away_skaters, n_home, n_away, rebuild_ok. Shots that are not rebuild_ok, not
    exactly five skaters per side, or have no xG are not credited.
    """
    ok = (shots["rebuild_ok"] & (shots["n_home"] == 5) & (shots["n_away"] == 5)
          & shots["xg"].notna())
    totals: dict[int, float] = {}
    for _, r in shots[ok].iterrows():
        defenders = r["away_skaters"] if r["team_id"] == r["home_team_id"] else r["home_skaters"]
        for pid in defenders:
            totals[pid] = totals.get(pid, 0.0) + float(r["xg"])
    return pd.Series(totals, dtype=float, name="xGA").sort_index()


def five_on_five_minutes(shifts: pd.DataFrame, skater_ids=None) -> pd.Series:
    """Minutes each skater was on the ice with exactly five skaters per side.

    Periods 1-3 only; shifts with NaN end_s are ignored. Intervals follow the
    same convention as attach_onice: a shift covers elapsed (P - start_s, P - end_s].
    """
    s = _usable(shifts, skater_ids)
    s = s[s["period"].isin([1, 2, 3])]
    mins: dict[int, float] = {}
    for _, g in s.groupby(["game_id", "period"]):
        lo = (PERIOD_SECONDS - g["start_s"]).to_numpy(float)
        hi = (PERIOD_SECONDS - g["end_s"]).to_numpy(float)
        home = (g["home"] == 1).to_numpy()
        pid = g["player_id"].to_numpy(int)
        pts = np.unique(np.concatenate([lo, hi]))
        for a, b in zip(pts[:-1], pts[1:]):
            on = (lo <= a) & (hi >= b)
            if on[home].sum() == 5 and on[~home].sum() == 5:
                for p in pid[on]:
                    mins[p] = mins.get(p, 0.0) + (b - a) / 60.0
    return pd.Series(mins, dtype=float, name="toi_min").sort_index()


def build_fa_frame(xga: pd.Series, toi: pd.Series) -> pd.DataFrame:
    """[player_id, xGA, toi_min], one row per player. Anyone with ice time but no
    shots against gets xGA 0; a credited player with no ice time gets toi 0."""
    xga = xga.groupby(level=0).sum()
    toi = toi.groupby(level=0).sum()
    idx = toi.index.union(xga.index)
    out = pd.DataFrame({
        "player_id": idx.astype(int),
        "xGA": xga.reindex(idx).fillna(0.0).to_numpy(),
        "toi_min": toi.reindex(idx).fillna(0.0).to_numpy(),
    })
    return out.reset_index(drop=True)


def load_schedule(cache_dir: Path = SDV_DIR, refresh: bool = False) -> pd.DataFrame:
    """The schedule master (has game_type), downloaded once into cache_dir."""
    path = Path(cache_dir) / "schedule_master.parquet"
    if refresh or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(f"{BASE_URL}/pwhl_schedule_master.parquet", tmp)
        tmp.replace(path)
    return pd.read_parquet(path)


def _regulation_shots(pbp: pd.DataFrame, reg_ids) -> pd.DataFrame:
    """Shots on goal in regular-season periods 1-3, goals de-duplicated."""
    p = dedupe_goals(pbp)
    return p[(p["event"] == "shot") & p["game_id"].isin(set(reg_ids))
             & p["period_of_game"].isin(["1", "2", "3"])].copy()


def build_fa_onice(season: str, train_seasons: list[str], cache_dir: Path = SDV_DIR) -> pd.DataFrame:
    """Drop-in fa_df [player_id, xGA, toi_min] from rebuilt on-ice sets.

    Regular season, periods 1-3, five-on-five by the rebuild (exactly five
    skaters per side). Diagnostics are stored in `.attrs["diagnostics"]`.
    """
    from .scope import regular_season_game_ids
    sched = load_schedule(cache_dir)

    train = []
    for t in train_seasons:
        train.append(_regulation_shots(load_season(t, cache_dir)["pbp"], regular_season_game_ids(sched, t)))
    model = fit_onice_xg(pd.concat(train, ignore_index=True))

    reg = regular_season_game_ids(sched, season)
    d = load_season(season, cache_dir)
    skaters = set(d["box"]["player_id"].astype(int))
    shifts = d["shifts"][d["shifts"]["game_id"].isin(set(reg))]

    shots = _regulation_shots(d["pbp"], reg)
    shots = attach_onice(shots, shifts, skaters)
    shots["xg"] = score_shots(model, shots)
    xga = credit_xga(shots)
    toi = five_on_five_minutes(shifts, skaters)

    rebuilt_5v5 = shots["rebuild_ok"] & (shots["n_home"] == 5) & (shots["n_away"] == 5)
    shipped_5v5 = (shots["strength_state"] == "5v5") & (shots["strength_state_valid"] == True)  # noqa: E712
    fa = build_fa_frame(xga, toi)
    fa.attrs["diagnostics"] = {
        "shots": len(shots),
        "rebuilt_5v5": int(rebuilt_5v5.sum()),
        "shipped_5v5": int(shipped_5v5.sum()),
        "shipped_5v5_also_rebuilt_5v5": int((shipped_5v5 & rebuilt_5v5).sum()),
        "shots_without_xg": int(shots["xg"].isna().sum()),
        "shifts_dropped_nan_end": shots.attrs.get("shifts_dropped_nan_end", 0),
        "xg_5v5_total": float(shots.loc[rebuilt_5v5, "xg"].sum()),
    }
    return fa
