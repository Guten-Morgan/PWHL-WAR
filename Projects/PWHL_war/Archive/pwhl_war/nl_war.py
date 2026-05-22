"""
nl_war.py
---------
Non-linear WAR model for PWHL skaters -- parallel development track.

This module is the counterpart to box_war.py (the linear model that powers
the public website). box_war.py is NEVER modified by this code. If this
model beats all three Exp-5 baselines, it may replace box_war.py; until
then it is an experimental track only.

Feature set confirmed by Phase 2 RF importance screen:
  [o_xG60, A1_60, A2_60, PIM60, is_D, ozs_pct]

Note: PP_ixG60 and EV_Shots60 were deferred -- these columns are not
available across all three seasons in the WAR CSVs (granular splits only
exist in game_data_2526.csv for 2025-26).

Phase 3:  data loading, feature engineering, validation harness (plumbing)
Phase 4:  NLWarModel class -- Expanded Feature Ridge with RidgeCV + StandardScaler
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from .constants import GPW as _GPW_CONSTANTS

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLS = ["o_xG60", "A1_60", "A2_60", "PIM60", "is_D", "ozs_pct"]
"""Feature set for the Expanded Feature Ridge (Phase 4).
PP_ixG60 / EV_Shots60 deferred: not available across all seasons in WAR CSVs."""

TARGET_COL = "oGAR"
"""Offensive goals above replacement -- target for Phase 4 Ridge supervision."""

_COL_MAP = {
    # war_2526.csv uses lower-case / abbreviated names; normalise to 2324/2425 schema
    "name": "Name",
    "player_id": "PlayerID",
    "team": "Team",
    "pos": "position",
    "gp": "GP",
}

_D_POSITIONS = {"LD", "RD", "D"}


def _normalize_pos(p: str) -> str:
    """Map granular position codes to F or D."""
    return "D" if str(p).upper() in _D_POSITIONS else "F"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def pool_seasons(
    csv_paths: dict[str, str],
    min_toi: dict[str, float],
    ozs_paths: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Load, filter, and pool per-season WAR CSVs into one DataFrame.

    Parameters
    ----------
    csv_paths : {season_label: path_to_war_csv}
                e.g. {"2023-24": ".../war_2324.csv", ...}
    min_toi   : {season_label: minimum_toi_minutes}
                e.g. {"2023-24": 100.0, "2024-25": 100.0, "2025-26": 50.0}
    ozs_paths : {season_label: path_to_ozs_csv}  (optional)
                Each CSV must have columns [player_id, ozs_pct].
                If None, ozs_pct is imputed to the season mean for all players.

    Returns
    -------
    DataFrame with columns including FEATURE_COLS, TARGET_COL, "season", "Name", "Team".
    """
    # Pre-load ozs lookups
    ozs_lookup: dict[str, dict[int, float]] = {}
    for season, path in (ozs_paths or {}).items():
        if os.path.exists(path):
            oz_df = pd.read_csv(path, encoding="utf-8")
            ozs_lookup[season] = dict(
                zip(oz_df["player_id"].astype(int), oz_df["ozs_pct"])
            )
        else:
            log.warning("ozs CSV not found for %s: %s", season, path)
            ozs_lookup[season] = {}

    frames: list[pd.DataFrame] = []
    for season, path in csv_paths.items():
        if not os.path.exists(path):
            log.warning("WAR CSV not found, skipping %s: %s", season, path)
            continue

        df = pd.read_csv(path, encoding="utf-8")
        df.rename(columns=_COL_MAP, inplace=True)
        df["season"] = season

        # Drop goalies
        if "position" in df.columns:
            df = df[df["position"] != "G"].copy()

        # TOI filter
        threshold = min_toi.get(season, 50.0)
        df = df[df["toi_min"] >= threshold].copy()

        # Normalise position to F / D only
        if "position" in df.columns:
            df["position"] = df["position"].apply(_normalize_pos)
        else:
            log.warning("No position column in %s -- is_D will be 0 for all", season)
            df["position"] = "F"

        toi = df["toi_min"].clip(lower=0.1)

        # Derive per-60 rates from totals in WAR CSV
        df["A1_60"]  = df["A1"]  / toi * 60
        df["A2_60"]  = df["A2"]  / toi * 60
        df["PIM60"]  = df["PIM"] / toi * 60
        df["is_D"]   = (df["position"] == "D").astype(int)

        # ozs_pct: join from lookup; impute season mean for missing / non-center players
        season_ozs = ozs_lookup.get(season, {})
        if season_ozs and "PlayerID" in df.columns:
            df["ozs_pct"] = df["PlayerID"].map(
                lambda pid: season_ozs.get(int(pid), np.nan)
            )
        else:
            df["ozs_pct"] = np.nan

        season_mean_ozs = df["ozs_pct"].mean()
        if np.isnan(season_mean_ozs):
            season_mean_ozs = 0.5  # global fallback
        df["ozs_pct"] = df["ozs_pct"].fillna(season_mean_ozs)

        n = len(df)
        n_imp = (df["ozs_pct"] == season_mean_ozs).sum()
        log.info(
            "%s: %d players after TOI filter | ozs_pct imputed for ~%d", season, n, n_imp
        )
        frames.append(df)

    if not frames:
        raise RuntimeError("No WAR CSVs loaded successfully -- check csv_paths.")

    pooled = pd.concat(frames, ignore_index=True)
    return pooled


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Build the offensive feature matrix for the Expanded Feature Ridge.

    Parameters
    ----------
    df : pooled player-season DataFrame from pool_seasons()

    Returns
    -------
    (X, feature_names)
    X              : ndarray, shape (N, len(FEATURE_COLS))
    feature_names  : list[str] matching column order in X
    """
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    nan_counts = df[FEATURE_COLS].isna().sum()
    if nan_counts.any():
        log.warning("NaN counts in feature matrix:\n%s", nan_counts[nan_counts > 0])

    X = df[FEATURE_COLS].values.astype(np.float64)
    return X, list(FEATURE_COLS)


# ---------------------------------------------------------------------------
# Temporal CV split
# ---------------------------------------------------------------------------

def temporal_cv_split(
    df: pd.DataFrame,
    train_seasons: list[str],
    test_season: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Split pooled DataFrame into train/test sets along season boundary.

    NEVER shuffles rows -- temporal integrity is mandatory.
    Verifies no test-season rows appear in train set.

    Parameters
    ----------
    df            : pooled DataFrame from pool_seasons()
    train_seasons : list of season labels for training (e.g. ["2023-24", "2024-25"])
    test_season   : season label for testing (e.g. "2025-26")

    Returns
    -------
    X_train, y_train, X_test, y_test, meta_test
    meta_test : DataFrame with Name, Team, season, toi_min for test-set players
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in DataFrame")

    train_mask = df["season"].isin(train_seasons)
    test_mask  = df["season"] == test_season

    # Temporal integrity check
    overlap = set(df.loc[train_mask, "season"]) & {test_season}
    if overlap:
        raise RuntimeError(f"Data leakage: test season {test_season} found in training set!")

    X_train = df.loc[train_mask, FEATURE_COLS].values.astype(np.float64)
    y_train = df.loc[train_mask, TARGET_COL].values.astype(np.float64)
    X_test  = df.loc[test_mask,  FEATURE_COLS].values.astype(np.float64)
    y_test  = df.loc[test_mask,  TARGET_COL].values.astype(np.float64)

    meta_cols = [c for c in ["Name", "Team", "season", "toi_min", "position"] if c in df.columns]
    meta_test = df.loc[test_mask, meta_cols].reset_index(drop=True)

    log.info(
        "Temporal CV split: train=%d (%s), test=%d (%s)",
        len(y_train), "+".join(train_seasons), len(y_test), test_season,
    )
    return X_train, y_train, X_test, y_test, meta_test


# ---------------------------------------------------------------------------
# Validation vs Exp-5 baseline
# ---------------------------------------------------------------------------

def validate_vs_baseline(
    war_df: pd.DataFrame,
    standings_df: pd.DataFrame,
    war_col: str = "WAR",
    team_col: str = "Team",
    points_col: str = "points",
    baseline_r: float = 0.722,
) -> dict:
    """
    Compute team-level Spearman r of summed WAR vs standings points.

    Parameters
    ----------
    war_df       : player-level DataFrame with war_col and team_col
    standings_df : team-level DataFrame with team_col and points_col
    war_col      : column name for player WAR values
    team_col     : team identifier column (must match in both DataFrames)
    points_col   : standings points column
    baseline_r   : Exp-5 baseline Spearman r to compare against (default 0.722)

    Returns
    -------
    dict with keys: spearman_r, p_value, delta_vs_baseline, pass_fail
    """
    team_war = (
        war_df.groupby(team_col)[war_col]
        .sum()
        .reset_index()
        .rename(columns={war_col: "team_WAR"})
    )
    merged = team_war.merge(standings_df[[team_col, points_col]], on=team_col, how="inner")

    if len(merged) < 4:
        log.warning(
            "Only %d teams in validation -- results unreliable (need >= 4)", len(merged)
        )

    r, p = stats.spearmanr(merged["team_WAR"], merged[points_col])
    delta = r - baseline_r
    passed = r >= baseline_r

    result = {
        "spearman_r": round(float(r), 4),
        "p_value":    round(float(p), 4),
        "n_teams":    len(merged),
        "delta_vs_baseline": round(float(delta), 4),
        "baseline_r": baseline_r,
        "pass_fail":  "PASS" if passed else "FAIL",
    }

    print(f"\n{'='*55}")
    print("TEAM SPEARMAN r: summed WAR vs standings points")
    print(f"  r = {r:+.4f}  p = {p:.4f}  N_teams = {len(merged)}")
    print(f"  Exp-5 baseline r = {baseline_r:.4f}")
    print(f"  Delta vs baseline: {delta:+.4f}")
    print(f"  Result: {result['pass_fail']}")
    print(f"{'='*55}")

    return result



# ---------------------------------------------------------------------------
# Phase 4: Expanded Feature Ridge model
# ---------------------------------------------------------------------------

class NLWarModel:
    """
    Expanded Feature Ridge WAR model (Phase 4).

    Replaces the linear model's single-feature (o_xG60) offensive scaling
    with a multi-feature RidgeCV that learns weights for the full feature
    set confirmed in Phase 2: [o_xG60, A1_60, A2_60, PIM60, is_D, ozs_pct].

    Training target: G60 (actual goals per 60 minutes).
    Using G60 -- not o_xG60 -- as the target prevents trivial collapse
    (Ridge assigning 1.0 to o_xG60 and 0 to all other features when the
    target is mechanically derived from the primary feature).  G60 is an
    external measurement that the Ridge must predict from ALL features,
    allowing A1_60, PIM60, etc. to receive meaningful regularized weights.

    Defensive WAR (dWAR) is reused unchanged from the Exp-5 linear model:
    the precomputed `dWAR` column in the WAR CSVs already reflects the
    4-step xGA60 pipeline from stats_utils.  xGA60 is not available in the
    WAR CSVs directly, so calling compute_defensive_value60() is not possible
    here; this design decision is documented as a known constraint.

    Parameters
    ----------
    alphas : list of float
        Ridge regularisation strengths for RidgeCV cross-validation.
        Defaults to the same grid used by rapm.py.
    replacement_pct : float
        Percentile of training-set Ridge predictions used as the replacement
        level.  Mirrors the linear model's DEFAULT_REPLACEMENT_PCT=25.
    """

    ALPHAS = [0.01, 0.1, 1, 10, 100, 1000, 10000]

    def __init__(
        self,
        alphas: list[float] | None = None,
        replacement_pct: float = 25.0,
    ) -> None:
        self.alphas          = alphas or self.ALPHAS
        self.replacement_pct = replacement_pct

        # Fitted attributes (set by fit())
        self.scaler_:      StandardScaler | None = None
        self.ridge_:       RidgeCV | None        = None
        self.alpha_:       float | None          = None
        self._repl_level:  float | None          = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> "NLWarModel":
        """
        Fit Ridge model on training player-season data.

        Training target: G60 (actual goals per 60 minutes).
        Using G60 avoids trivial collapse that occurs when o_xG60 is both
        a feature and the prediction target.  The Ridge learns a regularised
        linear combination of all six features that best predicts actual
        goal-scoring rate, giving A1_60, PIM60, etc. genuine coefficients.

        oGAR formula (applied in get_war):
            oGAR = (ridge_pred_G60 - repl_level) * toi_min / 60

        Parameters
        ----------
        train_df : DataFrame from pool_seasons() for training seasons.
                   Must contain FEATURE_COLS + "G" + "toi_min".
        """
        required = FEATURE_COLS + ["G", "toi_min"]
        missing = [c for c in required if c not in train_df.columns]
        if missing:
            raise ValueError(f"NLWarModel.fit(): missing columns {missing}")

        X, feat_names = build_feature_matrix(train_df)

        # Derive G60 = actual goals per 60 minutes
        toi = train_df["toi_min"].clip(lower=0.1).values
        y   = (train_df["G"].values / toi * 60.0).astype(np.float64)

        self.scaler_ = StandardScaler()
        X_s = self.scaler_.fit_transform(X)

        self.ridge_ = RidgeCV(alphas=self.alphas, cv=3)
        self.ridge_.fit(X_s, y)
        self.alpha_ = float(self.ridge_.alpha_)

        # Replacement level: pct-ile of training G60 predictions
        train_preds = self.ridge_.predict(X_s)
        self._repl_level = float(np.percentile(train_preds, self.replacement_pct))

        train_r2 = float(self.ridge_.score(X_s, y))
        coef_str = "  ".join(
            f"{n}={c:+.4f}" for n, c in zip(feat_names, self.ridge_.coef_)
        )
        log.info(
            "NLWarModel.fit(): N=%d  target=G60  alpha=%.4g  train_R2=%.4f  repl_level=%.4f",
            len(y), self.alpha_, train_r2, self._repl_level,
        )
        log.info("Ridge coefficients (scaled): %s", coef_str)
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def get_war(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict WAR for players in df.

        oGAR formula: ``(ridge_pred_G60 - repl_level) * toi_min / 60``
        dWAR:         reused from df["dWAR"] (Exp-5 values, unchanged)
        WAR:          oWAR + dWAR

        Parameters
        ----------
        df : DataFrame from pool_seasons() for any season(s).
             Must contain FEATURE_COLS + "toi_min" + "dWAR" + "goals_per_win".

        Returns
        -------
        DataFrame with columns:
          Name, PlayerID, Team, season, position, toi_min,
          g_ridge60,  -- raw Ridge G60 prediction
          oGAR, oWAR, dWAR, WAR
        """
        if self.scaler_ is None or self.ridge_ is None:
            raise RuntimeError("Call fit() before get_war()")

        out = df.copy()
        X, _ = build_feature_matrix(out)
        X_s = self.scaler_.transform(X)

        g_ridge60 = self.ridge_.predict(X_s)              # predicted G60
        toi       = out["toi_min"].clip(lower=0.0).values

        oGAR = (g_ridge60 - self._repl_level) * toi / 60.0

        # GPW: use canonical constants (season-level) not the CSV column, which
        # may carry GPW_FALLBACK=6.0 for older seasons (confirmed: war_2324.csv
        # has goals_per_win=6.0 vs true GPW 2023-24=4.569).
        if "season" in out.columns:
            gpw = out["season"].map(_GPW_CONSTANTS).fillna(4.7).values
        elif "goals_per_win" in out.columns:
            gpw = out["goals_per_win"].values
            log.warning("season column missing; falling back to CSV goals_per_win")
        else:
            gpw = np.full(len(out), 4.7)
            log.warning("No GPW source; using fallback %.1f", 4.7)

        oWAR = oGAR / gpw

        # dWAR: recompute from dGAR using correct GPW (avoids GPW_FALLBACK issue).
        # dGAR is the defensive goals-above-replacement accumulated; dWAR = dGAR / GPW.
        if "dGAR" in out.columns:
            dWAR = (out["dGAR"].values / gpw)
        elif "dWAR" in out.columns:
            log.warning("dGAR missing; using precomputed dWAR (may have wrong GPW for 2023-24)")
            dWAR = out["dWAR"].values
        else:
            log.warning("dGAR and dWAR both missing; setting dWAR=0")
            dWAR = np.zeros(len(out))

        WAR = oWAR + dWAR

        id_cols = [c for c in ["Name", "PlayerID", "Team", "season", "position", "toi_min", "GP"] if c in out.columns]
        result = out[id_cols].copy()
        result["g_ridge60"] = g_ridge60.round(4)
        result["oGAR"]      = oGAR.round(4)
        result["oWAR"]      = oWAR.round(4)
        result["dWAR"]      = dWAR.round(4)
        result["WAR"]       = WAR.round(4)
        return result.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print fitted model parameters to stdout."""
        if self.ridge_ is None:
            print("NLWarModel: not fitted yet.")
            return
        print(f"\n{'='*60}")
        print("  NLWarModel -- Expanded Feature Ridge (Phase 4)")
        print(f"  Training target:          G60 (actual goals per 60)")
        print(f"  alpha (RidgeCV selected): {self.alpha_:.4g}")
        print(f"  replacement_pct:          {self.replacement_pct}")
        print(f"  repl_level (G60):         {self._repl_level:.4f}")
        print(f"  Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
        print(f"  Ridge coefficients (on scaled features):")
        for name, coef in zip(FEATURE_COLS, self.ridge_.coef_):
            print(f"    {name:<12} {coef:+.4f}")
        print(f"{'='*60}\n")
