"""
predict_goals_2526.py
---------------------
Train an xG → goals calibration model on 2023-24 and 2024-25 PWHL seasons,
then apply it to 2025-26 season data to predict each player's full-season goals.

Model
-----
Three separate OLS regressions (EV, PP, SH) of the form:

    actual_goals = α + β × ixG                (player-season level)

Fitting on separate strength states matters because shooting percentage on the
power play is systematically different from even strength.  Total predicted
goals = EV_pred + PP_pred + SH_pred.

Output
------
  - Printed leaderboard (top 30 by predicted total goals)
  - pwhl_goal_predictions_2526.csv

Usage
-----
  python predict_goals_2526.py
  python predict_goals_2526.py --min-toi 25 --full-season-games 42
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("predict_goals")

# ---------------------------------------------------------------------------
# Paths — relative to this script's location
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
RAW_DIR    = SCRIPT_DIR / "pwhl_war" / "data" / "raw"

SHAREPOINT_CACHE = RAW_DIR / "hockeystats_game_data.xlsx"
CSV_2526         = RAW_DIR / "game_data_2526.csv"

# Season codes stored in the SharePoint CSV
SEASON_CODE = {
    "2023-24": "20232024",
    "2024-25": "20242025",
}

FULL_SEASON_GAMES = 42   # PWHL regular season length


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sharepoint(seasons: list[str]) -> pd.DataFrame:
    """Load and filter the cached SharePoint CSV for the requested seasons."""
    if not SHAREPOINT_CACHE.exists():
        log.error(
            "SharePoint cache not found: %s\n"
            "Run  python run_war.py --season 2024-25  once to download it.",
            SHAREPOINT_CACHE,
        )
        sys.exit(1)

    df = pd.read_csv(SHAREPOINT_CACHE, low_memory=False)
    codes = [SEASON_CODE[s] for s in seasons if s in SEASON_CODE]
    df = df[df["Season"].astype(str).isin(codes)].copy()
    df = df[df["SeasonStage"].str.strip() == "Regular"]
    df = df[df["position"].str.upper() != "G"]
    return df


def load_2526() -> pd.DataFrame:
    """Load 2025-26 per-game CSV."""
    if not CSV_2526.exists():
        log.error("2025-26 data not found: %s", CSV_2526)
        sys.exit(1)
    df = pd.read_csv(CSV_2526, encoding="utf-8")
    df = df[df["position"].str.upper() != "G"].copy()
    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_SUM_COLS = [
    "TOI",
    "EV_ixG", "EV_G",
    "PP_ixG", "PP_G",
    "SH_ixG", "SH_G",
    "EN_G",
    "EV_A1", "EV_A2", "PP_A1", "PP_A2", "SH_A1", "SH_A2",
    "plusMinus",
]

_GROUP_KEYS = ["PlayerID", "Name", "Team", "position"]


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Roll per-game rows up to one row per player."""
    sum_cols = [c for c in _SUM_COLS if c in df.columns]
    grp_keys = [c for c in _GROUP_KEYS if c in df.columns]

    for c in sum_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    agg = df.groupby(grp_keys, as_index=False)[sum_cols].sum()

    # Games played = rows with TOI > 0
    gp = (
        df[df["TOI"] > 0]
        .groupby(grp_keys, as_index=False)
        .size()
        .rename(columns={"size": "GP"})
    )
    agg = agg.merge(gp, on=grp_keys, how="left")
    agg["GP"] = agg["GP"].fillna(0).astype(int)

    agg = agg.rename(columns={"TOI": "toi_min"})
    agg["total_ixG"] = agg.get("EV_ixG", 0) + agg.get("PP_ixG", 0) + agg.get("SH_ixG", 0)
    agg["total_G"]   = (
        agg.get("EV_G", 0) + agg.get("PP_G", 0) +
        agg.get("SH_G", 0) + agg.get("EN_G", 0)
    )
    agg["A"]         = (
        agg.get("EV_A1", 0) + agg.get("EV_A2", 0) +
        agg.get("PP_A1", 0) + agg.get("PP_A2", 0) +
        agg.get("SH_A1", 0) + agg.get("SH_A2", 0)
    )
    agg["pos"] = agg["position"].str.upper().map(
        lambda p: "D" if p in {"LD", "RD", "D"} else "F"
    )
    return agg


# ---------------------------------------------------------------------------
# Strength-state model
# ---------------------------------------------------------------------------

class StrengthModel:
    """
    Fits separate OLS models for EV, PP, and SH goal prediction.
    Each model: goals = α + β × ixG (player-season level).
    Total predicted goals = EV_pred + PP_pred + SH_pred.
    """

    STRENGTHS = [
        ("EV", "EV_ixG", "EV_G"),
        ("PP", "PP_ixG", "PP_G"),
        ("SH", "SH_ixG", "SH_G"),
    ]

    def __init__(self):
        self._models: dict[str, LinearRegression] = {}
        self._intercepts: dict[str, float] = {}
        self._slopes: dict[str, float] = {}

    def fit(self, train_df: pd.DataFrame, min_toi: float = 50.0) -> "StrengthModel":
        """
        Fit on qualified player-season rows from the training set.
        min_toi guards against noise from 1-game cameos skewing the calibration.
        """
        qualified = train_df[train_df["toi_min"] >= min_toi]
        log.info("Training on %d qualified player-seasons (min_toi=%.0f min)",
                 len(qualified), min_toi)

        for label, xg_col, g_col in self.STRENGTHS:
            X = qualified[[xg_col]].values
            y = qualified[g_col].values

            # If there are very few PP/SH events, fall back to league-average rate
            if y.sum() < 5:
                rate = y.sum() / X.sum() if X.sum() > 0 else 1.0
                log.warning(
                    "%s: too few events — using flat rate %.3f goals/xG", label, rate
                )
                self._slopes[label]     = rate
                self._intercepts[label] = 0.0
                self._models[label]     = None
                continue

            reg = LinearRegression().fit(X, y)
            self._models[label]     = reg
            self._slopes[label]     = reg.coef_[0]
            self._intercepts[label] = reg.intercept_

            y_pred = reg.predict(X)
            mae    = mean_absolute_error(y, y_pred)
            r2     = r2_score(y, y_pred)
            log.info(
                "  %s: α=%.3f  β=%.3f  MAE=%.2f  R²=%.3f  (n=%d, Σxg=%.1f, Σg=%d)",
                label,
                reg.intercept_, reg.coef_[0],
                mae, r2,
                len(qualified),
                X.sum(), int(y.sum()),
            )

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the trained models to a player-season DataFrame."""
        df = df.copy()
        total_pred = np.zeros(len(df))

        for label, xg_col, _ in self.STRENGTHS:
            if xg_col not in df.columns:
                df[f"{label}_pred_G"] = 0.0
                continue

            xg = df[xg_col].fillna(0).values
            reg = self._models.get(label)
            if reg is not None:
                pred = reg.predict(xg.reshape(-1, 1))
            else:
                pred = xg * self._slopes[label] + self._intercepts[label]

            # Goals can't be negative
            pred = np.clip(pred, 0, None)
            df[f"{label}_pred_G"] = pred.round(2)
            total_pred += pred

        df["pred_G"] = total_pred.clip(0).round(2)
        return df

    def summary(self) -> None:
        print("\nCalibration coefficients (trained on 2023-24 + 2024-25):")
        print(f"  {'Strength':<8}  {'Intercept':>10}  {'Slope (β)':>10}  Notes")
        print(f"  {'-'*50}")
        for label, _, _ in self.STRENGTHS:
            print(
                f"  {label:<8}  {self._intercepts[label]:>+10.4f}  "
                f"{self._slopes[label]:>10.4f}  "
                f"{'β≈1 means xG is well calibrated' if abs(self._slopes[label] - 1.0) < 0.15 else ''}"
            )
        print()


# ---------------------------------------------------------------------------
# Pace projection
# ---------------------------------------------------------------------------

def pace_project(df: pd.DataFrame, full_games: int = FULL_SEASON_GAMES) -> pd.DataFrame:
    """
    Extrapolate current-season predictions to a full-season pace.

    For each team, find the max games played by any player — that's the
    team's game count.  Players are paced relative to their own GP to
    account for injuries/call-ups.

    Adds columns:
      team_gp        : games the team has completed
      gp_pct         : fraction of full season played (player GP / full_games)
      pred_G_full    : pace-projected full-season predicted goals
    """
    # Use player GP as the pace denominator (handles partial seasons, injuries)
    df = df.copy()
    df["gp_pct"]      = (df["GP"] / full_games).clip(upper=1.0)
    df["pred_G_full"] = (df["pred_G"] / df["gp_pct"].replace(0, np.nan)).round(1)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PWHL xG → goals prediction for 2025-26")
    p.add_argument("--min-toi",           type=float, default=25.0,
                   help="Min TOI (min) to qualify for prediction (default 25).")
    p.add_argument("--train-min-toi",     type=float, default=50.0,
                   help="Min TOI for training regression (default 50).")
    p.add_argument("--full-season-games", type=int,   default=FULL_SEASON_GAMES,
                   help="Games in a full PWHL regular season (default 42).")
    p.add_argument("--output",            type=str,   default="pwhl_goal_predictions_2526.csv",
                   help="Output CSV filename.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load training data (2023-24 + 2024-25)
    # ------------------------------------------------------------------
    log.info("Loading 2023-24 and 2024-25 data from SharePoint cache ...")
    train_raw = load_sharepoint(["2023-24", "2024-25"])
    train_df  = aggregate(train_raw)
    log.info("Training player-seasons: %d (all TOI levels)", len(train_df))

    # ------------------------------------------------------------------
    # 2. Train strength-state models
    # ------------------------------------------------------------------
    model = StrengthModel()
    model.fit(train_df, min_toi=args.train_min_toi)
    model.summary()

    # Quick sanity check: model's in-sample predictions on training data
    train_pred = model.predict(train_df[train_df["toi_min"] >= args.train_min_toi])
    mae_total  = mean_absolute_error(train_pred["total_G"], train_pred["pred_G"])
    r2_total   = r2_score(train_pred["total_G"], train_pred["pred_G"])
    log.info(
        "In-sample total goals: MAE=%.2f  R²=%.3f  (Σactual=%d  Σpred=%.0f)",
        mae_total, r2_total,
        int(train_pred["total_G"].sum()),
        train_pred["pred_G"].sum(),
    )

    # ------------------------------------------------------------------
    # 3. Load and aggregate 2025-26
    # ------------------------------------------------------------------
    log.info("Loading 2025-26 data ...")
    raw_2526 = load_2526()
    df_2526  = aggregate(raw_2526)
    log.info(
        "2025-26 players: %d  |  games in data: %d unique",
        len(df_2526),
        raw_2526["GameID"].nunique(),
    )

    # ------------------------------------------------------------------
    # 4. Predict and pace-project
    # ------------------------------------------------------------------
    pred_df = model.predict(df_2526)
    pred_df = pace_project(pred_df, full_games=args.full_season_games)

    # Filter to min-TOI threshold
    out = pred_df[pred_df["toi_min"] >= args.min_toi].copy()
    out = out.sort_values("pred_G_full", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 5. Print leaderboard
    # ------------------------------------------------------------------
    top = out.head(30)

    print(f"\n{'='*80}")
    print(f"  PWHL 2025-26 Predicted Goals Leaderboard")
    print(f"  xG→goals model trained on 2023-24 + 2024-25 (per-strength OLS)")
    print(f"  Min TOI: {args.min_toi} min  |  Full season: {args.full_season_games} games")
    print(f"{'='*80}")
    print(
        f"  {'Rank':<5}{'Name':<25}{'Pos':<5}{'Team':<6}"
        f"{'GP':>4}{'TOI':>7}"
        f"{'xG':>7}{'ActG':>6}"
        f"{'PredG':>7}{'PredGFull':>11}"
    )
    print(f"  {'-'*78}")

    for rank, (_, row) in enumerate(top.iterrows(), 1):
        print(
            f"  {rank:<5}{row['Name']:<25}{row['pos']:<5}{row['Team']:<6}"
            f"{row['GP']:>4}{row['toi_min']:>7.0f}"
            f"{row['total_ixG']:>7.2f}{row.get('total_G', 0):>6.0f}"
            f"{row['pred_G']:>7.2f}{row['pred_G_full']:>11.1f}"
        )

    print(f"{'='*80}\n")
    print("Columns:  xG=season ixG  |  ActG=actual goals  |  "
          "PredG=predicted from current ixG  |  PredGFull=pace to full season\n")

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    save_cols = [
        "Name", "PlayerID", "Team", "pos", "GP", "toi_min",
        "EV_ixG", "PP_ixG", "SH_ixG", "total_ixG",
        "EV_G", "PP_G", "SH_G", "total_G",
        "EV_pred_G", "PP_pred_G", "SH_pred_G", "pred_G",
        "gp_pct", "pred_G_full",
    ]
    save_cols = [c for c in save_cols if c in out.columns]
    out[save_cols].to_csv(args.output, index=False, encoding="utf-8")
    log.info("Saved: %s  (%d players)", args.output, len(out))


if __name__ == "__main__":
    main()
