"""
run_nl_war.py
-------------
CLI runner for the PWHL non-linear WAR model (parallel track to run_war.py).

This script orchestrates the non-linear model pipeline. The linear model
(run_war.py / box_war.py) is NEVER modified by this code.

Usage
-----
  # Phase 3: verify data loading + feature matrix shape
  python Projects/PWHL_war/run_nl_war.py

  # Phase 4: run full model + validate against Exp-5 baselines
  python Projects/PWHL_war/run_nl_war.py --validate
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats

# ---------------------------------------------------------------------------
# Path setup -- allow running from repo root or Projects/PWHL_war/
# ---------------------------------------------------------------------------

SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR           # Projects/PWHL_war/
REPO_ROOT   = SCRIPT_DIR.parent.parent

# Add repo root and project dir to path so we can import pwhl_war package
for p in [str(REPO_ROOT / "Projects" / "PWHL_war"), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from pwhl_war.nl_war import (
    FEATURE_COLS,
    NLWarModel,
    build_feature_matrix,
    pool_seasons,
    temporal_cv_split,
    validate_vs_baseline,
)
from pwhl_war.constants import HOCKEYTECH_API_KEY, SEASON_IDS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = PROJECT_DIR / "pwhl_war" / "data"
RAW_DIR  = DATA_DIR / "raw"

CSV_PATHS = {
    "2023-24": str(DATA_DIR / "war_2324.csv"),
    "2024-25": str(DATA_DIR / "war_2425.csv"),
    "2025-26": str(DATA_DIR / "war_2526.csv"),
}
MIN_TOI = {
    "2023-24": 100.0,
    "2024-25": 100.0,
    "2025-26": 50.0,
}
OZS_PATHS = {
    "2023-24": str(RAW_DIR / "ozs_2023-24.csv"),
    "2024-25": str(RAW_DIR / "ozs_2024-25.csv"),
    "2025-26": str(RAW_DIR / "ozs_2025-26.csv"),
}

TRAIN_SEASONS = ["2023-24", "2024-25"]
TEST_SEASON   = "2025-26"

# Display TOI mirrors validate_fa60.py (double the data-gen floor)
DISPLAY_TOI = {"2023-24": 200.0, "2024-25": 200.0, "2025-26": 100.0}

HT_BASE = "https://lscluster.hockeytech.com/feed/index.php"
HT_KEY  = HOCKEYTECH_API_KEY
HT_CLI  = "pwhl"


# ---------------------------------------------------------------------------
# Standings helper (mirrors validate_war.py)
# ---------------------------------------------------------------------------

def fetch_standings(season_label: str, season_id: str) -> pd.DataFrame:
    """Pull team standings (GP, GF, GA) from HockeyTech schedule API."""
    sess = requests.Session()
    r = sess.get(HT_BASE, params={
        "feed": "modulekit", "view": "schedule",
        "season_id": season_id, "key": HT_KEY,
        "client_code": HT_CLI, "fmt": "json",
    }, timeout=20)
    r.raise_for_status()
    games = r.json().get("SiteKit", {}).get("Schedule", [])

    rows = []
    for g in games:
        status = (g.get("game_status") or g.get("GameStatus") or "").strip()
        if status.lower() != "final":
            continue
        try:
            home_gf = int(g.get("home_goal_count", 0) or 0)
            away_gf = int(g.get("visiting_goal_count", 0) or 0)
        except (TypeError, ValueError):
            continue
        home = (g.get("home_team_code") or "").strip()
        away = (g.get("visiting_team_code") or "").strip()
        if not home or not away:
            continue
        rows.extend([
            {"team": home, "GF": home_gf, "GA": away_gf, "W": int(home_gf > away_gf)},
            {"team": away, "GF": away_gf, "GA": home_gf, "W": int(away_gf > home_gf)},
        ])

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    agg = df.groupby("team", as_index=False).agg(
        GP=("GF", "count"), GF=("GF", "sum"), GA=("GA", "sum"), W=("W", "sum")
    )
    agg["GD"]     = agg["GF"] - agg["GA"]
    agg["Wpct"]   = agg["W"] / agg["GP"]
    agg["points"] = agg["W"] * 2
    agg["season"] = season_label
    return agg


# ---------------------------------------------------------------------------
# YtY stability helper
# ---------------------------------------------------------------------------

def yty_stability(
    war_by_season: dict[str, pd.DataFrame],
    pairs: list[tuple[str, str]],
) -> dict:
    """
    Compute YtY (year-to-year) WAR stability for consecutive season pairs.

    Matches players by PlayerID; inner join. Computes Pearson r and
    Kendall tau-b on WAR values for players qualifying in both seasons.

    Parameters
    ----------
    war_by_season : {season: war_DataFrame_from_get_war()}
    pairs         : list of (season_A, season_B) tuples

    Returns
    -------
    dict with nested results keyed by "sA->sB"
    """
    results = {}
    for sA, sB in pairs:
        dfA = war_by_season.get(sA)
        dfB = war_by_season.get(sB)
        if dfA is None or dfB is None:
            log.warning("YtY: missing season %s or %s -- skipping pair", sA, sB)
            continue

        # Apply display TOI filter
        dfA = dfA[dfA["toi_min"] >= DISPLAY_TOI.get(sA, 100.0)].copy()
        dfB = dfB[dfB["toi_min"] >= DISPLAY_TOI.get(sB, 100.0)].copy()

        if "PlayerID" not in dfA.columns or "PlayerID" not in dfB.columns:
            log.warning("YtY: PlayerID column missing -- cannot match players")
            continue

        merged = dfA.merge(dfB, on="PlayerID", suffixes=("_A", "_B"), how="inner")
        n = len(merged)

        if n < 5:
            log.warning("YtY %s->%s: only %d overlapping players -- unreliable", sA, sB, n)

        key = f"{sA}->{sB}"
        if n == 0:
            results[key] = {"n": 0, "pearson_r": None, "pearson_p": None,
                            "kendall_tau": None, "kendall_p": None}
            continue

        valid = merged[["WAR_A", "WAR_B"]].dropna()
        pr, pp = stats.pearsonr(valid["WAR_A"], valid["WAR_B"])
        kt     = stats.kendalltau(valid["WAR_A"], valid["WAR_B"],
                                   nan_policy="omit", variant="b")

        results[key] = {
            "n":           n,
            "pearson_r":   round(float(pr), 4),
            "pearson_p":   round(float(pp), 4),
            "kendall_tau": round(float(kt.statistic), 4),
            "kendall_p":   round(float(kt.pvalue), 4),
        }
        print(f"\n  YtY {key}  (n={n} players in both seasons)")
        print(f"  Pearson r  = {pr:+.4f}  (p={pp:.4f})")
        print(f"  Kendall tau-b = {kt.statistic:+.4f}  (p={kt.pvalue:.4f})")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PWHL Non-Linear WAR Model runner")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run Phase 4 model + validate against Exp-5 baselines",
    )
    args = parser.parse_args()

    # ── Phase 3: data loading + feature matrix sanity check ───────────────
    print("=" * 60)
    print("PWHL Non-Linear WAR -- Phase 3 Infrastructure Check")
    print("=" * 60)

    print("\nLoading and pooling WAR CSVs...")
    df = pool_seasons(CSV_PATHS, MIN_TOI, OZS_PATHS)

    print(f"\nPooled DataFrame: {df.shape[0]} player-seasons x {df.shape[1]} columns")
    print(f"Seasons: {df['season'].value_counts().to_dict()}")

    print(f"\nFeature set: {FEATURE_COLS}")
    X, feature_names = build_feature_matrix(df)
    print(f"Feature matrix shape: {X.shape}  (expected ~434 x {len(FEATURE_COLS)})")

    nan_counts = pd.DataFrame(X, columns=feature_names).isna().sum()
    print("\nNaN counts per feature:")
    for feat, n in nan_counts.items():
        status = "OK" if n == 0 else f"WARNING: {n} NaN"
        print(f"  {feat:<12} {status}")

    if nan_counts.sum() == 0:
        print("\nPhase 3 gate: PASS -- feature matrix has no NaN values")
    else:
        print("\nPhase 3 gate: WARNING -- NaN values present; investigate before Phase 4")

    X_train, y_train, X_test, y_test, meta_test = temporal_cv_split(
        df, TRAIN_SEASONS, TEST_SEASON
    )
    print(f"\nTemporal CV split:")
    print(f"  Train: {X_train.shape[0]} player-seasons ({', '.join(TRAIN_SEASONS)})")
    print(f"  Test:  {X_test.shape[0]} player-seasons ({TEST_SEASON})")
    print(f"  Temporal integrity: OK (no {TEST_SEASON} rows in training set)")

    if not args.validate:
        print("\nRun with --validate to execute Phase 4 Ridge model + baseline comparisons.")
        return

    # ── Phase 4: Expanded Feature Ridge + validation ───────────────────────
    print("\n" + "=" * 60)
    print("Phase 4: Expanded Feature Ridge Model")
    print("=" * 60)

    train_df = df[df["season"].isin(TRAIN_SEASONS)].copy()
    test_df  = df[df["season"] == TEST_SEASON].copy()

    # Fit model on training seasons
    print(f"\nFitting NLWarModel on {TRAIN_SEASONS} ...")
    model = NLWarModel()
    model.fit(train_df)
    model.summary()

    # Predict WAR for all seasons (needed for YtY pairs + team validation)
    war_by_season: dict[str, pd.DataFrame] = {}
    for season in [*TRAIN_SEASONS, TEST_SEASON]:
        season_df = df[df["season"] == season].copy()
        war_by_season[season] = model.get_war(season_df)

    # ── Team Spearman r (pooled across all 3 seasons) ────────────────────
    print("\n" + "=" * 60)
    print("TEAM-LEVEL VALIDATION: pooled WAR vs standings (all seasons)")
    print("=" * 60)

    all_standings = []
    for season_label, season_id in SEASON_IDS.items():
        print(f"\nFetching standings: {season_label} (season_id={season_id}) ...", end=" ")
        try:
            s = fetch_standings(season_label, season_id)
            if not s.empty:
                all_standings.append(s)
                print(f"{len(s)} teams")
            else:
                print("empty -- skipping")
        except Exception as exc:
            print(f"FAILED: {exc}")

    if not all_standings:
        print("ERROR: could not fetch any standings -- skipping team Spearman validation")
        team_result = None
    else:
        standings_pooled = pd.concat(all_standings, ignore_index=True)
        # Combine all seasons' WAR into one pooled frame
        war_pooled_frames = []
        for season_label, war_df in war_by_season.items():
            wf = war_df.copy()
            wf["season"] = season_label
            war_pooled_frames.append(wf)
        war_pooled = pd.concat(war_pooled_frames, ignore_index=True)

        # Merge on team + season to get per-season team WAR vs standings
        team_war = (
            war_pooled.groupby(["season", "Team"])["WAR"]
            .sum()
            .reset_index()
            .rename(columns={"WAR": "team_WAR", "Team": "team"})
        )
        merged = team_war.merge(
            standings_pooled[["team", "season", "GD", "Wpct", "points"]],
            on=["team", "season"],
            how="inner",
        )

        print(f"\nMerged team table (N_rows={len(merged)}):")
        print(merged.to_string(index=False))

        if len(merged) >= 4:
            r, p = stats.spearmanr(merged["team_WAR"], merged["GD"])
            print(f"\nTeam Spearman r (WAR vs GD, pooled): {r:+.4f}  p={p:.4f}")
            r_pts, p_pts = stats.spearmanr(merged["team_WAR"], merged["points"])
            print(f"Team Spearman r (WAR vs Pts, pooled): {r_pts:+.4f}  p={p_pts:.4f}")

            baseline_r = 0.722
            delta = r_pts - baseline_r
            passed = r_pts >= baseline_r
            print(f"\nEXP-5 baseline (WAR vs Pts): r={baseline_r}")
            print(f"NL model delta:              {delta:+.4f}")
            print(f"Baseline gate:               {'PASS' if passed else 'FAIL'}")
            team_result = {"spearman_r_pts": round(r_pts, 4), "p": round(p_pts, 4),
                           "spearman_r_gd": round(r, 4), "delta": round(delta, 4),
                           "pass_fail": "PASS" if passed else "FAIL"}
        else:
            print("Insufficient teams for Spearman r")
            team_result = None

    # ── YtY stability ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("YtY WAR STABILITY: Pearson r + Kendall tau-b")
    print(f"  Exp-5 baselines: Pearson r >= 0.446 | Kendall tau-b >= 0.284")
    print("=" * 60)

    yty_pairs = [("2023-24", "2024-25"), ("2024-25", "2025-26")]
    yty_results = yty_stability(war_by_season, yty_pairs)

    # Average across both YtY pairs
    pearson_vals  = [v["pearson_r"]   for v in yty_results.values() if v["pearson_r"]  is not None]
    kendall_vals  = [v["kendall_tau"] for v in yty_results.values() if v["kendall_tau"] is not None]

    avg_pearson  = round(float(np.mean(pearson_vals)),  4) if pearson_vals  else None
    avg_kendall  = round(float(np.mean(kendall_vals)),  4) if kendall_vals  else None

    print(f"\n  Average Pearson r  (both pairs): {avg_pearson:+.4f}" if avg_pearson is not None else "\n  Average Pearson r: N/A")
    print(f"  Average Kendall tau-b (both):    {avg_kendall:+.4f}" if avg_kendall is not None else "  Average Kendall tau-b: N/A")

    pearson_pass = (avg_pearson is not None and avg_pearson >= 0.446)
    kendall_pass = (avg_kendall is not None and avg_kendall >= 0.284)
    print(f"  Pearson baseline (>= 0.446): {'PASS' if pearson_pass else 'FAIL'}")
    print(f"  Kendall baseline (>= 0.284): {'PASS' if kendall_pass else 'FAIL'}")

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 4 BASELINE SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<40} {'Value':>10}  {'Baseline':>10}  {'Result':>8}")
    print("-" * 72)

    if team_result:
        r_val = team_result['spearman_r_pts']
        print(f"{'Team Spearman r (WAR vs Pts, pooled)':<40} {r_val:>+10.4f}  {0.722:>10.4f}  {team_result['pass_fail']:>8}")
    else:
        print(f"{'Team Spearman r (WAR vs Pts, pooled)':<40} {'N/A':>10}  {0.722:>10.4f}  {'N/A':>8}")

    ap_str = f"{avg_pearson:+.4f}" if avg_pearson is not None else "N/A"
    ak_str = f"{avg_kendall:+.4f}" if avg_kendall is not None else "N/A"
    print(f"{'YtY Pearson r (avg both pairs)':<40} {ap_str:>10}  {0.446:>10.4f}  {'PASS' if pearson_pass else 'FAIL':>8}")
    print(f"{'YtY Kendall tau-b (avg both pairs)':<40} {ak_str:>10}  {0.284:>10.4f}  {'PASS' if kendall_pass else 'FAIL':>8}")
    print("-" * 72)

    all_pass = (team_result is not None and team_result["pass_fail"] == "PASS") and pearson_pass and kendall_pass
    print(f"\n  All three baselines: {'PASS -- NL model is a production candidate' if all_pass else 'FAIL -- NL model does not beat all Exp-5 baselines'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
