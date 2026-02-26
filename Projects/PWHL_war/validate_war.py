"""
validate_war.py
---------------
Compare box-WAR (pm60 model) and xGA-WAR (Fenwick model) against
two external validation metrics at the TEAM level:
  1. Goal differential  (GF - GA, regular-season)
  2. Win percentage     (W / GP)

Approach
--------
  * Aggregate individual WAR → team WAR totals from each CSV.
  * Compute team GD and W% from the hockey-statistics.com schedule table.
  * Run Spearman rank correlation AND OLS regression for each
    model × metric × season combination.
  * Print a clean comparison table.

Usage
-----
  python validate_war.py
"""

import sys
from pathlib import Path

import requests
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

from pwhl_war.constants import HOCKEYTECH_API_KEY, SEASON_IDS, TEAM_MAP, GPW

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HT_BASE = "https://lscluster.hockeytech.com/feed/index.php"
HT_KEY  = HOCKEYTECH_API_KEY
HT_CLI  = "pwhl"

BOX_CSV = Path("pwhl_war_results.csv")
XGA_CSV = Path("pwhl_xga_war_results.csv")


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def fetch_standings(season_label: str, season_id: str) -> pd.DataFrame:
    """Pull team standings (GP, W, GF, GA) from the HockeyTech schedule API."""
    sess = requests.Session()
    r = sess.get(HT_BASE, params={
        "feed": "modulekit", "view": "schedule",
        "season_id": season_id, "key": HT_KEY,
        "client_code": HT_CLI, "fmt": "json",
    }, timeout=20)
    r.raise_for_status()
    data = r.json().get("SiteKit", {})
    games = data.get("Schedule", [])

    rows = []
    for g in games:
        # Only count completed games
        status = (g.get("game_status") or g.get("GameStatus") or "").strip()
        if status.lower() != "final":
            continue
        try:
            home_gf = int(g.get("home_goal_count", 0) or 0)
            away_gf = int(g.get("visiting_goal_count", 0) or 0)
        except (TypeError, ValueError):
            continue

        home_abbr = (g.get("home_team_code") or "").strip()
        away_abbr = (g.get("visiting_team_code") or "").strip()
        if not home_abbr or not away_abbr:
            continue

        rows.append({"team": home_abbr, "GF": home_gf, "GA": away_gf,
                     "W": int(home_gf > away_gf)})
        rows.append({"team": away_abbr, "GF": away_gf, "GA": home_gf,
                     "W": int(away_gf > home_gf)})

    if not rows:
        print(f"  [warn] No completed games found for {season_label}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    agg = df.groupby("team", as_index=False).agg(
        GP=("GF", "count"), GF=("GF", "sum"), GA=("GA", "sum"), W=("W", "sum")
    )
    agg["GD"]    = agg["GF"] - agg["GA"]
    agg["Wpct"]  = agg["W"] / agg["GP"]
    agg["Season"] = season_label
    return agg


def regress(x: pd.Series, y: pd.Series, x_label: str, y_label: str) -> dict:
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return {"n": len(x), "spearman_r": np.nan, "spearman_p": np.nan,
                "ols_r2": np.nan, "ols_slope": np.nan}
    sp_r, sp_p = stats.spearmanr(x, y)
    slope, intercept, r, p_val, stderr = stats.linregress(x, y)
    return {
        "n": len(x),
        "spearman_r": round(sp_r, 3),
        "spearman_p": round(sp_p, 4),
        "ols_r2":     round(r**2, 3),
        "ols_slope":  round(slope, 4),
    }


def check_gpw_constants(standings_by_season: dict) -> None:
    """
    Derive goals-per-win from fetched standings and warn if the derived value
    differs from the inline constant in constants.py by more than 0.3.

    Parameters
    ----------
    standings_by_season : {season_label: DataFrame with GF, GA, GP columns}
    """
    for season, df in standings_by_season.items():
        if df.empty:
            continue
        total_goals = df["GF"].sum() + df["GA"].sum()
        n_team_games = df["GP"].sum()
        if n_team_games == 0:
            continue
        derived_gpw = 2.0 * (total_goals / n_team_games)
        inline_gpw = GPW.get(season)
        if inline_gpw is None:
            print(f"  [gpw_check] No inline constant for {season} — derived={derived_gpw:.3f}")
            continue
        diff = abs(derived_gpw - inline_gpw)
        if diff > 0.3:
            print(
                f"  [gpw_check] WARNING: {season} derived gpw={derived_gpw:.3f} "
                f"differs from constant {inline_gpw:.3f} by {diff:.3f} (>0.3)"
            )


if __name__ == "__main__":
    # ──────────────────────────────────────────────────────────────────────────
    # 1. Load WAR results
    # ──────────────────────────────────────────────────────────────────────────

    box = pd.read_csv(BOX_CSV)
    xga = pd.read_csv(XGA_CSV)

    # Normalise column names so both have: Season, team, WAR
    # After Phase 4, box_war.get_war() already outputs lowercase 'team',
    # but the CSV on disk may still have the old name.
    if "Team" in box.columns and "team" not in box.columns:
        box = box.rename(columns={"Team": "team"})

    # Aggregate to team × season totals
    box_team = (
        box.groupby(["Season", "team"], as_index=False)["WAR"]
        .sum()
        .rename(columns={"WAR": "box_WAR"})
    )
    xga_team = (
        xga.groupby(["Season", "team"], as_index=False)["WAR"]
        .sum()
        .rename(columns={"WAR": "xga_WAR"})
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Fetch team standings via HockeyTech API
    # ──────────────────────────────────────────────────────────────────────────

    all_standings = []
    standings_by_season: dict = {}
    for label, sid in SEASON_IDS.items():
        print(f"Fetching standings: {label} (season_id={sid}) ...", end=" ")
        try:
            df = fetch_standings(label, sid)
            if not df.empty:
                all_standings.append(df)
                standings_by_season[label] = df
                print(f"{len(df)} teams")
            else:
                print("empty")
        except Exception as exc:
            print(f"FAILED: {exc}")

    if not all_standings:
        sys.exit("Could not fetch any standings data.")

    # GPW cross-check
    check_gpw_constants(standings_by_season)

    team_stats = pd.concat(all_standings, ignore_index=True)

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Merge
    # ──────────────────────────────────────────────────────────────────────────

    merged = (
        team_stats
        .merge(box_team, left_on=["Season", "team"], right_on=["Season", "team"], how="left")
        .merge(xga_team, left_on=["Season", "team"], right_on=["Season", "team"], how="left")
        .drop(columns=["team_x", "team_y"], errors="ignore")
    )

    print("\nMerged team table:")
    print(merged.to_string(index=False))

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Run regressions per season + pooled
    # ──────────────────────────────────────────────────────────────────────────

    seasons = sorted(merged["Season"].dropna().unique())
    METRICS = [("GD", "Goal Differential"), ("Wpct", "Win %")]
    MODELS  = [("box_WAR", "Box-WAR (pm60)"), ("xga_WAR", "xGA-WAR (Fenwick)")]

    rows = []
    for season_label, data in [*[(s, merged[merged["Season"] == s]) for s in seasons],
                                 ("ALL (pooled)", merged)]:
        for war_col, war_name in MODELS:
            for metric_col, metric_name in METRICS:
                res = regress(data[war_col], data[metric_col], war_name, metric_name)
                rows.append({
                    "Season": season_label,
                    "Model": war_name,
                    "Metric": metric_name,
                    **res,
                })

    results = pd.DataFrame(rows)

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Print nicely
    # ──────────────────────────────────────────────────────────────────────────

    print("\n" + "=" * 88)
    print("TEAM-LEVEL VALIDATION:  WAR vs. Goal Differential / Win %")
    print("=" * 88)
    print(f"{'Season':<18} {'Model':<22} {'Metric':<22} {'n':>3}  "
          f"{'Spearman r':>10}  {'p':>7}  {'OLS R^2':>7}  {'slope':>8}")
    print("-" * 88)

    for _, row in results.iterrows():
        sp_star = "*" if (row["spearman_p"] < 0.05 and pd.notna(row["spearman_p"])) else " "
        print(
            f"{row['Season']:<18} {row['Model']:<22} {row['Metric']:<22}"
            f" {int(row['n']):>3}  "
            f"{row['spearman_r']:>10.3f}{sp_star} "
            f"{row['spearman_p']:>7.4f}  "
            f"{row['ols_r2']:>7.3f}  "
            f"{row['ols_slope']:>8.4f}"
        )
        # blank line between seasons
        if row["Season"] != results.iloc[-1]["Season"] and \
           row["Model"] == MODELS[-1][0] and row["Metric"] == METRICS[-1][0]:
            print()

    print("-" * 88)
    print("* p < 0.05")

    # ──────────────────────────────────────────────────────────────────────────
    # 6. Head-to-head summary
    # ──────────────────────────────────────────────────────────────────────────

    print("\n" + "=" * 80)
    print("HEAD-TO-HEAD SUMMARY  (Spearman r, pooled across all seasons)")
    print("=" * 80)
    pooled = results[results["Season"] == "ALL (pooled)"]
    for metric_col, metric_name in METRICS:
        subset = pooled[pooled["Metric"] == metric_name]
        print(f"\n  {metric_name}:")
        for _, row in subset.iterrows():
            print(f"    {row['Model']:<22}  r = {row['spearman_r']:.3f}  "
                  f"R^2 = {row['ols_r2']:.3f}  (n={int(row['n'])})")
