"""
permutation_tests.py
---------------------
Distribution-free significance tests for the PWHL WAR model.

Three tests that complement the parametric p-values in validate_fa60.py:

  1. TEAM WAR vs Pts / GD  (permutation Pearson r)
     n=6 per season makes normal-theory p-values unreliable.
     Permutation gives an exact p by shuffling team labels.

  2. YtY dWAR stability  (permutation Pearson r, one-tailed)
     Tests whether the observed year-to-year dWAR correlation is above
     what random re-labeling would produce.  Covers both transitions:
     2023-24 -> 2024-25 and 2024-25 -> 2025-26.

  3. Position composition in top-20 dWAR  (hypergeometric exact test)
     Tests whether the D count in the top-20 dWAR list is significantly
     higher (or lower) than expected given the league-wide F/D ratio.
     No simulation needed; hypergeom.sf gives an exact p.

Run
---
  py -3 Projects/PWHL_war/tests/validation/permutation_tests.py

Expected runtime: ~3-5 min (API fetches + 10 000 permutations per test).
"""

import sys
import logging
import requests
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import hypergeom

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pwhl_war.box_war      import XGWar
from pwhl_war.csv_loader   import PWHLCsvLoader
from pwhl_war.xga_war      import XGAWar, PWHLApiLoader
from pwhl_war.coord_loader import CoordLoader
from pwhl_war.constants    import GPW, TEAM_MAP

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE      = Path(__file__).parent.parent.parent
PBP_CACHE = BASE / "pwhl_war/data/raw/pbp_cache"
RAW       = BASE / "pwhl_war/data/raw"

SEASONS     = ["2023-24", "2024-25", "2025-26"]
MIN_TOI     = {"2023-24": 50,  "2024-25": 50,  "2025-26": 25}
DISPLAY_TOI = {"2023-24": 100, "2024-25": 100, "2025-26": 50}
DEF_W       = -0.10
N_PERM      = 10_000
SEED        = 42

SEP  = "=" * 70
SEP2 = "-" * 60

# ---------------------------------------------------------------------------
# Permutation helpers
# ---------------------------------------------------------------------------

def perm_pearson_r(
    x: np.ndarray,
    y: np.ndarray,
    n_iter: int = N_PERM,
    one_tailed: bool = False,
    seed: int = SEED,
) -> tuple[float, float, np.ndarray]:
    """
    Observed Pearson r and permutation p-value.

    Parameters
    ----------
    x, y        : arrays of equal length (NaNs dropped pairwise)
    one_tailed  : if True, p = P(null >= obs_r)  [use for stability tests
                  where we expect positive r under HA]
                  if False, p = P(|null| >= |obs_r|)  [two-tailed]

    Returns
    -------
    obs_r, p_value, null_distribution
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return np.nan, np.nan, np.array([])

    obs_r, _ = stats.pearsonr(x, y)
    rng = np.random.default_rng(seed)
    null = np.array([stats.pearsonr(rng.permutation(x), y)[0] for _ in range(n_iter)])

    if one_tailed:
        p = float((null >= obs_r).mean())
    else:
        p = float((np.abs(null) >= abs(obs_r)).mean())

    return float(obs_r), p, null


def null_ci(null: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    lo = (1 - level) / 2 * 100
    return float(np.percentile(null, lo)), float(np.percentile(null, 100 - lo))


def position_hypergeom(
    war_df: pd.DataFrame,
    top_n: int = 20,
    pos_col: str = "pos",
    metric: str = "dWAR",
) -> tuple[int, int, int, float, float]:
    """
    Hypergeometric test for D over-representation in the top-N by `metric`.

    Returns
    -------
    obs_D, total_D, total_players, p_over, p_under
      p_over  = P(X >= obs_D)  -- are D over-represented?
      p_under = P(X <= obs_D)  -- are D under-represented?
    """
    df = war_df.dropna(subset=[metric, pos_col])
    total   = len(df)
    total_D = int((df[pos_col] == "D").sum())
    top     = df.nlargest(top_n, metric)
    obs_D   = int((top[pos_col] == "D").sum())

    # hypergeom(M, n, N): drawing N from M total, n are "successes"
    p_over  = float(hypergeom.sf(obs_D - 1, total, total_D, top_n))
    p_under = float(hypergeom.cdf(obs_D,    total, total_D, top_n))
    return obs_D, total_D, total, p_over, p_under


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data...")
csv_loader = PWHLCsvLoader()
fa_loader  = PWHLApiLoader(cache_dir=PBP_CACHE)

season_data: dict = {}
for season in SEASONS:
    print(f"  {season}...", end=" ", flush=True)
    if season == "2025-26":
        game_data = pd.read_csv(RAW / "game_data_2526.csv", encoding="utf-8")
        game_data = game_data[game_data["position"] != "G"].copy()
        schedule  = None
    else:
        game_data = csv_loader.get_game_data(season=season)
        schedule  = csv_loader.get_schedule(season=season)

    try:
        pbp_df = CoordLoader().fetch_pbp([season])
    except Exception:
        pbp_df = None

    try:
        fa_df = XGAWar.build_fa_season(season, fa_loader, coord_df=pbp_df)
    except Exception as e:
        print(f"\n  [warn] xGA build failed for {season}: {e}")
        fa_df = None

    season_data[season] = dict(game_data=game_data, schedule=schedule,
                               fa_df=fa_df, pbp_df=pbp_df)
    print("ok")


# ---------------------------------------------------------------------------
# Team standings
# ---------------------------------------------------------------------------
SEASON_YEAR_API = {"2023-24": "2023/2024", "2024-25": "2024/2025", "2025-26": "2025/2026"}
_pwhl_games_cache: list | None = None


def _get_pwhl_games() -> list:
    global _pwhl_games_cache
    if _pwhl_games_cache is None:
        r = requests.get(
            "https://pwhl.hockey-statistics.com/api/schedule", timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; PWHL-WAR/1.0)"},
        )
        r.raise_for_status()
        _pwhl_games_cache = r.json().get("games", [])
    return _pwhl_games_cache


def fetch_standings(season: str) -> pd.DataFrame:
    season_year = SEASON_YEAR_API[season]
    blank = {"GP": 0, "W": 0, "OTL": 0, "L": 0, "GF": 0, "GA": 0}
    ts: dict = {}
    for g in _get_pwhl_games():
        if g.get("season_year") != season_year:
            continue
        if not str(g.get("status", "")).startswith("Final"):
            continue
        try:
            home = TEAM_MAP.get(g["home_team"], g["home_team"][:3].upper())
            away = TEAM_MAP.get(g["away_team"], g["away_team"][:3].upper())
            hg, ag = int(g["home_score"]), int(g["away_score"])
        except (KeyError, ValueError):
            continue
        is_ot = str(g.get("status", "")) != "Final"
        for t in (home, away):
            if t not in ts:
                ts[t] = dict(blank)
        ts[home]["GF"] += hg;  ts[home]["GA"] += ag
        ts[away]["GF"] += ag;  ts[away]["GA"] += hg
        winner, loser = (home, away) if hg > ag else (away, home)
        ts[winner]["W"] += 1;  ts[winner]["GP"] += 1
        ts[loser]["GP"] += 1
        if is_ot:
            ts[loser]["OTL"] += 1
        else:
            ts[loser]["L"] += 1
    if not ts:
        return pd.DataFrame()
    df = pd.DataFrame([{"team": t, **v} for t, v in ts.items()])
    df["Pts"] = 2 * df["W"] + df["OTL"]
    df["GD"]  = df["GF"] - df["GA"]
    return df


print("\nFetching standings...")
standings: dict = {}
for season in SEASONS:
    try:
        st = fetch_standings(season)
        standings[season] = st
        print(f"  {season}: {len(st)} teams")
    except Exception as e:
        print(f"  {season}: ERROR ({e})")
        standings[season] = pd.DataFrame()


# ---------------------------------------------------------------------------
# Run model
# ---------------------------------------------------------------------------
def run_model(season: str) -> pd.DataFrame:
    d = season_data[season]
    model = XGWar(
        min_toi_min    = MIN_TOI[season],
        defense_weight = DEF_W,
        goals_per_win  = GPW[season],
        team_adjust    = False,
    )
    model.fit(d["game_data"], schedule_df=d["schedule"],
              fa_df=d["fa_df"], blocks_df=None, ozs_df=None, pbp_df=d["pbp_df"])
    return model.get_war(min_toi=MIN_TOI[season])


print("\nRunning model (defense_weight=%.2f)..." % DEF_W)
war_by_season = {s: run_model(s) for s in SEASONS}


def display_war(season: str) -> pd.DataFrame:
    return war_by_season[season][
        war_by_season[season]["toi_min"] >= DISPLAY_TOI[season]
    ].reset_index(drop=True)


# ---------------------------------------------------------------------------
# TEST 1: Team WAR vs Pts / GD — permutation Pearson r
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print(f"  TEST 1: Team WAR vs Pts / GD  (permutation Pearson r, n_perm={N_PERM:,})")
print(f"  Parametric p-values are unreliable at n=6 teams — permutation gives exact p.")
print(SEP)

team_rows_pts: list[pd.DataFrame] = []
team_rows_gd:  list[pd.DataFrame] = []

for season in SEASONS:
    st = standings[season]
    if st.empty:
        continue
    war_df   = war_by_season[season]
    team_war = war_df.groupby("team")[["oWAR", "dWAR", "WAR"]].sum().reset_index()
    st_cols  = [c for c in ["team", "Pts", "GD"] if c in st.columns]
    merged   = team_war.merge(st[st_cols], on="team", how="inner")
    merged["Season"] = season
    if len(merged) < 3:
        continue

    for outcome, rows_list in [("Pts", team_rows_pts), ("GD", team_rows_gd)]:
        if outcome not in merged.columns:
            continue
        obs_r, p_perm, null = perm_pearson_r(
            merged["WAR"].values, merged[outcome].values, one_tailed=False
        )
        par_r, par_p = stats.pearsonr(merged["WAR"].values, merged[outcome].values)
        ci_lo, ci_hi = null_ci(null)
        print(f"\n  {season}  WAR vs {outcome}  (n={len(merged)} teams)")
        print(f"    obs r = {obs_r:+.3f}")
        print(f"    parametric p = {par_p:.4f}   perm p = {p_perm:.4f}")
        print(f"    null 95% CI: [{ci_lo:+.3f}, {ci_hi:+.3f}]")

        rows_list.append(merged.assign(war_=merged["WAR"]))

# Pooled
print(f"\n  {SEP2}")
print(f"  POOLED (all seasons combined)")
for outcome, rows_list, label in [
    ("Pts", team_rows_pts, "WAR vs Pts"),
    ("GD",  team_rows_gd,  "WAR vs GD"),
]:
    if not rows_list:
        continue
    pool = pd.concat(rows_list, ignore_index=True)
    if outcome not in pool.columns:
        continue
    obs_r, p_perm, null = perm_pearson_r(
        pool["WAR"].values, pool[outcome].values, one_tailed=False
    )
    par_r, par_p = stats.pearsonr(pool["WAR"].values, pool[outcome].values)
    ci_lo, ci_hi = null_ci(null)
    print(f"\n  {label}  (n={len(pool)} team-seasons)")
    print(f"    obs r = {obs_r:+.3f}")
    print(f"    parametric p = {par_p:.4f}   perm p = {p_perm:.4f}")
    print(f"    null 95% CI: [{ci_lo:+.3f}, {ci_hi:+.3f}]")


# ---------------------------------------------------------------------------
# TEST 2: YtY dWAR / WAR stability — one-tailed permutation
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print(f"  TEST 2: Year-to-year stability  (one-tailed permutation, n_perm={N_PERM:,})")
print(f"  H1: observed YtY r > null (positive signal expected under HA).")
print(SEP)

TRANSITIONS = [("2023-24", "2024-25"), ("2024-25", "2025-26")]
METRICS = [
    ("WAR",       "WAR (total)"),
    ("dWAR",      "dWAR"),
    ("oWAR",      "oWAR"),
    ("war60",     "war60"),
    ("d_value60", "d_value60 (raw def)"),
]

for s1, s2 in TRANSITIONS:
    dfA = display_war(s1)
    dfB = display_war(s2)
    mg  = dfA.merge(dfB, on="player_id", suffixes=("_A", "_B"), how="inner")
    n   = len(mg)
    print(f"\n  {s1} -> {s2}  (n={n} players qualifying both seasons)")
    print(f"  {'Metric':<16}  {'obs r':>7}  {'par p':>7}  {'perm p':>7}  {'null 95% CI':>20}  {'significant?':>13}")
    print(f"  {'-'*74}")

    for col, label in METRICS:
        cA, cB = col + "_A", col + "_B"
        if cA not in mg.columns or cB not in mg.columns:
            continue
        valid = mg[[cA, cB]].dropna()
        if len(valid) < 5:
            continue
        x, y = valid[cA].values, valid[cB].values
        obs_r, p_perm, null = perm_pearson_r(x, y, one_tailed=True)
        par_r, par_p        = stats.pearsonr(x, y)
        ci_lo, ci_hi        = null_ci(null)
        sig = "YES *" if p_perm < 0.05 else "no"
        print(f"  {label:<16}  {obs_r:>+7.3f}  {par_p:>7.4f}  {p_perm:>7.4f}  "
              f"[{ci_lo:+.3f}, {ci_hi:+.3f}]  {sig:>13}")


# ---------------------------------------------------------------------------
# TEST 3: Position composition in top-20 dWAR — hypergeometric exact test
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print(f"  TEST 3: Position composition of top-20 dWAR  (hypergeometric exact test)")
print(f"  H1 (over): D count > expected given league F/D ratio.")
print(f"  H1 (under): D count < expected.")
print(SEP)

print(f"\n  {'Season':<10}  {'n_qual':>6}  {'n_D':>5}  {'pct_D':>6}  "
      f"{'top20_D':>8}  {'exp_D':>6}  {'p_over':>8}  {'p_under':>8}  {'verdict':>20}")
print(f"  {'-'*85}")

all_dfs_pos = []
for season in SEASONS:
    df = display_war(season)
    df["Season"] = season
    all_dfs_pos.append(df)

    obs_D, total_D, total, p_over, p_under = position_hypergeom(df)
    expected_D = round(total_D / total * 20, 1)
    pct_D      = total_D / total * 100

    if p_over < 0.05:
        verdict = "D over-represented *"
    elif p_under < 0.05:
        verdict = "D under-represented *"
    else:
        verdict = "not significant"

    print(f"  {season:<10}  {total:>6}  {total_D:>5}  {pct_D:>5.1f}%  "
          f"{obs_D:>8}  {expected_D:>6.1f}  {p_over:>8.4f}  {p_under:>8.4f}  {verdict:>20}")

# Pooled
pool_pos = pd.concat(all_dfs_pos, ignore_index=True)
obs_D, total_D, total, p_over, p_under = position_hypergeom(pool_pos)
expected_D = round(total_D / total * 20, 1)
pct_D      = total_D / total * 100
verdict = ("D over-represented *" if p_over < 0.05
           else "D under-represented *" if p_under < 0.05
           else "not significant")
print(f"  {'POOLED':<10}  {total:>6}  {total_D:>5}  {pct_D:>5.1f}%  "
      f"{obs_D:>8}  {expected_D:>6.1f}  {p_over:>8.4f}  {p_under:>8.4f}  {verdict:>20}")
print(f"  * p < 0.05")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("  SUMMARY")
print(SEP)
print("""
  Test 1 — Team WAR vs Pts/GD
    Key question: Is WAR predictive of team success beyond random chance?
    Use the perm p over the parametric p when n_teams <= 6.

  Test 2 — YtY stability (one-tailed)
    Key question: Is WAR/dWAR stable enough to be a repeatable skill signal?
    perm p answers: how often would a random shuffle produce this r?
    Compare dWAR vs oWAR stability — dWAR should lag oWAR.

  Test 3 — Position composition (hypergeometric)
    Key question: Is the F/D split in top-20 dWAR systematically skewed?
    Significant over-representation of D may indicate positional bias.
    Significant under-representation of D is unexpected and warrants review.
""")
print("Permutation tests complete.")
