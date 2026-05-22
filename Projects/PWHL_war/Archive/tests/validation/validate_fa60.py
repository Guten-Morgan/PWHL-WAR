"""
validate_fa60.py
----------------
Full model validation — xGA60 + is_D, no blocks, no OZS%

Tests:
  1. Team WAR vs team goal differential  (primary external validity check)
     - Per season + pooled Pearson/Spearman r
     - defense_weight sweep to find optimal for this exact variant
  2. Year-to-year player stability  (signal repeatability)
     - 2023-24 → 2024-25  and  2024-25 → 2025-26
     - WAR, war60, oWAR, dWAR, o_xG60, d_value60

Run: py -3 Projects/PWHL_war/tests/validation/validate_fa60.py
"""

import sys
import logging
import requests
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pwhl_war.box_war      import XGWar
from pwhl_war.csv_loader   import PWHLCsvLoader
from pwhl_war.xga_war      import XGAWar, PWHLApiLoader
from pwhl_war.coord_loader import CoordLoader
from pwhl_war.constants    import GPW, TEAM_MAP

logging.basicConfig(level=logging.WARNING)

BASE      = Path(__file__).parent.parent.parent
PBP_CACHE = BASE / "pwhl_war/data/raw/pbp_cache"
RAW       = BASE / "pwhl_war/data/raw"

SEASONS     = ["2023-24", "2024-25", "2025-26"]
MIN_TOI     = {"2023-24": 50,  "2024-25": 50,  "2025-26": 25}
DISPLAY_TOI = {"2023-24": 100, "2024-25": 100, "2025-26": 50}
DEF_W       = -0.10

SEP  = "=" * 72
SEP2 = "-" * 60

# ---------------------------------------------------------------------------
# Load shared data (once)
# ---------------------------------------------------------------------------
print("Loading data...")
csv_loader = PWHLCsvLoader()
fa_loader  = PWHLApiLoader(cache_dir=PBP_CACHE)

season_data = {}
for season in SEASONS:
    print(f"  {season}...", end=" ", flush=True)

    if season == "2025-26":
        game_data = pd.read_csv(RAW / "game_data_2526.csv")
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

    season_data[season] = dict(
        game_data=game_data, schedule=schedule,
        fa_df=fa_df, pbp_df=pbp_df,
    )
    print("ok")


# ---------------------------------------------------------------------------
# Compute team GD from PWHL API schedule (works for all seasons)
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
    """Compute team standings from PWHL API completed game results.

    Columns: team, GP, W, OTL, L, Pts, GF, GA, GD
    Points: 2 per win, 1 per OT/SO loss, 0 per regulation loss.
    Status values: 'Final' (reg), 'Final OT/OT3/OT4' (OT), 'Final SO' (SO).
    """
    season_year = SEASON_YEAR_API[season]
    blank = {"GP": 0, "W": 0, "OTL": 0, "L": 0, "GF": 0, "GA": 0}
    ts: dict[str, dict] = {}

    for g in _get_pwhl_games():
        if g.get("season_year") != season_year:
            continue
        status = str(g.get("status", ""))
        if not status.startswith("Final"):
            continue
        try:
            home = TEAM_MAP.get(g["home_team"], g["home_team"][:3].upper())
            away = TEAM_MAP.get(g["away_team"], g["away_team"][:3].upper())
            hg   = int(g["home_score"])
            ag   = int(g["away_score"])
        except (KeyError, ValueError):
            continue

        is_ot = status != "Final"   # OT or SO — loser earns 1 pt

        for t in (home, away):
            if t not in ts:
                ts[t] = dict(blank)

        # goals
        ts[home]["GF"] += hg;  ts[home]["GA"] += ag
        ts[away]["GF"] += ag;  ts[away]["GA"] += hg

        # results
        winner, loser = (home, away) if hg > ag else (away, home)
        ts[winner]["W"]  += 1
        ts[winner]["GP"] += 1
        ts[loser]["GP"]  += 1
        if is_ot:
            ts[loser]["OTL"] += 1
        else:
            ts[loser]["L"] += 1

    if not ts:
        return pd.DataFrame(columns=["team", "GP", "W", "OTL", "L", "Pts", "GF", "GA", "GD"])
    df = pd.DataFrame([{"team": t, **v} for t, v in ts.items()])
    df["Pts"] = 2 * df["W"] + df["OTL"]
    df["GD"]  = df["GF"] - df["GA"]
    return df


print("\nFetching team standings from PWHL API...")
standings = {}
for season in SEASONS:
    try:
        st = fetch_standings(season)
        standings[season] = st
        if st.empty:
            print(f"  {season}: EMPTY — will skip team validation")
        else:
            print(f"  {season}: {len(st)} teams  GD range [{st['GD'].min()}..{st['GD'].max()}]")
    except Exception as e:
        print(f"  {season}: ERROR ({e})")
        standings[season] = pd.DataFrame(columns=["team", "W", "L", "GF", "GA", "GD"])


# ---------------------------------------------------------------------------
# Model runner
# ---------------------------------------------------------------------------
def run_model(season: str, defense_weight: float = DEF_W,
              team_adjust: bool = False) -> pd.DataFrame:
    d = season_data[season]
    model = XGWar(
        min_toi_min    = MIN_TOI[season],
        defense_weight = defense_weight,
        goals_per_win  = GPW[season],
        team_adjust    = team_adjust,
    )
    model.fit(
        d["game_data"],
        schedule_df = d["schedule"],
        fa_df       = d["fa_df"],
        blocks_df   = None,
        ozs_df      = None,
        pbp_df      = d["pbp_df"],
    )
    return model.get_war(min_toi=MIN_TOI[season])


# Pre-compute at default weight (reused in Part 2 and Part 3)
print("\nRunning model at default defense_weight=%.2f..." % DEF_W)
war_by_season = {s: run_model(s) for s in SEASONS}


# ---------------------------------------------------------------------------
# PART 1: Team WAR vs Standings (Pts, W, GD)
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print(f"  PART 1: Team WAR vs Standings  (defense_weight={DEF_W})")
print(SEP)

team_rows = []
for season in SEASONS:
    st = standings[season]
    if st.empty:
        continue

    war_df   = war_by_season[season]
    team_war = war_df.groupby("team")[["oWAR", "dWAR", "WAR"]].sum().reset_index()
    st_cols  = [c for c in ["team", "GP", "W", "OTL", "L", "Pts", "GF", "GA", "GD"]
                if c in st.columns]
    merged   = team_war.merge(st[st_cols], on="team", how="inner")
    merged["Season"] = season
    team_rows.append(merged)

    n = len(merged)
    if n < 3:
        print(f"\n  {season}: only {n} matched teams — skipping")
        continue

    def _r(x, y): return stats.pearsonr(merged[x], merged[y])
    def _rs(x, y): return stats.spearmanr(merged[x], merged[y])

    pr_pts, pp_pts = _r("WAR", "Pts")
    sr_pts, sp_pts = _rs("WAR", "Pts")
    pr_w,   pp_w   = _r("WAR", "W")
    pr_gd,  pp_gd  = _r("WAR", "GD")
    pr_o,   pp_o   = _r("oWAR", "Pts")
    pr_d,   pp_d   = _r("dWAR", "Pts")

    print(f"\n  {season}  (n_teams={n})")
    print(f"    r(WAR, Pts): Pearson {pr_pts:+.3f}  p={pp_pts:.4f}   Spearman {sr_pts:+.3f}  p={sp_pts:.4f}")
    print(f"    r(WAR, W  ): Pearson {pr_w:+.3f}  p={pp_w:.4f}")
    print(f"    r(WAR, GD ): Pearson {pr_gd:+.3f}  p={pp_gd:.4f}")
    print(f"    r(oWAR,Pts): Pearson {pr_o:+.3f}  p={pp_o:.4f}")
    print(f"    r(dWAR,Pts): Pearson {pr_d:+.3f}  p={pp_d:.4f}")
    show_cols = [c for c in ["team", "Pts", "W", "OTL", "L", "GD", "oWAR", "dWAR", "WAR"]
                 if c in merged.columns]
    print(merged.sort_values("Pts", ascending=False)[show_cols].to_string(index=False))

if team_rows:
    pooled_t = pd.concat(team_rows, ignore_index=True)
    n = len(pooled_t)
    pr_pts, pp_pts = stats.pearsonr(pooled_t["WAR"],  pooled_t["Pts"])
    pr_w,   pp_w   = stats.pearsonr(pooled_t["WAR"],  pooled_t["W"])
    pr_gd,  pp_gd  = stats.pearsonr(pooled_t["WAR"],  pooled_t["GD"])
    sr_pts, sp_pts = stats.spearmanr(pooled_t["WAR"], pooled_t["Pts"])
    pr_o,   pp_o   = stats.pearsonr(pooled_t["oWAR"], pooled_t["Pts"])
    pr_d,   pp_d   = stats.pearsonr(pooled_t["dWAR"], pooled_t["Pts"])
    print(f"\n  {SEP2}")
    print(f"  POOLED  ({n} team-seasons)")
    print(f"    r(WAR, Pts): Pearson {pr_pts:+.3f}  p={pp_pts:.4f}   Spearman {sr_pts:+.3f}  p={sp_pts:.4f}")
    print(f"    r(WAR, W  ): Pearson {pr_w:+.3f}  p={pp_w:.4f}")
    print(f"    r(WAR, GD ): Pearson {pr_gd:+.3f}  p={pp_gd:.4f}")
    print(f"    r(oWAR,Pts): Pearson {pr_o:+.3f}  p={pp_o:.4f}")
    print(f"    r(dWAR,Pts): Pearson {pr_d:+.3f}  p={pp_d:.4f}")
    print(f"  {SEP2}")


# ---------------------------------------------------------------------------
# PART 1b: defense_weight sweep vs Pts
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print(f"  PART 1b: defense_weight sweep vs Points  (xGA60 + is_D only)")
print(SEP)

valid_seasons = [s for s in SEASONS if not standings[s].empty]
sweep_weights = np.round(np.arange(-2.0, 2.05, 0.10), 2)
sweep_results = []

for w in sweep_weights:
    t_rows = []
    for season in valid_seasons:
        st = standings[season]
        war_df   = run_model(season, defense_weight=w)
        team_war = war_df.groupby("team")[["WAR"]].sum().reset_index()
        merged   = team_war.merge(st[["team", "Pts"]], on="team", how="inner")
        t_rows.append(merged)

    if not t_rows:
        continue
    all_t = pd.concat(t_rows, ignore_index=True)
    if len(all_t) < 3:
        continue
    pr, pp = stats.pearsonr(all_t["WAR"], all_t["Pts"])
    sweep_results.append((w, pr, pp, len(all_t)))

if sweep_results:
    best = max(sweep_results, key=lambda x: x[1])
    print(f"\n  {'Weight':>8}  {'Pearson r':>10}  {'p':>8}")
    print(f"  {'─'*34}")
    for w, r, p, n in sweep_results:
        tag = "  ← best" if abs(w - best[0]) < 0.01 else (
              "  ← current" if abs(w - DEF_W) < 0.01 else "")
        print(f"  {w:8.2f}  {r:+10.3f}  {p:8.4f}{tag}")
    print(f"\n  Best: defense_weight={best[0]}  r={best[1]:+.3f}  p={best[2]:.4f}")
    print(f"  Current (Exp-5): defense_weight={DEF_W}  "
          f"r={dict((w, r) for w, r, p, n in sweep_results).get(DEF_W, float('nan')):+.3f}")


# ---------------------------------------------------------------------------
# PART 1b-fine: Fine-grained sweep  −0.01 to −0.20 by −0.01
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print(f"  PART 1b-fine: Fine defense_weight sweep  (−0.01 → −0.20, step 0.01)")
print(SEP)

fine_weights   = np.round(np.arange(0.01, 0.205, 0.01), 2) * -1   # [-0.01, -0.02, ..., -0.20]
fine_results   = []

for w in fine_weights:
    t_rows = []
    for season in valid_seasons:
        st = standings[season]
        war_df   = run_model(season, defense_weight=float(w))
        team_war = war_df.groupby("team")[["WAR"]].sum().reset_index()
        merged   = team_war.merge(st[["team", "Pts"]], on="team", how="inner")
        t_rows.append(merged)

    if not t_rows:
        continue
    all_t = pd.concat(t_rows, ignore_index=True)
    if len(all_t) < 3:
        continue
    pr, pp = stats.pearsonr(all_t["WAR"], all_t["Pts"])
    fine_results.append((float(w), pr, pp, len(all_t)))

if fine_results:
    best_f = max(fine_results, key=lambda x: x[1])
    print(f"\n  {'Weight':>8}  {'Pearson r':>10}  {'p':>8}")
    print(f"  {'─'*34}")
    for w, r, p, n in fine_results:
        tag = "  ← best" if abs(w - best_f[0]) < 0.005 else (
              "  ← current" if abs(w - DEF_W) < 0.005 else "")
        print(f"  {w:8.2f}  {r:+10.3f}  {p:8.4f}{tag}")
    print(f"\n  Best: defense_weight={best_f[0]:.2f}  r={best_f[1]:+.3f}  p={best_f[2]:.4f}")
    print(f"  Current (Exp-5): defense_weight={DEF_W}")


# ---------------------------------------------------------------------------
# PART 1c: OLS-derived defense_weight from team-level multivariate regression
#
# Fit: TeamGD = α + β1·team_avg(o_xG60) + β2·team_avg(d_adj_xGA60)
#
# β1 = marginal goal-differential value of 1 unit of offensive xG rate.
# β2 = marginal goal-differential value of 1 unit of pre-weight defensive
#      adjusted xGA rate (already residualized and league-centered).
#
# Implied defense_weight = β2/β1:
#   rescales d_adj_xGA60 so that 1 unit is worth the same goal-differential
#   contribution as 1 unit of o_xG60 — no sweep needed.
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print(f"  PART 1c: OLS-Derived defense_weight")
print(f"  GD = α + β1·team_avg(o_xG60) + β2·team_avg(d_adj_xGA60)")
print(SEP)

from sklearn.linear_model import LinearRegression as _LR

ols_rows = []
for season in SEASONS:
    st = standings[season]
    if st.empty:
        continue
    war_df = war_by_season[season]
    if "d_adj_xGA60" not in war_df.columns:
        print(f"  {season}: d_adj_xGA60 not available — skipping")
        continue

    team_o = (
        war_df.groupby("team")
        .apply(lambda g: np.average(g["o_xG60"], weights=g["toi_min"].clip(lower=0.1)))
        .reset_index(name="team_o_xG60")
    )
    team_d = (
        war_df.groupby("team")
        .apply(lambda g: np.average(g["d_adj_xGA60"], weights=g["toi_min"].clip(lower=0.1)))
        .reset_index(name="team_d_adj")
    )
    merged = team_o.merge(team_d, on="team").merge(st[["team", "GD"]], on="team")
    merged["Season"] = season
    ols_rows.append(merged)

if ols_rows:
    ols_df = pd.concat(ols_rows, ignore_index=True)
    X = ols_df[["team_o_xG60", "team_d_adj"]].values
    y = ols_df["GD"].values

    reg_ols = _LR().fit(X, y)
    beta1, beta2 = reg_ols.coef_
    alpha_ols    = reg_ols.intercept_
    implied_w    = beta2 / beta1

    print(f"\n  Pooled OLS  (n={len(ols_df)} team-seasons)")
    print(f"    α                   = {alpha_ols:+.3f}")
    print(f"    β1  (o_xG60)        = {beta1:+.3f}")
    print(f"    β2  (d_adj_xGA60)   = {beta2:+.3f}")
    print(f"    R²                  = {reg_ols.score(X, y):.4f}")
    print(f"\n    Implied defense_weight  = β2/β1 = {implied_w:+.4f}")
    print(f"    Current Exp-5 weight            = {DEF_W:+.4f}")

    # Standardized betas: scale both inputs to unit variance so the ratio is
    # comparable regardless of the different player-level normalization of d_adj.
    o_std = ols_df["team_o_xG60"].std()
    d_std = ols_df["team_d_adj"].std()
    ols_df["team_o_z"] = ols_df["team_o_xG60"] / o_std
    ols_df["team_d_z"] = ols_df["team_d_adj"]  / d_std
    reg_std = _LR().fit(ols_df[["team_o_z", "team_d_z"]].values, y)
    b1_std, b2_std = reg_std.coef_
    implied_w_std = (b2_std / b1_std) * (o_std / d_std)   # convert back to raw scale

    print(f"\n  Standardized betas (each input scaled to unit variance among team-seasons):")
    print(f"    β1_std  (o_xG60)        = {b1_std:+.3f}  (GD per 1-team-σ of offensive rate)")
    print(f"    β2_std  (d_adj_xGA60)   = {b2_std:+.3f}  (GD per 1-team-σ of def adjustment)")
    print(f"    Ratio β2_std/β1_std     = {b2_std/b1_std:+.4f}")
    print(f"    Implied defense_weight (raw scale) = {implied_w_std:+.4f}")
    print(f"    (team-σ o={o_std:.4f}, team-σ d={d_std:.4f})")

    print(f"\n  Per-season breakdown:")
    print(f"  {'Season':<10}  {'β1 (off)':>10}  {'β2 (def)':>10}  {'β2/β1':>8}  {'R²':>6}")
    for ssn in SEASONS:
        sub = ols_df[ols_df["Season"] == ssn]
        if len(sub) < 3:
            continue
        r_s = _LR().fit(sub[["team_o_xG60", "team_d_adj"]].values, sub["GD"].values)
        b1s, b2s = r_s.coef_
        print(f"  {ssn:<10}  {b1s:+10.3f}  {b2s:+10.3f}  {b2s/b1s:+8.4f}  {r_s.score(sub[['team_o_xG60','team_d_adj']].values, sub['GD'].values):6.4f}")
else:
    print("  No seasons with d_adj_xGA60 available.")


# ---------------------------------------------------------------------------
# OLS Residual Diagnostics
# ---------------------------------------------------------------------------

def run_ols_diagnostics(df, mask, feat_cols, plot=False):
    """Print Shapiro-Wilk, Breusch-Pagan, Cook D for xGA60 ~ feat_cols."""
    sub = df[mask].copy().dropna(subset=["xGA60"] + feat_cols)
    n = len(sub)
    if n < 10:
        print(f"    n={n} too small for diagnostics")
        return
    X = sm.add_constant(sub[feat_cols].values)
    y = sub["xGA60"].values
    result = sm.OLS(y, X).fit()
    resid = result.resid
    fitted = result.fittedvalues
    sw_stat, sw_p = stats.shapiro(resid)
    bp_lm, bp_p, _, _ = het_breuschpagan(resid, X)
    cooks_d = result.get_influence().cooks_distance[0]
    cook_threshold = 4 / n
    n_flagged = int((cooks_d > cook_threshold).sum())
    if "name" in sub.columns:
        flagged = sub.loc[cooks_d > cook_threshold, "name"].tolist()[:5]
    else:
        flagged = sub.index[cooks_d > cook_threshold].tolist()[:5]
    print(f"    n={n}  R2={result.rsquared:.4f}  SW p={sw_p:.4f}  BP p={bp_p:.4f}  "
          f"Cook D>{cook_threshold:.3f}: {n_flagged} flagged  {flagged}")
    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sm.qqplot(resid, line="s", ax=axes[0])
        axes[0].set_title("QQ Plot of Residuals")
        axes[1].scatter(fitted, resid, alpha=0.5)
        axes[1].axhline(0, color="red", linestyle="--")
        axes[1].set_xlabel("Fitted values")
        axes[1].set_ylabel("Residuals")
        axes[1].set_title("Residuals vs Fitted")
        plt.tight_layout()
        plt.show()

print(f"\n{SEP}")
print("  OLS Residual Diagnostics (xGA60 ~ o_xG60 + is_D)")
print(SEP)
for _season in SEASONS:
    _df = war_by_season[_season].copy()
    if "xGA60" not in _df.columns:
        print(f"  {_season}: xGA60 column absent -- skipping")
        continue
    _df["is_D"] = (_df["pos"] == "D").astype(int)
    _mask = _df["toi_min"] >= MIN_TOI[_season]
    print(f"  {_season}:")
    run_ols_diagnostics(_df, _mask, feat_cols=["o_xG60", "is_D"], plot=False)


# ---------------------------------------------------------------------------
# PART 2: Year-to-year player stability
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print(f"  PART 2: Year-to-Year Player Stability  (defense_weight={DEF_W})")
print(SEP)

def display_war(season: str) -> pd.DataFrame:
    df = war_by_season[season].copy()
    return df[df["toi_min"] >= DISPLAY_TOI[season]].reset_index(drop=True)

def yty_report(s1: str, s2: str):
    label1 = s1.replace("-", "")
    label2 = s2.replace("-", "")
    dfA = display_war(s1)
    dfB = display_war(s2)
    merged = dfA.merge(dfB, on="player_id", suffixes=(f"_A", "_B"), how="inner")
    n = len(merged)
    print(f"\n  {s1} → {s2}  (n={n} players qualifying in both seasons)")
    print(f"  {SEP2}")

    METRICS = [
        ("WAR",       "WAR_A",       "WAR_B"),
        ("war60",     "war60_A",     "war60_B"),
        ("oWAR",      "oWAR_A",      "oWAR_B"),
        ("dWAR",      "dWAR_A",      "dWAR_B"),
        ("o_xG60",    "o_xG60_A",    "o_xG60_B"),
        ("d_value60", "d_value60_A", "d_value60_B"),
    ]

    print(f"  {'Metric':<14}  {'Pearson r':>10}  {'p':>8}  {'Spearman r':>11}  {'p':>8}  {'Kendall tau':>12}  {'p':>8}")
    for label, cA, cB in METRICS:
        if cA not in merged or cB not in merged:
            continue
        x, y = merged[cA].dropna(), merged[cB].dropna()
        # align index after dropna
        valid = merged[[cA, cB]].dropna()
        if len(valid) < 5:
            continue
        pr, pp = stats.pearsonr(valid[cA], valid[cB])
        sr, sp = stats.spearmanr(valid[cA], valid[cB])
        tr = stats.kendalltau(valid[cA], valid[cB], nan_policy="omit", variant="b")
        print(f"  {label:<14}  {pr:>+10.3f}  {pp:>8.4f}  {sr:>+11.3f}  {sp:>8.4f}  {tr.statistic:>+12.3f}  {tr.pvalue:>8.4f}")

    # Top-20 overlap
    top_A = set(dfA.nlargest(20, "WAR")["player_id"].tolist())
    top_B = set(dfB.nlargest(20, "WAR")["player_id"].tolist())
    overlap = len(top_A & top_B & set(merged["player_id"].tolist()))
    print(f"\n  Top-20 WAR repeat: {overlap}/20 players in top-20 both seasons")

    # Biggest movers
    if "WAR_A" in merged and "WAR_B" in merged:
        merged["delta"] = merged["WAR_B"] - merged["WAR_A"]
        name_col = "name_A" if "name_A" in merged.columns else "player_id"
        pos_col  = "pos_A"  if "pos_A"  in merged.columns else "pos"
        show_cols = [name_col, pos_col, "WAR_A", "WAR_B", "delta"]
        show_cols = [c for c in show_cols if c in merged.columns]

        gainers  = merged.nlargest(5, "delta")[show_cols]
        declines = merged.nsmallest(5, "delta")[show_cols]
        print(f"\n  Top-5 WAR gainers ({s1}→{s2}):")
        print(gainers.to_string(index=False))
        print(f"\n  Top-5 WAR decliners ({s1}→{s2}):")
        print(declines.to_string(index=False))

yty_report("2023-24", "2024-25")
yty_report("2024-25", "2025-26")


# ---------------------------------------------------------------------------
# PART 3: oWAR/dWAR independence summary
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print(f"  PART 3: oWAR / dWAR Correlation by Season")
print(SEP)

all_dfs = []
for season in SEASONS:
    df = display_war(season).copy()
    df["Season"] = season
    all_dfs.append(df)

    pr, pp = stats.pearsonr(df["oWAR"], df["dWAR"])
    sr, sp = stats.spearmanr(df["oWAR"], df["dWAR"])
    top20  = df.nlargest(20, "dWAR")
    fc     = (top20["pos"] == "F").sum()
    dc     = (top20["pos"] == "D").sum()
    pm     = df.groupby("pos")["dWAR"].mean()
    print(f"\n  {season}  (n={len(df)})")
    print(f"    r(oWAR,dWAR): Pearson {pr:+.3f} p={pp:.4f}   Spearman {sr:+.3f} p={sp:.4f}")
    print(f"    Top-20 dWAR: {fc}F / {dc}D   mean dWAR: F={pm.get('F',np.nan):+.3f}  D={pm.get('D',np.nan):+.3f}")

pooled = pd.concat(all_dfs, ignore_index=True)
pr, pp = stats.pearsonr(pooled["oWAR"], pooled["dWAR"])
sr, sp = stats.spearmanr(pooled["oWAR"], pooled["dWAR"])
top20_p = pooled.nlargest(20, "dWAR")
pm_p = pooled.groupby("pos")["dWAR"].mean()
print(f"\n  POOLED  (n={len(pooled)})")
print(f"    r(oWAR,dWAR): Pearson {pr:+.3f} p={pp:.4f}   Spearman {sr:+.3f} p={sp:.4f}")
print(f"    Top-20 dWAR: {(top20_p['pos']=='F').sum()}F / {(top20_p['pos']=='D').sum()}D"
      f"   mean dWAR: F={pm_p.get('F',np.nan):+.3f}  D={pm_p.get('D',np.nan):+.3f}")
print()


# ---------------------------------------------------------------------------
# PART 4: defense_weight = −0.92 comparison
#
# Variants tested:
#   A  w=−0.10  team_adjust=False  (Exp-5 current)
#   B  w=−0.92  team_adjust=False  (OLS-implied, 2024-25 season only)
#   C  w=−0.92  team_adjust=True
#
# Tests run for each variant:
#   4a  Team WAR vs Pts / GD  (Pearson r pooled)
#       Team oWAR vs GF, team dWAR vs GA  (incremental signal split)
#       ΔR² test: does adding dWAR improve team prediction beyond oWAR alone?
#   4b  r(oWAR, dWAR) independence  (pooled player-level Pearson)
#   4c  YtY dWAR stability  (Pearson r, both transitions)
#   4d  F/D composition of top-20 dWAR + mean dWAR by position
#
# Tests NOT run here (worth adding if data becomes available):
#   - Player-level on-ice GA rate vs dWAR  (requires shift-level RAPM data)
#   - Split-half reliability within season  (requires game-log splitting)
#   - Bootstrap CI on YtY r(dWAR)          (would require ~500 resamples)
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print(f"  PART 4: defense_weight −0.92 comparison  (3 variants)")
print(SEP)

VARIANTS = [
    ("A: −0.10 ta=F (Exp-5)", -0.10,  False),
    ("B: −0.92 ta=F",         -0.92,  False),
    ("C: −0.92 ta=T",         -0.92,  True),
]

# Pre-run all variants for all seasons
print("  Pre-computing variants...", end=" ", flush=True)
variant_wars: dict[str, dict[str, pd.DataFrame]] = {}
for label, w, ta in VARIANTS:
    variant_wars[label] = {s: run_model(s, defense_weight=w, team_adjust=ta)
                           for s in SEASONS}
print("done")


def _team_stats(war_df: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player WAR to team level and merge with standings."""
    team_war = war_df.groupby("team")[["oWAR", "dWAR", "WAR"]].sum().reset_index()
    st_cols  = [c for c in ["team", "GP", "Pts", "GF", "GA", "GD"] if c in st.columns]
    return team_war.merge(st[st_cols], on="team", how="inner")


def _display(war_df: pd.DataFrame, season: str) -> pd.DataFrame:
    return war_df[war_df["toi_min"] >= DISPLAY_TOI[season]].reset_index(drop=True)


# ── 4a: Team-level validity ──────────────────────────────────────────────────
print(f"\n  {'─'*70}")
print(f"  4a  Team-level validity  (pooled, {len(SEASONS)} seasons)")
print(f"  {'─'*70}")
print(f"  {'Variant':<26}  {'r(WAR,Pts)':>11}  {'p':>7}  "
      f"{'r(oWAR,GF)':>11}  {'r(dWAR,GA)':>11}  {'ΔR²(+dWAR)':>11}")

for label, w, ta in VARIANTS:
    t_rows = []
    for season in SEASONS:
        st = standings[season]
        if st.empty:
            continue
        t_rows.append(_team_stats(variant_wars[label][season], st))

    if not t_rows:
        continue
    all_t = pd.concat(t_rows, ignore_index=True)
    n = len(all_t)

    r_pts, p_pts = stats.pearsonr(all_t["WAR"],  all_t["Pts"])
    r_ogf, _     = stats.pearsonr(all_t["oWAR"], all_t["GF"])
    r_dga, _     = stats.pearsonr(all_t["dWAR"], all_t["GA"])

    # ΔR²: does dWAR explain team Pts beyond oWAR alone?
    from sklearn.linear_model import LinearRegression as _LR2
    X_o  = all_t[["oWAR"]].values
    X_od = all_t[["oWAR", "dWAR"]].values
    y_pts = all_t["Pts"].values
    r2_o  = _LR2().fit(X_o,  y_pts).score(X_o,  y_pts)
    r2_od = _LR2().fit(X_od, y_pts).score(X_od, y_pts)
    delta_r2 = r2_od - r2_o

    print(f"  {label:<26}  {r_pts:+11.3f}  {p_pts:>7.4f}  "
          f"{r_ogf:+11.3f}  {r_dga:+11.3f}  {delta_r2:+11.4f}")

# ── 4b: oWAR / dWAR independence ────────────────────────────────────────────
print(f"\n  {'─'*70}")
print(f"  4b  oWAR / dWAR independence  (pooled player-level Pearson)")
print(f"  {'─'*70}")
print(f"  {'Variant':<26}  {'r(oWAR,dWAR)':>13}  {'p':>8}")

for label, w, ta in VARIANTS:
    frames = [_display(variant_wars[label][s], s) for s in SEASONS]
    pool   = pd.concat(frames, ignore_index=True)
    pr, pp = stats.pearsonr(pool["oWAR"], pool["dWAR"])
    print(f"  {label:<26}  {pr:+13.3f}  {pp:>8.4f}")

# ── 4c: YtY dWAR stability ───────────────────────────────────────────────────
print(f"\n  {'─'*70}")
print(f"  4c  Year-to-year dWAR stability  (Pearson r, display-TOI filter)")
print(f"  {'─'*70}")
print(f"  {'Variant':<26}  {'23→24 dWAR':>11}  {'p':>7}  {'tau':>7}  {'24→25 dWAR':>11}  {'p':>7}  {'tau':>7}  "
      f"{'23→24 d_val60':>13}  {'p':>7}  {'tau':>7}")

for label, w, ta in VARIANTS:
    wars = variant_wars[label]
    row_parts = []
    for s1, s2 in [("2023-24", "2024-25"), ("2024-25", "2025-26")]:
        dfA = _display(wars[s1], s1)
        dfB = _display(wars[s2], s2)
        mg  = dfA.merge(dfB, on="player_id", suffixes=("_A", "_B"), how="inner")
        if len(mg) >= 5:
            pr_d, pp_d = stats.pearsonr(mg["dWAR_A"], mg["dWAR_B"])
            tk_d = stats.kendalltau(mg["dWAR_A"], mg["dWAR_B"], nan_policy="omit", variant="b")
        else:
            pr_d, pp_d = np.nan, np.nan
            tk_d = type("T", (), {"statistic": float("nan"), "pvalue": float("nan")})()
        row_parts.append((pr_d, pp_d, tk_d.statistic, tk_d.pvalue))

    # d_value60 stability (2023→2024 only; same filter)
    dfA0 = _display(wars["2023-24"], "2023-24")
    dfB0 = _display(wars["2024-25"], "2024-25")
    mg0  = dfA0.merge(dfB0, on="player_id", suffixes=("_A", "_B"), how="inner")
    if "d_value60_A" in mg0 and len(mg0) >= 5:
        pr_dv, pp_dv = stats.pearsonr(mg0["d_value60_A"], mg0["d_value60_B"])
        tk_dv = stats.kendalltau(mg0["d_value60_A"], mg0["d_value60_B"], nan_policy="omit", variant="b")
    else:
        pr_dv, pp_dv = np.nan, np.nan
        tk_dv = type("T", (), {"statistic": float("nan"), "pvalue": float("nan")})()

    (r1, p1, t1, tp1), (r2, p2, t2, tp2) = row_parts
    print(f"  {label:<26}  {r1:+11.3f}  {p1:>7.4f}  {t1:>+7.3f}  {r2:+11.3f}  {p2:>7.4f}  {t2:>+7.3f}  "
          f"{pr_dv:+13.3f}  {pp_dv:>7.4f}  {tk_dv.statistic:>+7.3f}")

# ── 4d: F/D composition of top-20 dWAR ──────────────────────────────────────
print(f"\n  {'─'*70}")
print(f"  4d  F/D composition of top-20 dWAR + mean dWAR by position (pooled)")
print(f"  {'─'*70}")
print(f"  {'Variant':<26}  {'top20 F':>7}  {'top20 D':>7}  "
      f"{'mean F':>8}  {'mean D':>8}  {'|F−D|':>7}")

for label, w, ta in VARIANTS:
    frames = [_display(variant_wars[label][s], s) for s in SEASONS]
    pool   = pd.concat(frames, ignore_index=True)
    top20  = pool.nlargest(20, "dWAR")
    fc     = (top20["pos"] == "F").sum()
    dc     = (top20["pos"] == "D").sum()
    pm     = pool.groupby("pos")["dWAR"].mean()
    mf     = pm.get("F", np.nan)
    md     = pm.get("D", np.nan)
    print(f"  {label:<26}  {fc:>7}  {dc:>7}  {mf:>+8.3f}  {md:>+8.3f}  {abs(mf-md):>7.3f}")
