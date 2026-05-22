"""
validate_assist_weight.py
--------------------------
Validates whether adding primary-assist weight to o_xG60 improves the WAR
model's correlation with team standings (Points = 2W + OTL).

Method
------
1. Runs the live Eq-5 model (xGA60, is_D, defense_weight=−0.10) for each season
   exactly as validate_fa60.py does — fresh xGA data from the API.
2. Post-processes get_war() output: for a grid of A1 weights, recomputes
   o_xG60_adj = o_xG60 + w_A1 × A1/60, then recalculates replacement level,
   oGAR, oWAR, and WAR (dWAR is unchanged).
3. Compares team-level WAR vs Points, per season and pooled.
4. Identifies the weight that maximises pooled Pearson r.
5. Reports YtY player repeatability for the best-weight variant.

Usage
-----
  py -3 Projects/PWHL_war/tests/validation/validate_assist_weight.py

Expected runtime: ~2–3 min (API fetches dominate).
"""

import sys
import logging
import requests
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pwhl_war.box_war      import XGWar
from pwhl_war.csv_loader   import PWHLCsvLoader
from pwhl_war.xga_war      import XGAWar, PWHLApiLoader
from pwhl_war.coord_loader import CoordLoader
from pwhl_war.constants    import GPW, TEAM_MAP

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Config — mirrors validate_fa60.py
# ---------------------------------------------------------------------------
BASE      = Path(__file__).parent.parent.parent
PBP_CACHE = BASE / "pwhl_war/data/raw/pbp_cache"
RAW       = BASE / "pwhl_war/data/raw"

SEASONS     = ["2023-24", "2024-25", "2025-26"]
MIN_TOI     = {"2023-24": 50,  "2024-25": 50,  "2025-26": 25}
DEF_W       = -0.10
A1_WEIGHTS  = np.round(np.arange(0.0, 0.85, 0.10), 2)  # sweep 0.0 → 0.80

SEP  = "=" * 70
SEP2 = "-" * 60

# ---------------------------------------------------------------------------
# Load shared data  (same pipeline as validate_fa60.py)
# ---------------------------------------------------------------------------
print("Loading data...")
csv_loader = PWHLCsvLoader()
fa_loader  = PWHLApiLoader(cache_dir=PBP_CACHE)

season_data = {}
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
# Team standings — PWHL API with OTL  (same as validate_fa60.py)
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
        is_ot = status != "Final"
        for t in (home, away):
            if t not in ts:
                ts[t] = dict(blank)
        ts[home]["GF"] += hg;  ts[home]["GA"] += ag
        ts[away]["GF"] += ag;  ts[away]["GA"] += hg
        winner, loser = (home, away) if hg > ag else (away, home)
        ts[winner]["W"]  += 1;  ts[winner]["GP"] += 1
        ts[loser]["GP"]  += 1
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

print("\nFetching team standings from PWHL API...")
standings: dict[str, pd.DataFrame] = {}
for season in SEASONS:
    try:
        st = fetch_standings(season)
        standings[season] = st
        if not st.empty:
            print(f"  {season}: {len(st)} teams  Pts range [{st['Pts'].min()}..{st['Pts'].max()}]")
        else:
            print(f"  {season}: EMPTY")
    except Exception as e:
        print(f"  {season}: ERROR ({e})")
        standings[season] = pd.DataFrame()


# ---------------------------------------------------------------------------
# Run base model for each season  (no assist adjustment yet)
# ---------------------------------------------------------------------------
def run_base_model(season: str) -> pd.DataFrame:
    d = season_data[season]
    model = XGWar(
        min_toi_min    = MIN_TOI[season],
        defense_weight = DEF_W,
        goals_per_win  = GPW[season],
        team_adjust    = False,
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

print("\nRunning base model (defense_weight=%.2f)..." % DEF_W)
base_war: dict[str, pd.DataFrame] = {}
for season in SEASONS:
    base_war[season] = run_base_model(season)
    print(f"  {season}: {len(base_war[season])} qualified players  "
          f"oWAR range [{base_war[season]['oWAR'].min():.2f}..{base_war[season]['oWAR'].max():.2f}]")


# ---------------------------------------------------------------------------
# Assist-adjustment post-processor
# ---------------------------------------------------------------------------
def apply_assist_weight(war_df: pd.DataFrame, w_A1: float,
                        gpw: float, min_toi: float) -> pd.DataFrame:
    """
    Recomputes oWAR (and total WAR) after adding primary-assist rate to o_xG60.

    o_xG60_adj = o_xG60 + w_A1 × (A1 / toi_min × 60)

    Replacement level is recalculated on the adjusted offensive rate among
    qualified players (toi_min >= min_toi), keeping the same p25 convention.
    dWAR is unchanged.
    """
    df = war_df.copy()
    if "A1" not in df.columns or "toi_min" not in df.columns:
        return df

    df["A1_60"]       = df["A1"] / df["toi_min"].clip(lower=0.1) * 60
    df["o_xG60_adj"]  = df["o_xG60"] + w_A1 * df["A1_60"]

    qual = df["toi_min"] >= min_toi
    o_repl_adj = float(np.percentile(df.loc[qual, "o_xG60_adj"], 25))

    df["oGAR_adj"] = (df["o_xG60_adj"] - o_repl_adj) * (df["toi_min"] / 60)
    df["oWAR_adj"] = df["oGAR_adj"] / gpw
    df["WAR_adj"]  = df["oWAR_adj"] + df["dWAR"]

    return df


# ---------------------------------------------------------------------------
# Helper: team-level correlation  (one outcome column)
# ---------------------------------------------------------------------------
def team_corr(war_df_by_season: dict, war_col: str,
              outcome: str = "Pts") -> tuple[list, float, float, float, float]:
    """
    Returns (team_rows_list, pooled_pearson_r, pooled_p, pooled_spearman_r, pooled_sp).
    """
    team_rows = []
    for season, war_df in war_df_by_season.items():
        st = standings.get(season, pd.DataFrame())
        if st.empty or war_col not in war_df.columns:
            continue
        tw = war_df.groupby("team")[[war_col]].sum().reset_index()
        m  = tw.merge(st[["team", outcome]], on="team", how="inner")
        if len(m) < 3:
            continue
        m["Season"] = season
        team_rows.append(m)

    if not team_rows:
        return [], np.nan, np.nan, np.nan, np.nan

    pooled = pd.concat(team_rows, ignore_index=True)
    pr, pp = stats.pearsonr(pooled[war_col], pooled[outcome])
    sr, sp = stats.spearmanr(pooled[war_col], pooled[outcome])
    return team_rows, float(pr), float(pp), float(sr), float(sp)


# ---------------------------------------------------------------------------
# PART 1: Base model performance
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("  PART 1: BASE MODEL (Eq 5, no assists)  —  WAR vs Pts")
print(SEP)

_, pr_base, pp_base, sr_base, sp_base = team_corr(base_war, "WAR")
print(f"\n  Pooled (n={20}):  Pearson r={pr_base:+.3f}  p={pp_base:.4f}"
      f"  |  Spearman r={sr_base:+.3f}  p={sp_base:.4f}")
print(f"\n  Per-season:")
for season in SEASONS:
    st = standings.get(season, pd.DataFrame())
    if st.empty:
        continue
    tw = base_war[season].groupby("team")[["oWAR","dWAR","WAR"]].sum().reset_index()
    m  = tw.merge(st[["team","Pts","GD"]], on="team", how="inner")
    if len(m) < 3:
        continue
    pr, pp = stats.pearsonr(m["WAR"], m["Pts"])
    sr, sp = stats.spearmanr(m["WAR"], m["Pts"])
    pr_o, _ = stats.pearsonr(m["oWAR"], m["Pts"])
    pr_d, _ = stats.pearsonr(m["dWAR"], m["Pts"])
    print(f"    {season}  n={len(m)}  WAR vs Pts: Pearson={pr:+.3f}(p={pp:.4f}) "
          f" Spearman={sr:+.3f}")
    print(f"           oWAR vs Pts: Pearson={pr_o:+.3f}   dWAR vs Pts: Pearson={pr_d:+.3f}")
    print(m.sort_values("Pts", ascending=False)[["team","Pts","GD","oWAR","dWAR","WAR"]].to_string(index=False))


# ---------------------------------------------------------------------------
# PART 2: A1 weight sweep
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("  PART 2: PRIMARY-ASSIST WEIGHT SWEEP  (w_A1 = 0.0 → 0.8, w_A2 = 0)")
print("  o_xG60_adj = o_xG60 + w_A1 × (A1 / toi_min × 60)")
print("  Replacement level recalculated per season at p25 of qualified players.")
print(SEP)

print(f"\n  {'w_A1':>6}  {'Pearson r':>10}  {'p':>8}  {'Spearman r':>11}  {'Δ Pearson':>10}")
print("  " + "-" * 55)

best_pr, best_w1 = pr_base, 0.0
sweep_results: list[tuple] = []

for w1 in A1_WEIGHTS:
    adj_war: dict[str, pd.DataFrame] = {}
    for season in SEASONS:
        adj_war[season] = apply_assist_weight(
            base_war[season], w1, GPW[season], MIN_TOI[season]
        )
    _, pr, pp, sr, sp = team_corr(adj_war, "WAR_adj")
    delta = pr - pr_base
    flag  = " ←" if pr > best_pr else ""
    print(f"  {w1:>6.2f}  {pr:>+10.3f}  {pp:>8.4f}  {sr:>+11.3f}  {delta:>+10.3f}{flag}")
    sweep_results.append((w1, pr, pp, sr, sp))
    if pr > best_pr:
        best_pr, best_w1 = pr, w1

print(f"\n  Best weight:  w_A1 = {best_w1:.2f}  →  Pearson r = {best_pr:+.3f}")
print(f"  vs base:      w_A1 = 0.00  →  Pearson r = {pr_base:+.3f}  (Δ = {best_pr - pr_base:+.3f})")


# ---------------------------------------------------------------------------
# PART 3: Per-season detail at best weight
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print(f"  PART 3: PER-SEASON DETAIL at best weight w_A1={best_w1:.2f}")
print(SEP)

adj_best: dict[str, pd.DataFrame] = {}
for season in SEASONS:
    adj_best[season] = apply_assist_weight(
        base_war[season], best_w1, GPW[season], MIN_TOI[season]
    )

team_rows_best = []
for season in SEASONS:
    st = standings.get(season, pd.DataFrame())
    if st.empty:
        continue
    df_adj = adj_best[season]
    agg_cols = ["oWAR", "dWAR", "WAR", "WAR_adj", "oWAR_adj"]
    agg_cols = [c for c in agg_cols if c in df_adj.columns]
    tw  = df_adj.groupby("team")[agg_cols].sum().reset_index()
    m   = tw.merge(st[["team","Pts","GD","W","OTL"]], on="team", how="inner")
    m["Season"] = season
    team_rows_best.append(m)
    n = len(m)
    if n < 3:
        continue
    pr_adj, pp_adj = stats.pearsonr(m["WAR_adj"], m["Pts"])
    sr_adj, sp_adj = stats.spearmanr(m["WAR_adj"], m["Pts"])
    pr_base_s, _   = stats.pearsonr(m["WAR"],     m["Pts"])
    print(f"\n  {season}  (n={n})")
    print(f"    WAR (base):   Pearson vs Pts = {pr_base_s:+.3f}")
    print(f"    WAR_adj (A1={best_w1:.2f}): Pearson vs Pts = {pr_adj:+.3f}(p={pp_adj:.4f})  Spearman={sr_adj:+.3f}")
    show = [c for c in ["team","Pts","GD","W","OTL","oWAR","dWAR","WAR","WAR_adj"] if c in m.columns]
    print(m.sort_values("Pts", ascending=False)[show].to_string(index=False))

if team_rows_best:
    pooled_best = pd.concat(team_rows_best, ignore_index=True)
    n = len(pooled_best)
    pr_adj_p, pp_adj_p = stats.pearsonr(pooled_best["WAR_adj"], pooled_best["Pts"])
    sr_adj_p, sp_adj_p = stats.spearmanr(pooled_best["WAR_adj"], pooled_best["Pts"])
    pr_base_p, _       = stats.pearsonr(pooled_best["WAR"],      pooled_best["Pts"])
    print(f"\n  {SEP2}")
    print(f"  POOLED (n={n} team-seasons):")
    print(f"    WAR base:    Pearson r={pr_base_p:+.3f}")
    print(f"    WAR_adj A1={best_w1:.2f}: Pearson r={pr_adj_p:+.3f}(p={pp_adj_p:.4f})  Spearman={sr_adj_p:+.3f}(p={sp_adj_p:.4f})")
    print(f"    Δ Pearson = {pr_adj_p - pr_base_p:+.3f}")
    print(f"  {SEP2}")


# ---------------------------------------------------------------------------
# PART 4: Fine-grained sweep around the best weight (step 0.05)
# ---------------------------------------------------------------------------
best_region = [round(best_w1 - 0.15, 2), round(best_w1 + 0.15, 2)]
fine_weights = np.round(np.arange(max(0.0, best_region[0]),
                                   min(0.81, best_region[1] + 0.01), 0.05), 2)
if len(fine_weights) > 1:
    print(f"\n{SEP}")
    print(f"  PART 4: FINE SWEEP  w_A1 = [{fine_weights[0]:.2f} → {fine_weights[-1]:.2f}, step 0.05]")
    print(SEP)
    print(f"\n  {'w_A1':>6}  {'Pearson r':>10}  {'p':>8}  {'Δ Pearson':>10}")
    print("  " + "-" * 40)
    fine_best_pr, fine_best_w1 = pr_base, 0.0
    for w1 in fine_weights:
        adj_w = {s: apply_assist_weight(base_war[s], w1, GPW[s], MIN_TOI[s]) for s in SEASONS}
        _, pr, pp, sr, sp = team_corr(adj_w, "WAR_adj")
        delta = pr - pr_base
        flag  = " ←" if pr > fine_best_pr else ""
        print(f"  {w1:>6.2f}  {pr:>+10.3f}  {pp:>8.4f}  {delta:>+10.3f}{flag}")
        if pr > fine_best_pr:
            fine_best_pr, fine_best_w1 = pr, w1
    print(f"\n  Fine-sweep best:  w_A1 = {fine_best_w1:.2f}  →  Pearson r = {fine_best_pr:+.3f}")
else:
    fine_best_w1 = best_w1


# ---------------------------------------------------------------------------
# PART 5: YtY player repeatability at best weight
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print(f"  PART 5: YtY PLAYER REPEATABILITY  (2024-25 → 2025-26)")
print(f"  Comparing base o_xG60 and o_xG60_adj (w_A1={fine_best_w1:.2f})")
print(SEP)

QUAL_24 = 50
QUAL_25 = 25

adj_24 = apply_assist_weight(base_war["2024-25"], fine_best_w1, GPW["2024-25"], QUAL_24)
adj_25 = apply_assist_weight(base_war["2025-26"], fine_best_w1, GPW["2025-26"], QUAL_25)

shared_cols_24 = [c for c in ["player_id","o_xG60","o_xG60_adj","d_value60","WAR","WAR_adj"] if c in adj_24.columns]
shared_cols_25 = [c for c in ["player_id","o_xG60","o_xG60_adj","d_value60","WAR","WAR_adj"] if c in adj_25.columns]

d24 = adj_24[adj_24["toi_min"] >= QUAL_24][shared_cols_24].add_suffix("_24").rename(columns={"player_id_24":"player_id"})
d25 = adj_25[adj_25["toi_min"] >= QUAL_25][shared_cols_25].add_suffix("_25").rename(columns={"player_id_25":"player_id"})
yty = d24.merge(d25, on="player_id")
n_yty = len(yty)

print(f"\n  Players in both seasons (qualified): {n_yty}")
if n_yty >= 5:
    for var, lbl in [("o_xG60",    "o_xG60 (base)"),
                     ("o_xG60_adj",f"o_xG60_adj (A1={fine_best_w1:.2f})"),
                     ("d_value60",  "d_value60"),
                     ("WAR",        "WAR (base)"),
                     ("WAR_adj",    f"WAR_adj (A1={fine_best_w1:.2f})")]:
        c24 = f"{var}_24"
        c25 = f"{var}_25"
        if c24 not in yty or c25 not in yty:
            continue
        sr, sp = stats.spearmanr(yty[c24], yty[c25])
        pr, pp = stats.pearsonr(yty[c24], yty[c25])
        print(f"  {lbl:<35}  Spearman r={sr:+.3f}(p={sp:.4f})  Pearson r={pr:+.3f}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("  SUMMARY")
print(SEP)
print(f"""
  Base Eq 5 (no assists):  pooled WAR vs Pts  Pearson = {pr_base:+.3f}
  Best assist weight:       w_A1 = {fine_best_w1:.2f}         Pearson = {fine_best_pr:+.3f}
  Improvement:              Δ Pearson = {fine_best_pr - pr_base:+.3f}

  Recommendation (pending review of Section 3–5 output):
    If Δ Pearson > 0 and YtY repeatability of o_xG60_adj ≥ o_xG60:
      → Implement w_A1 in box_war.py offensive component.
      → Tune defense_weight sweep with A1 included to confirm −0.10 remains optimal.
    If improvement is marginal (|Δ| < 0.02) or YtY drops:
      → Keep base Eq 5 unchanged; A1 adds player-level nuance but not team signal.
""")
print("Validation complete.")
