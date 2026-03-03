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

Run: py -3 Projects/PWHL_war/validate_fa60.py
"""

import sys
import logging
import requests
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from pwhl_war.box_war      import XGWar
from pwhl_war.csv_loader   import PWHLCsvLoader
from pwhl_war.xga_war      import XGAWar, PWHLApiLoader
from pwhl_war.coord_loader import CoordLoader
from pwhl_war.constants    import GPW, TEAM_MAP

logging.basicConfig(level=logging.WARNING)

BASE      = Path(__file__).parent
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
def run_model(season: str, defense_weight: float = DEF_W) -> pd.DataFrame:
    d = season_data[season]
    model = XGWar(
        min_toi_min    = MIN_TOI[season],
        defense_weight = defense_weight,
        goals_per_win  = GPW[season],
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

    print(f"  {'Metric':<14}  {'Pearson r':>10}  {'p':>8}  {'Spearman r':>11}  {'p':>8}")
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
        print(f"  {label:<14}  {pr:>+10.3f}  {pp:>8.4f}  {sr:>+11.3f}  {sp:>8.4f}")

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
