"""
Phase 2: Block Weight Sweep (0.00 – 0.10)
------------------------------------------
Sweep block_weight and find the value that maximises team-level Spearman r
vs goal differential.

Approach: re-derives d_value60 and dWAR from saved WAR CSVs using the
saved blocks60_adj column — no need to re-run the full model per weight.

Run: py -3 Projects/PWHL_war/tests/validation/tune_block_weight.py

# RESULT: block_weight has negligible impact on r_WAR across 0–0.10 (all values ~0.29, p > 0.20).
# Best found: 0.015 (r=0.299), statistically indistinguishable from 0.04.
# DECISION: Keep DEFAULT_BLOCK_WEIGHT = 0.04 (interpretable: ~0.04 xG per adjusted block/60).
# The oWAR/dWAR correlation (~+0.75) is not reduced by any block_weight — driver is FA60.
# Fix must come from OZS% second covariate in Phase 5.
"""

import sys
import logging
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tune_fa_weight import fetch_standings
from pwhl_war.constants import SEASON_IDS, DEFAULT_REPLACEMENT_PCT

logging.basicConfig(level=logging.WARNING)

REPL_PCT  = DEFAULT_REPLACEMENT_PCT    # 25th percentile
BASE      = Path(__file__).parent

SEASON_FILES = {
    "2023-24": BASE / "pwhl_war_2324.csv",
    "2024-25": BASE / "pwhl_war_2425.csv",
    "2025-26": BASE / "pwhl_war_2526.csv",
}

# ---------------------------------------------------------------------------
# Load standings (one network round-trip per season)
# ---------------------------------------------------------------------------
print("Fetching standings...", end=" ", flush=True)
standing_frames = []
for s, sid in SEASON_IDS.items():
    df = fetch_standings(s, sid)
    if not df.empty:
        standing_frames.append(df)
standings = pd.concat(standing_frames, ignore_index=True)
print(f"done ({len(standings)} team-seasons)")

# ---------------------------------------------------------------------------
# Load saved WAR CSVs; extract FA-only component
# ---------------------------------------------------------------------------
dfs = {}
for season, path in SEASON_FILES.items():
    df = pd.read_csv(path)
    # d_value60 = d_val60_FA + block_val60  →  isolate FA component
    df["d_val60_FA"] = df["d_value60"] - df["block_val60"]
    dfs[season] = df


# ---------------------------------------------------------------------------
# Recompute WAR at an arbitrary block_weight
# ---------------------------------------------------------------------------
def compute_war_at_block_weight(dfs: dict, block_weight: float) -> pd.DataFrame:
    """Return pooled player DataFrame with dWAR recomputed at block_weight."""
    all_rows = []
    for season, df in dfs.items():
        d = df.copy()
        d["new_block_val60"] = d["blocks60_adj"] * block_weight
        d["new_d_value60"]   = d["d_val60_FA"] + d["new_block_val60"]
        # Replacement level (p25 of qualified pool — all rows in CSV are qualified)
        d_repl = float(np.percentile(d["new_d_value60"], REPL_PCT))
        d["new_dGAR"] = (d["new_d_value60"] - d_repl) * d["toi_min"] / 60
        d["new_dWAR"] = d["new_dGAR"] / d["goals_per_win"]
        d["new_WAR"]  = d["oWAR"] + d["new_dWAR"]
        d["Season"]   = season
        all_rows.append(d)
    return pd.concat(all_rows, ignore_index=True)


def team_spearman(player_df: pd.DataFrame):
    """Aggregate player WAR to team level; return (r_WAR, p_WAR, r_oWAR)."""
    rows = []
    for season, grp in player_df.groupby("Season"):
        t = grp.groupby("team", as_index=False).agg(
            team_WAR  = ("new_WAR",  "sum"),
            team_oWAR = ("oWAR",     "sum"),
        )
        t["Season"] = season
        rows.append(t)
    team_war = pd.concat(rows, ignore_index=True)
    merged   = standings.merge(team_war, on=["Season", "team"], how="inner")
    if len(merged) < 3:
        return np.nan, np.nan, np.nan
    r_war,  p_war  = stats.spearmanr(merged["team_WAR"],  merged["GD"])
    r_owar, _      = stats.spearmanr(merged["team_oWAR"], merged["GD"])
    return r_war, p_war, r_owar


# ---------------------------------------------------------------------------
# Sweep block_weight: 0.000 → 0.100 in steps of 0.005 (21 values)
# ---------------------------------------------------------------------------
weights = np.round(np.arange(0.0, 0.105, 0.005), 4)

print(f"\n{'weight':>8}  {'r_WAR':>8}  {'p':>8}  {'r_oWAR':>8}  {'r(o,d)':>8}")
print("-" * 56)

best_w, best_r, best_p = 0.04, -999.0, 1.0
results = []

for w in weights:
    player_df = compute_war_at_block_weight(dfs, w)
    r_war, p_war, r_owar = team_spearman(player_df)
    pearson_od, _        = stats.pearsonr(player_df["oWAR"], player_df["new_dWAR"])
    sig = "*" if (p_war is not np.nan and p_war < 0.05) else " "
    print(f"{w:8.3f}  {r_war:8.3f}  {p_war:8.4f}  {r_owar:8.3f}  {pearson_od:+8.3f}  {sig}")
    results.append((w, r_war, p_war, r_owar, pearson_od))
    if r_war > best_r:
        best_r, best_w, best_p = r_war, w, p_war

print(f"\n{'='*56}")
print(f"  Best block_weight : {best_w:.3f}")
print(f"  r_WAR (GD)        : {best_r:.3f}  (p={best_p:.4f})")
# Also report at the current default (0.04)
cur = next(x for x in results if abs(x[0] - 0.04) < 1e-9)
print(f"  Current (0.040)   : r_WAR={cur[1]:.3f}  p={cur[2]:.4f}")
print(f"{'='*56}")
print(f"\nRecommend updating DEFAULT_BLOCK_WEIGHT to {best_w:.3f} in constants.py")
