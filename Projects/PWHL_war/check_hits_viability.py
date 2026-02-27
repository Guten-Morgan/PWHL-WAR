"""
Phase 3: Hits Viability Check
------------------------------
Determines whether hits60_adj (position+team-adjusted hits per 60) has
enough year-over-year repeatability and cross-arena consistency to be
included as a defensive component in the WAR model.

Decision thresholds:
  - YtY Spearman r(2023-24, 2024-25) >= 0.30 among shared qualified players
  - No team shows > 20% home vs away hits disparity

Run: py -3 Projects/PWHL_war/check_hits_viability.py

# VERDICT: TBD after running
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

from pwhl_war.csv_loader import PWHLCsvLoader

MIN_TOI = 50.0   # minutes; qualification floor

loader = PWHLCsvLoader()

# ---------------------------------------------------------------------------
# Load per-game data for both seasons that have blocks/hits from CSV source
# ---------------------------------------------------------------------------
print("Loading game data...")
gd_2324 = loader.get_game_data(season="2023-24")
gd_2425 = loader.get_game_data(season="2024-25")


def compute_hits_adj(gd: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Aggregate per-game data to season totals; compute position+team-adjusted
    hits per 60 using the same pipeline as blocks in box_war.py.
    """
    if "hits" not in gd.columns:
        print(f"  WARNING: 'hits' column not in {label} game data")
        return pd.DataFrame()

    # Season aggregate per player
    grp_keys = ["PlayerID", "Name", "Team", "position"]
    agg = gd.groupby(grp_keys, as_index=False).agg(
        hits    = ("hits",    "sum"),
        toi_min = ("TOI",     "sum"),
        gp      = ("TOI",     "count"),
    )
    agg["Season"] = label

    # Qualification mask
    qual = agg["toi_min"] >= MIN_TOI

    # hits per 60
    agg["hits60"] = agg["hits"] / agg["toi_min"].clip(lower=0.1) * 60

    # Position group
    agg["_pos"] = agg["position"].str.upper().map(
        lambda p: "D" if p in {"LD", "RD", "D"} else "F"
    )

    # Position-adjust (remove F/D baseline difference — qualified players only)
    pos_means = agg[qual].groupby("_pos")["hits60"].mean()
    agg["hits60_pos_adj"] = agg["hits60"] - agg["_pos"].map(pos_means).fillna(0)

    # Team-adjust (TOI-weighted team mean of qualified players)
    blk_team = (
        agg[qual]
        .groupby("Team")["hits60_pos_adj"]
        .apply(lambda g: np.average(g, weights=agg.loc[g.index, "toi_min"].clip(0.1)))
    )
    agg["hits60_adj"] = agg["hits60_pos_adj"] - agg["Team"].map(blk_team).fillna(0)

    n_qual = qual.sum()
    print(f"  {label}: {len(agg)} players total, {n_qual} qualified (toi >= {MIN_TOI} min)")
    print(f"    hits60 pos means: F={pos_means.get('F', float('nan')):.3f}  D={pos_means.get('D', float('nan')):.3f}")

    return agg


print("\n--- Computing hits60_adj ---")
h2324 = compute_hits_adj(gd_2324, "2023-24")
h2425 = compute_hits_adj(gd_2425, "2024-25")

# ---------------------------------------------------------------------------
# Year-over-year repeatability
# ---------------------------------------------------------------------------
print("\n--- Year-over-year repeatability (2023-24 → 2024-25) ---")
qual_2324 = h2324[h2324["toi_min"] >= MIN_TOI][["PlayerID", "Name", "hits60_adj"]].rename(
    columns={"hits60_adj": "h_2324"}
)
qual_2425 = h2425[h2425["toi_min"] >= MIN_TOI][["PlayerID", "hits60_adj"]].rename(
    columns={"hits60_adj": "h_2425"}
)

merged = qual_2324.merge(qual_2425, on="PlayerID", how="inner")
print(f"  Players in both seasons (qualified both): {len(merged)}")

if len(merged) >= 5:
    sp_r, sp_p = stats.spearmanr(merged["h_2324"], merged["h_2425"])
    pe_r, pe_p = stats.pearsonr(merged["h_2324"], merged["h_2425"])
    print(f"  Spearman r = {sp_r:+.3f}  (p={sp_p:.4f})")
    print(f"  Pearson  r = {pe_r:+.3f}  (p={pe_p:.4f})")
    yty_pass = sp_r >= 0.30
    print(f"  Threshold r >= 0.30: {'PASS' if yty_pass else 'FAIL'}")
else:
    print("  Insufficient overlap for correlation.")
    yty_pass = False

# ---------------------------------------------------------------------------
# Home / Away scorer bias
# ---------------------------------------------------------------------------
print("\n--- Home/Away scorer bias per team ---")
gd_all = pd.concat([gd_2324, gd_2425], ignore_index=True)

gd_all["hits60_game"] = gd_all["hits"] / gd_all["TOI"].clip(lower=0.01) * 60

home_away = (
    gd_all
    .groupby(["Team", "Venue"], as_index=False)
    .agg(avg_hits60=("hits60_game", "mean"), n_games=("hits", "count"))
)

bias_ok = True
print(f"  {'Team':>6}  {'Home hits60':>12}  {'Away hits60':>12}  {'ratio':>8}  {'flag':>6}")
for team, grp in home_away.groupby("Team"):
    home_val = grp.loc[grp["Venue"] == "Home", "avg_hits60"].values
    away_val = grp.loc[grp["Venue"] == "Away", "avg_hits60"].values
    if len(home_val) == 0 or len(away_val) == 0:
        continue
    h, a = float(home_val[0]), float(away_val[0])
    ratio = h / a if a > 0 else float("nan")
    flag = "  *** BIAS" if abs(ratio - 1.0) > 0.20 else ""
    if flag:
        bias_ok = False
    print(f"  {team:>6}  {h:>12.3f}  {a:>12.3f}  {ratio:>8.3f}{flag}")

bias_threshold_ok = bias_ok
print(f"\n  Home/away bias < 20% for all teams: {'PASS' if bias_threshold_ok else 'FAIL'}")

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
if yty_pass and bias_threshold_ok:
    verdict = "INCLUDE"
    reason  = f"YtY Spearman r={sp_r:.3f} >= 0.30 AND no team shows > 20% home bias"
elif not yty_pass:
    verdict = "EXCLUDE"
    reason  = f"YtY Spearman r={sp_r:.3f} < 0.30 — insufficient repeatability"
else:
    verdict = "EXCLUDE"
    reason  = "Home/away scorer bias > 20% for at least one team"

print(f"  VERDICT: {verdict}")
print(f"  Reason : {reason}")
print("=" * 55)
