"""
Phase 1 Diagnostic Baseline — PWHL dWAR
Quantifies current oWAR/dWAR correlation and forward-domination before any model changes.

Run: py -3 Projects/PWHL_war/diagnose_dwar.py
"""

import os
import sys
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(__file__)

SEASON_FILES = {
    "2023-24": os.path.join(BASE_DIR, "pwhl_war_2324.csv"),
    "2024-25": os.path.join(BASE_DIR, "pwhl_war_2425.csv"),
    "2025-26": os.path.join(BASE_DIR, "pwhl_war_2526.csv"),
}
# Display TOI floors: 2x data-gen floor
TOI_FLOOR = {"2023-24": 100, "2024-25": 100, "2025-26": 50}

all_dfs = []

for season, path in SEASON_FILES.items():
    df = pd.read_csv(path)
    df["Season"] = season
    floor = TOI_FLOOR[season]
    df = df[df["toi_min"] >= floor].copy()
    all_dfs.append(df)

    print(f"\n{'='*60}")
    print(f"  {season}  (n={len(df)}, toi_min >= {floor})")
    print(f"{'='*60}")

    # Pearson and Spearman r(oWAR, dWAR)
    pearson_r, pearson_p = stats.pearsonr(df["oWAR"], df["dWAR"])
    spearman_r, spearman_p = stats.spearmanr(df["oWAR"], df["dWAR"])
    print(f"  r(oWAR, dWAR)  Pearson  r={pearson_r:+.3f}  p={pearson_p:.4f}")
    print(f"  r(oWAR, dWAR)  Spearman r={spearman_r:+.3f}  p={spearman_p:.4f}")

    # Mean dWAR by position
    pos_means = df.groupby("pos")["dWAR"].mean().round(3)
    print(f"\n  Mean dWAR by position:")
    for pos, val in pos_means.items():
        print(f"    {pos}: {val:+.3f}")

    # Top-20 dWAR by position
    top20 = df.nlargest(20, "dWAR")[["name", "pos", "toi_min", "oWAR", "dWAR", "WAR"]]
    f_count = (top20["pos"] == "F").sum()
    d_count = (top20["pos"] == "D").sum()
    print(f"\n  Top-20 dWAR — Forwards: {f_count}  Defensemen: {d_count}")
    print(top20.to_string(index=False))

pooled = pd.concat(all_dfs, ignore_index=True)
print(f"\n{'='*60}")
print(f"  POOLED (all 3 seasons, n={len(pooled)})")
print(f"{'='*60}")
p_r, p_p = stats.pearsonr(pooled["oWAR"], pooled["dWAR"])
s_r, s_p = stats.spearmanr(pooled["oWAR"], pooled["dWAR"])
print(f"  r(oWAR, dWAR)  Pearson  r={p_r:+.3f}  p={p_p:.4f}")
print(f"  r(oWAR, dWAR)  Spearman r={s_r:+.3f}  p={s_p:.4f}")
pos_means_pooled = pooled.groupby("pos")["dWAR"].mean().round(3)
print(f"\n  Mean dWAR by position (pooled):")
for pos, val in pos_means_pooled.items():
    print(f"    {pos}: {val:+.3f}")
