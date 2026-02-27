"""
compare_dwar_variants.py
------------------------
Compare 4 defensive component combinations, all using the Exp-4 is_D dummy.

Variants (is_D is ALWAYS included as an OLS covariate in the FA60 path):
  1. FA60              (no blocks, no OZS%)
  2. FA60 + blocks     (no OZS%)
  3. FA60 + OZS%       (no blocks)
  4. FA60 + blocks + OZS%   [current Exp-4]

defense_weight=1.40 held fixed across all variants (Exp-4 tuned value).

Run: py -3 Projects/PWHL_war/compare_dwar_variants.py
"""

import sys
import logging
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
from pwhl_war.io_utils     import load_blocks
from pwhl_war.constants    import GPW

logging.basicConfig(level=logging.WARNING)

BASE     = Path(__file__).parent
PBP_CACHE = BASE / "pwhl_war/data/raw/pbp_cache"
RAW      = BASE / "pwhl_war/data/raw"

SEASONS     = ["2023-24", "2024-25", "2025-26"]
MIN_TOI     = {"2023-24": 50,  "2024-25": 50,  "2025-26": 25}
DISPLAY_TOI = {"2023-24": 100, "2024-25": 100, "2025-26": 50}

VARIANTS = [
    ("1: FA60",                       False, False),
    ("2: FA60 + blocks",              True,  False),
    ("3: FA60 + OZS%",                False, True),
    ("4: FA60 + blocks + OZS% [Exp-4]", True, True),
]

# ---------------------------------------------------------------------------
# Load shared data once per season
# ---------------------------------------------------------------------------
print("Loading shared data for all seasons...")

csv_loader = PWHLCsvLoader()
fa_loader  = PWHLApiLoader(cache_dir=PBP_CACHE)

season_data = {}

for season in SEASONS:
    print(f"  {season}...", end=" ", flush=True)

    # Game data
    if season == "2025-26":
        game_data = pd.read_csv(RAW / "game_data_2526.csv")
        game_data = game_data[game_data["position"] != "G"].copy()
        schedule  = None
    else:
        game_data = csv_loader.get_game_data(season=season)
        schedule  = csv_loader.get_schedule(season=season)

    # FA (Fenwick Shots Against, cached PBP)
    try:
        fa_df = XGAWar.build_fa_season(season, fa_loader)
    except Exception as e:
        print(f"\n    [warn] FA failed for {season}: {e}")
        fa_df = None

    # Blocks
    try:
        blocks_df = load_blocks(season)
    except Exception:
        blocks_df = None

    # OZS%
    ozs_path = RAW / f"ozs_{season}.csv"
    ozs_df   = pd.read_csv(ozs_path)[["player_id", "ozs_pct"]] if ozs_path.exists() else None

    # PBP coordinate xG (cached)
    try:
        pbp_df = CoordLoader().fetch_pbp([season])
    except Exception:
        pbp_df = None

    season_data[season] = dict(
        game_data=game_data,
        schedule=schedule,
        fa_df=fa_df,
        blocks_df=blocks_df,
        ozs_df=ozs_df,
        pbp_df=pbp_df,
    )
    print("ok")

# ---------------------------------------------------------------------------
# Run all variants
# ---------------------------------------------------------------------------
print("\nRunning 4 variants × 3 seasons...")

variant_results = {}  # name -> list of DataFrames (one per season)

for var_name, use_blocks, use_ozs in VARIANTS:
    dfs = []
    for season in SEASONS:
        d = season_data[season]
        model = XGWar(
            min_toi_min    = MIN_TOI[season],
            defense_weight = 1.40,
            goals_per_win  = GPW[season],
        )
        model.fit(
            d["game_data"],
            schedule_df = d["schedule"],
            fa_df       = d["fa_df"],
            blocks_df   = d["blocks_df"] if use_blocks else None,
            ozs_df      = d["ozs_df"]    if use_ozs    else None,
            pbp_df      = d["pbp_df"],
        )
        war = model.get_war(min_toi=DISPLAY_TOI[season])
        war["Season"] = season
        dfs.append(war)
    variant_results[var_name] = dfs

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
SEP  = "=" * 72
SEP2 = "-" * 72

for var_name, dfs in variant_results.items():
    print(f"\n{SEP}")
    print(f"  VARIANT {var_name}")
    print(SEP)

    all_season_dfs = []

    for df in dfs:
        season = df["Season"].iloc[0]
        floor  = DISPLAY_TOI[season]
        n      = len(df)

        pr, pp = stats.pearsonr(df["oWAR"], df["dWAR"])
        sr, sp = stats.spearmanr(df["oWAR"], df["dWAR"])

        top20    = df.nlargest(20, "dWAR")
        f_count  = (top20["pos"] == "F").sum()
        d_count  = (top20["pos"] == "D").sum()

        pos_means = df.groupby("pos")["dWAR"].mean()
        f_mean    = pos_means.get("F", np.nan)
        d_mean    = pos_means.get("D", np.nan)

        print(f"\n  {season}  (n={n}, toi>={floor} min)")
        print(f"    r(oWAR,dWAR)  Pearson  r={pr:+.3f}  p={pp:.4f}"
              f"   Spearman r={sr:+.3f}  p={sp:.4f}")
        print(f"    Top-20 dWAR   {f_count}F / {d_count}D")
        print(f"    Mean dWAR     F={f_mean:+.3f}   D={d_mean:+.3f}")

        # Top-10 dWAR names
        top10 = df.nlargest(10, "dWAR")[["name", "pos", "toi_min", "oWAR", "dWAR"]]
        print(f"    Top-10 dWAR:")
        for _, row in top10.iterrows():
            print(f"      {row['name']:<25s} {row['pos']}  "
                  f"toi={row['toi_min']:6.1f}  oWAR={row['oWAR']:+.3f}  dWAR={row['dWAR']:+.3f}")

        all_season_dfs.append(df)

    pooled   = pd.concat(all_season_dfs, ignore_index=True)
    pr_p, pp_p = stats.pearsonr(pooled["oWAR"], pooled["dWAR"])
    sr_p, sp_p = stats.spearmanr(pooled["oWAR"], pooled["dWAR"])
    pos_p    = pooled.groupby("pos")["dWAR"].mean()
    top20_p  = pooled.nlargest(20, "dWAR")
    fc_p     = (top20_p["pos"] == "F").sum()
    dc_p     = (top20_p["pos"] == "D").sum()

    print(f"\n  {SEP2}")
    print(f"  POOLED  (n={len(pooled)})")
    print(f"    r(oWAR,dWAR)  Pearson  r={pr_p:+.3f}  p={pp_p:.4f}"
          f"   Spearman r={sr_p:+.3f}  p={sp_p:.4f}")
    print(f"    Top-20 dWAR   {fc_p}F / {dc_p}D")
    print(f"    Mean dWAR     F={pos_p.get('F', np.nan):+.3f}   D={pos_p.get('D', np.nan):+.3f}")
    print(f"  {SEP2}")
