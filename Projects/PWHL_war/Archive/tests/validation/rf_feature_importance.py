"""
Random Forest Feature Importance Screen -- Phase 2 of PWHL Non-Linear WAR Model

Diagnostic only. Output is a ranked feature importance list used to decide which
features belong in the Expanded Feature Ridge (Phase 4). NOT a WAR model.

Features tested:
  o_xG60     -- offensive xG per 60 (primary predictor, linear model basis)
  A1_60      -- all-situation 1st assists per 60 (proxy for EV_A1_60)
  A2_60      -- all-situation 2nd assists per 60 (proxy for EV_A2_60)
  PIM60      -- penalty minutes per 60 (proxy for PD60 -- penalties drawn - taken)
  is_D       -- 1 if defenseman, 0 otherwise
  ozs_pct    -- offensive zone start percentage (from raw ozs CSVs)

Note: PP_ixG60 and EV_Shots60 are not available across all three seasons from the
WAR CSVs (only 2025-26 has granular splits in game_data_2526.csv). This diagnostic
uses all-situation totals as proxies; results are directionally valid for feature
screening but not precise feature-engineering specifications.

Temporal CV: train on 2023-24 + 2024-25, test on 2025-26.
Gate: confirm A1_60 > A2_60 in importance; assess PIM60 rank.

Inputs:
  pwhl_war/data/war_{2324,2425,2526}.csv
  pwhl_war/data/raw/ozs_{2023-24,2024-25,2025-26}.csv
Outputs:
  prints ranked feature importances + test-set R2
  saves diagnostics/rf_feature_importance.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_DIR, "pwhl_war", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")

SEASON_CSVS = {
    "2023-24": os.path.join(DATA_DIR, "war_2324.csv"),
    "2024-25": os.path.join(DATA_DIR, "war_2425.csv"),
    "2025-26": os.path.join(DATA_DIR, "war_2526.csv"),
}
OZS_CSVS = {
    "2023-24": os.path.join(RAW_DIR, "ozs_2023-24.csv"),
    "2024-25": os.path.join(RAW_DIR, "ozs_2024-25.csv"),
    "2025-26": os.path.join(RAW_DIR, "ozs_2025-26.csv"),
}

MIN_TOI = {"2023-24": 100.0, "2024-25": 100.0, "2025-26": 50.0}

FEATURES = ["o_xG60", "A1_60", "A2_60", "PIM60", "is_D", "ozs_pct"]
TARGET = "oGAR"
OUT_PLOT = os.path.join(SCRIPT_DIR, "rf_feature_importance.png")


# ---------------------------------------------------------------------------
# Load ozs lookup: player_id -> ozs_pct per season
# ---------------------------------------------------------------------------
def load_ozs_lookup() -> dict:
    """Returns {season: {player_id: ozs_pct}}."""
    lookup = {}
    for season, path in OZS_CSVS.items():
        if not os.path.exists(path):
            print(f"WARNING: {path} not found -- ozs_pct will be imputed for {season}")
            lookup[season] = {}
            continue
        df = pd.read_csv(path, encoding="utf-8")
        lookup[season] = dict(zip(df["player_id"].astype(int), df["ozs_pct"]))
    return lookup


# ---------------------------------------------------------------------------
# Load and pool seasons
# ---------------------------------------------------------------------------
def load_and_pool(ozs_lookup: dict) -> pd.DataFrame:
    frames = []
    for season, path in SEASON_CSVS.items():
        if not os.path.exists(path):
            print(f"WARNING: {path} not found -- skipping {season}", file=sys.stderr)
            continue
        df = pd.read_csv(path, encoding="utf-8")
        # Normalize column names across seasons (war_2526 uses different casing/names)
        col_map = {
            "name": "Name", "player_id": "PlayerID", "team": "Team",
            "pos": "position", "gp": "GP",
        }
        df.rename(columns=col_map, inplace=True)
        df["season"] = season

        # Drop goalies
        if "position" in df.columns:
            df = df[df["position"] != "G"]

        # TOI filter
        df = df[df["toi_min"] >= MIN_TOI[season]].copy()

        # Derive per-60 features from totals in WAR CSV
        toi = df["toi_min"].clip(lower=0.1)
        df["A1_60"] = df["A1"] / toi * 60
        df["A2_60"] = df["A2"] / toi * 60
        df["PIM60"] = df["PIM"] / toi * 60

        # is_D dummy
        if "position" in df.columns:
            df["is_D"] = (df["position"] == "D").astype(int)
        else:
            df["is_D"] = 0  # fallback: treat all as forwards

        # ozs_pct -- join from lookup; impute with season mean for missing
        season_ozs = ozs_lookup.get(season, {})
        if "PlayerID" in df.columns:
            df["ozs_pct"] = df["PlayerID"].map(lambda pid: season_ozs.get(int(pid), np.nan))
        else:
            df["ozs_pct"] = np.nan

        season_mean_ozs = df["ozs_pct"].mean()
        if np.isnan(season_mean_ozs):
            season_mean_ozs = 0.5  # global fallback
        n_imputed = df["ozs_pct"].isna().sum()
        df["ozs_pct"] = df["ozs_pct"].fillna(season_mean_ozs)

        n = len(df)
        print(f"  {season}: {n} players | ozs_pct imputed for {n_imputed}/{n}")
        frames.append(df)

    if not frames:
        raise RuntimeError("No WAR CSVs loaded")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# RF feature importance screen
# ---------------------------------------------------------------------------
def run_rf_screen(df: pd.DataFrame) -> None:
    # Verify required columns
    required = set(FEATURES + [TARGET, "season"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Drop any rows with NaN in feature matrix or target
    df_clean = df.dropna(subset=FEATURES + [TARGET]).copy()
    dropped = len(df) - len(df_clean)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with NaN in features or target")

    train_mask = df_clean["season"].isin(["2023-24", "2024-25"])
    test_mask = df_clean["season"] == "2025-26"

    X_train = df_clean.loc[train_mask, FEATURES].values
    y_train = df_clean.loc[train_mask, TARGET].values
    X_test = df_clean.loc[test_mask, FEATURES].values
    y_test = df_clean.loc[test_mask, TARGET].values

    print(f"\n  Train: {X_train.shape[0]} player-seasons (2023-24 + 2024-25)")
    print(f"  Test:  {X_test.shape[0]} player-seasons (2025-26)")

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=4,
        random_state=42,
    )
    rf.fit(X_train, y_train)

    # Test R2 (reference only -- not a WAR quality metric)
    y_pred = rf.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)

    # Feature importances
    importances = rf.feature_importances_
    ranked = sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 55)
    print("RANDOM FOREST FEATURE IMPORTANCE SCREEN")
    print(f"n_estimators=200, max_depth=4, random_state=42")
    print(f"Test-set R2 (reference only): {test_r2:.3f}")
    print("=" * 55)
    print(f"{'Rank':<5} {'Feature':<12} {'Importance':>12}")
    print("-" * 30)
    for rank, (feat, imp) in enumerate(ranked, 1):
        print(f"  {rank:<4} {feat:<12} {imp:>12.4f}")

    # A1 vs A2 check
    imp_dict = dict(zip(FEATURES, importances))
    a1_gt_a2 = imp_dict["A1_60"] > imp_dict["A2_60"]
    pim60_rank = next(i + 1 for i, (f, _) in enumerate(ranked) if f == "PIM60")
    a1_rank = next(i + 1 for i, (f, _) in enumerate(ranked) if f == "A1_60")

    print("\n" + "-" * 55)
    print(f"A1_60 rank: {a1_rank} | A2_60 rank: {next(i+1 for i,(f,_) in enumerate(ranked) if f=='A2_60')}")
    print(f"A1_60 > A2_60: {'YES (confirmed)' if a1_gt_a2 else 'NO -- re-examine feature list'}")
    print(f"PIM60 rank: {pim60_rank} / {len(FEATURES)}")
    if pim60_rank <= 4:
        print("PIM60 assessment: SIGNIFICANT -- include PD60 in Ridge feature set")
    else:
        print("PIM60 assessment: LOW SIGNAL -- consider dropping PD60 from Ridge feature set")
    print("=" * 55)

    # ---------------------------------------------------------------------------
    # Bar chart
    # ---------------------------------------------------------------------------
    feat_names, feat_imps = zip(*ranked)
    colors = ["#1f77b4" if f in ("o_xG60", "A1_60") else "#aec7e8" for f in feat_names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(feat_names[::-1], feat_imps[::-1], color=colors[::-1])
    ax.set_xlabel("Mean Decrease in Impurity (importance)")
    ax.set_title(
        f"RF Feature Importance Screen -- Phase 2\n"
        f"Train: 2023-24+2024-25 | Test: 2025-26 | Test R2={test_r2:.3f}"
    )
    ax.axvline(0, color="black", linewidth=0.8)
    for bar, imp in zip(bars, feat_imps[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{imp:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=150)
    print(f"\nPlot saved to: {OUT_PLOT}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading WAR CSVs + ozs lookup...")
    ozs_lookup = load_ozs_lookup()
    df = load_and_pool(ozs_lookup)
    print(f"Pooled: {len(df)} player-seasons")
    run_rf_screen(df)
