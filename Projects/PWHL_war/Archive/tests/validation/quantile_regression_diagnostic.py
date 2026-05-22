"""
Quantile Regression Diagnostic -- Phase 1 of PWHL Non-Linear WAR Model

Tests whether the xG-to-wins (o_xG60 -> oGAR) relationship is linear at the
player-season level by comparing slopes at q=0.25, 0.50, 0.75.

Interpretation:
  - Similar slopes across quantiles  => linear model is empirically validated;
    skip GAM path (Phase 5A).
  - Slopes diverge > ~20% across quantiles => non-linearity is real at PWHL scale;
    activate GAM track (Phase 5A).

Inputs : pwhl_war/data/war_2324.csv, war_2425.csv, war_2526.csv
Outputs: prints slope table; saves diagnostics/qr_diagnostic.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_DIR, "pwhl_war", "data")

SEASON_CSVS = {
    "2023-24": os.path.join(DATA_DIR, "war_2324.csv"),
    "2024-25": os.path.join(DATA_DIR, "war_2425.csv"),
    "2025-26": os.path.join(DATA_DIR, "war_2526.csv"),
}

# Minimum TOI thresholds (full seasons = 100 min; partial 2025-26 = 50 min)
MIN_TOI = {
    "2023-24": 100.0,
    "2024-25": 100.0,
    "2025-26": 50.0,
}

QUANTILES = [0.25, 0.50, 0.75]
OUT_PLOT = os.path.join(SCRIPT_DIR, "qr_diagnostic.png")


# ---------------------------------------------------------------------------
# Load and pool seasons
# ---------------------------------------------------------------------------
def load_and_pool() -> pd.DataFrame:
    frames = []
    for season, path in SEASON_CSVS.items():
        if not os.path.exists(path):
            print(f"WARNING: {path} not found -- skipping {season}", file=sys.stderr)
            continue
        df = pd.read_csv(path, encoding="utf-8")
        df["season"] = season
        min_toi = MIN_TOI[season]
        before = len(df)
        df = df[df["toi_min"] >= min_toi].copy()
        # Drop goalies if position column present
        if "position" in df.columns:
            df = df[df["position"] != "G"]
        after = len(df)
        print(f"  {season}: {before} rows -> {after} after TOI/goalie filter")
        frames.append(df)
    if not frames:
        raise RuntimeError("No WAR CSVs loaded -- check DATA_DIR path")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Quantile regression
# ---------------------------------------------------------------------------
def run_diagnostic(df: pd.DataFrame) -> None:
    required = {"o_xG60", "oGAR", "toi_min"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Required columns missing from WAR CSVs: {missing}")

    y = df["oGAR"].values
    X_raw = df["o_xG60"].values
    X = sm.add_constant(X_raw)  # [const, o_xG60]

    print("\n" + "=" * 60)
    print("QUANTILE REGRESSION DIAGNOSTIC: o_xG60 -> oGAR")
    print(f"N = {len(df)} player-seasons (pooled, qualified)")
    print("=" * 60)

    slopes = {}
    intercepts = {}
    ci_low = {}
    ci_high = {}

    for q in QUANTILES:
        model = QuantReg(y, X)
        result = model.fit(q=q, vcov="iid")
        slopes[q] = result.params[1]
        intercepts[q] = result.params[0]
        ci = result.conf_int(alpha=0.05)
        ci_low[q] = ci[1, 0]
        ci_high[q] = ci[1, 1]
        print(
            f"  q={q:.2f}  slope={slopes[q]:+.4f}  "
            f"95% CI [{ci_low[q]:+.4f}, {ci_high[q]:+.4f}]  "
            f"intercept={intercepts[q]:+.4f}"
        )

    # OLS baseline
    ols_result = sm.OLS(y, X).fit()
    ols_slope = ols_result.params[1]
    ols_intercept = ols_result.params[0]
    print(f"\n  OLS   slope={ols_slope:+.4f}  "
          f"(R2={ols_result.rsquared:.3f})  "
          f"intercept={ols_intercept:+.4f}")

    # Divergence assessment
    slope_vals = list(slopes.values())
    slope_range = max(slope_vals) - min(slope_vals)
    pct_divergence = slope_range / abs(ols_slope) * 100 if ols_slope != 0 else float("inf")

    print("\n" + "-" * 60)
    print(f"Slope range across quantiles: {slope_range:.4f}")
    print(f"Divergence vs OLS slope:      {pct_divergence:.1f}%")

    if pct_divergence > 20:
        verdict = "DIVERGING (>20%) -- non-linearity detected; consider GAM (Phase 5A)"
    else:
        verdict = "CONVERGING (<=20%) -- linear model empirically validated; skip GAM"

    print(f"VERDICT: {verdict}")
    print("=" * 60)

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    x_range = np.linspace(X_raw.min(), X_raw.max(), 200)

    colors = {0.25: "steelblue", 0.50: "darkorange", 0.75: "seagreen"}
    for q in QUANTILES:
        y_hat = intercepts[q] + slopes[q] * x_range
        ax.plot(x_range, y_hat, color=colors[q], linewidth=2,
                label=f"QuantReg q={q:.2f} (slope={slopes[q]:+.3f})")

    y_ols = ols_intercept + ols_slope * x_range
    ax.plot(x_range, y_ols, color="black", linewidth=2, linestyle="--",
            label=f"OLS (slope={ols_slope:+.3f}, R2={ols_result.rsquared:.3f})")

    ax.scatter(X_raw, y, alpha=0.25, s=20, color="gray", zorder=0)
    ax.set_xlabel("o_xG60 (offensive xG per 60 min)")
    ax.set_ylabel("oGAR (offensive goals above replacement)")
    ax.set_title(
        f"Quantile Regression Diagnostic: o_xG60 -> oGAR\n"
        f"N={len(df)} player-seasons pooled | Divergence={pct_divergence:.1f}% | {verdict.split('--')[0].strip()}"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=150)
    print(f"\nPlot saved to: {OUT_PLOT}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading WAR CSVs...")
    df = load_and_pool()
    run_diagnostic(df)
