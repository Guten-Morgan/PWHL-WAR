"""
run_combined_war.py
-------------------
PWHL WAR: offensive signal from the box model, defensive signal from the
Fenwick xGA model (goalie-agnostic shot-quality attribution).

  dWAR      = xga_dWAR   (100% Fenwick — no pm60 component)
  combo_WAR = oWAR (box) + dWAR

Rationale
---------
  * Offensive WAR: box model's ixG60-based oWAR is used directly.
    The xGA model's oWAR is nearly identical (Spearman r=0.918); box oWAR
    shows marginally better team-level GD signal so it is preferred.

  * Defensive WAR: xGA dWAR only (Fenwick TOI-share attribution).
    Year-to-year stability analysis confirmed that box pm60 dWAR is
    essentially noise at the individual level (r=0.03-0.18, n.s.), while
    xGA dWAR shows consistent, significant repeatability (r=0.21-0.28).
    Both models are goalie-agnostic in design; pm60 leaks goaltender
    variance into skater dWAR, which xGA avoids by construction.

  * Validation (2024-25, n=6 teams):
      Box-only WAR            r_GD=0.600  YtY_WAR=0.316
      xGA dWAR + box oWAR     r_GD=0.714  YtY_WAR=0.781

Usage
-----
  python run_combined_war.py
  python run_combined_war.py --box-csv pwhl_war_results.csv \\
                              --xga-csv pwhl_xga_war_results.csv \\
                              --output  pwhl_combined_war_results.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_combined_war")

# ── Model blending weights ────────────────────────────────────────────────────
# Defensive signal: 100% FA/xGA (Fenwick), 0% pm60 box.
#
# Rationale for BOX_WEIGHT = 0.00:
#   FA-dWAR was validated as more repeatable year-over-year (Spearman r=0.247,
#   p=0.015) than pm60-dWAR (r=0.174, p=0.088) across the 2023-24 → 2024-25
#   and 2024-25 → 2025-26 transitions.  pm60-dWAR is near-random at the
#   individual level; xGA/FA-dWAR is goalie-agnostic by construction.
#   The name "combined" is retained for historical reasons — the weights
#   are intentionally 0.0/1.0, not a placeholder.
XGA_WEIGHT = 1.00
BOX_WEIGHT = 0.00


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PWHL WAR: box oWAR + xGA dWAR (goalie-agnostic)")
    p.add_argument("--box-csv", default="pwhl_war_results.csv")
    p.add_argument("--xga-csv", default="pwhl_xga_war_results.csv")
    p.add_argument("--output",  default="pwhl_combined_war_results.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log.info("Loading box WAR from %s", args.box_csv)
    box = pd.read_csv(args.box_csv)
    # box_war.get_war() outputs lowercase snake_case since Phase 4 refactor;
    # keep this rename as a no-op fallback for pre-refactor CSVs on disk.
    box = box.rename(columns={"Team": "team", "PlayerID": "player_id", "Name": "name"})

    log.info("Loading xGA WAR from %s", args.xga_csv)
    xga = pd.read_csv(args.xga_csv)

    # ── Aggregate box to player+team level ───────────────────────────────────
    # The box model can produce multiple rows per player per team when a player
    # is recorded at different positions (e.g. LW and RW).  Sum numeric WAR
    # components and keep the most-common position label.
    box_agg = (
        box.groupby(["Season", "player_id", "name", "team"], as_index=False)
        .agg(
            position = ("position", "first"),
            GP       = ("GP",       "sum"),
            toi_min  = ("toi_min",  "sum"),
            oWAR     = ("oWAR",     "sum"),
            dWAR     = ("dWAR",     "sum"),
            WAR      = ("WAR",      "sum"),
        )
    )

    xga_sub = (
        xga[["Season", "player_id", "team", "pos", "gp", "toi_min", "dWAR"]]
        .rename(columns={"dWAR": "xga_dWAR", "gp": "xga_gp", "pos": "xga_pos"})
        .drop(columns=["toi_min"], errors="ignore")
    )

    n_box = len(box_agg)
    n_xga = len(xga_sub)

    merged = box_agg.merge(
        xga_sub,
        on=["Season", "player_id", "team"],
        how="inner",
    )

    n_merged = len(merged)
    log.info("Players: box=%d  xga=%d  matched=%d  dropped=%d",
             n_box, n_xga, n_merged, n_box - n_merged)

    # ── Blend ─────────────────────────────────────────────────────────────────
    merged["box_dWAR"] = merged["dWAR"]   # rename for clarity
    merged["combo_dWAR"] = (BOX_WEIGHT * merged["box_dWAR"]
                           + XGA_WEIGHT * merged["xga_dWAR"])
    merged["combo_WAR"]  = merged["oWAR"] + merged["combo_dWAR"]

    # ── Output columns ────────────────────────────────────────────────────────
    out = merged[[
        "Season", "name", "player_id", "team", "position",
        "GP", "toi_min",
        "oWAR", "box_dWAR", "xga_dWAR", "combo_dWAR", "combo_WAR",
    ]].rename(columns={
        "combo_dWAR": "dWAR",
        "combo_WAR":  "WAR",
    }).copy()

    # Sort by season then WAR desc
    out = out.sort_values(["Season", "WAR"], ascending=[True, False]).reset_index(drop=True)

    # ── Per-season summary ────────────────────────────────────────────────────
    for season in sorted(out["Season"].unique()):
        sub = out[out["Season"] == season]
        log.info("Season %s: %d players  total WAR=%.2f  max WAR=%.3f (%s)",
                 season, len(sub), sub["WAR"].sum(),
                 sub["WAR"].max(), sub.iloc[0]["name"])

    # ── Save ──────────────────────────────────────────────────────────────────
    out.to_csv(args.output, index=False, encoding="utf-8")
    log.info("Saved %d rows to %s", len(out), args.output)

    # ── Console table: top 15 across all seasons ──────────────────────────────
    print(f"\nTop 15 (all seasons, box oWAR + xGA dWAR):")
    top = out.nlargest(15, "WAR")[
        ["Season", "name", "team", "position", "GP", "oWAR", "box_dWAR", "xga_dWAR", "dWAR", "WAR"]
    ]
    top = top.round(3)
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
