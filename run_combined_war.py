"""
run_combined_war.py
-------------------
Blended PWHL WAR: combines offensive signal from the box model with a
60/40 weighted average of the two defensive signals.

  combo_dWAR = 0.40 * box_dWAR  +  0.60 * xga_dWAR
  combo_WAR  = oWAR (box)        +  combo_dWAR

Rationale
---------
  * Offensive WAR: box model's ixG60-based oWAR is used directly.
    The xGA model's oWAR is nearly identical (Spearman r=0.918) so
    either would work; box oWAR shows marginally better team-level signal.

  * Defensive WAR: the two defensive signals are near-uncorrelated
    (Spearman r=0.013), meaning they capture genuinely different aspects:
      - box dWAR  (pm60 residual): actual on-ice goals, noisy but outcome-true
      - xGA dWAR  (Fenwick TOI-share): shot quality suppressed, smoother signal
    A 60/40 xGA/box blend sits comfortably inside the validated plateau
    (r vs. team GD is flat from 28% to 72% box) while tilting toward
    xGA anticipating that signal will strengthen with more season data.

  * Validation (2024-25, n=6 teams):
      Box-only WAR        r_GD=0.600   r_Wpct=0.486
      xGA-only dWAR blend r_GD=0.714   r_Wpct=0.371
      60/40 blend         r_GD=0.714   r_Wpct=0.371  (inside plateau)

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

# Blend weights (must sum to 1)
XGA_WEIGHT = 0.60
BOX_WEIGHT = 0.40


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PWHL blended WAR (60% xGA / 40% box)")
    p.add_argument("--box-csv", default="pwhl_war_results.csv")
    p.add_argument("--xga-csv", default="pwhl_xga_war_results.csv")
    p.add_argument("--output",  default="pwhl_combined_war_results.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log.info("Loading box WAR from %s", args.box_csv)
    box = pd.read_csv(args.box_csv)
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
    print(f"\nTop 15 (all seasons, 60% xGA / 40% box dWAR blend):")
    top = out.nlargest(15, "WAR")[
        ["Season", "name", "team", "position", "GP", "oWAR", "box_dWAR", "xga_dWAR", "dWAR", "WAR"]
    ]
    top = top.round(3)
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
