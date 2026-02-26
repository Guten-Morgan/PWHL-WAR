"""
run_xga_war.py
--------------
PWHL xGA-based WAR — experimental alternative defensive metric.

Uses Fenwick shot quality (xG from shot events) to compute on-ice xGA
per player via TOI-share attribution, instead of residual plus/minus.

Usage
-----
  python run_xga_war.py --season 2024-25
  python run_xga_war.py --season 2023-24
  python run_xga_war.py --season 2025-26
  python run_xga_war.py --season 2024-25 --no-blocks
  python run_xga_war.py --season 2024-25 --output my_results.csv
"""

import argparse
import logging
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from pwhl_war.xga_war   import XGAWar, PWHLApiLoader
from pwhl_war.constants import DEFAULT_MIN_TOI, DEFAULT_DEFENSE_WEIGHT, DEFAULT_BLOCK_WEIGHT
from pwhl_war.io_utils  import load_blocks

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt= "%H:%M:%S",
)
log = logging.getLogger("run_xga_war")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PWHL xGA-WAR (experimental)")
    p.add_argument("--season", choices=["2023-24", "2024-25", "2025-26"],
                   required=True, help="Season to analyse.")
    p.add_argument("--min-toi",         type=float, default=50.0)
    p.add_argument("--replacement-pct", type=float, default=25.0)
    p.add_argument("--defense-weight",  type=float, default=0.039)
    p.add_argument("--block-weight",    type=float, default=0.04)
    p.add_argument("--no-blocks",       action="store_true")
    p.add_argument("--output",          type=str,
                   default="pwhl_xga_war_results.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    loader = PWHLApiLoader()

    log.info("=== Step 1: Get game IDs (season=%s) ===", args.season)
    game_ids, sched_map = loader.get_game_ids(args.season)
    if not game_ids:
        log.error("No completed games found for %s.", args.season)
        sys.exit(1)

    blocks = None
    if not args.no_blocks:
        blocks = load_blocks(args.season)

    log.info("=== Step 2: Fetch PBP + summaries (%d games) ===", len(game_ids))
    model = XGAWar(
        min_toi_min     = args.min_toi,
        replacement_pct = args.replacement_pct,
        defense_weight  = args.defense_weight,
        block_weight    = args.block_weight,
    )
    model.fit(game_ids, loader, sched_map=sched_map, blocks_df=blocks)

    war_df = model.get_war()
    model.summary(top_n=20)

    name_col = "name"
    print(f"\nSeason summary ({args.season}):")
    print(f"  Qualified players : {len(war_df)}")
    print(f"  Total WAR         : {war_df['WAR'].sum():.2f}")
    print(f"  Max WAR           : {war_df['WAR'].max():.3f}  ({war_df.iloc[0][name_col]})")
    print(f"  Goals per win     : {model.goals_per_win_:.3f}")

    log.info("=== Step 3: Output ===")
    war_df.to_csv(args.output, index=False, encoding="utf-8")
    log.info("Saved: %s", args.output)
    log.info("Done.")


if __name__ == "__main__":
    main()
