"""
regen_2526.py
-------------
Regenerate war_2526.csv for the complete 2025-26 season (all 120 games).

Run from Projects/PWHL_war/:
    /c/Users/morga/AppData/Local/Python/bin/python.exe regen_2526.py
"""

import sys
import logging
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent))

from pwhl_war.xga_war   import XGAWar, PWHLApiLoader, _pid_from_url, _parse_toi
from pwhl_war.coord_loader import CoordLoader
from pwhl_war.box_war   import XGWar
from pwhl_war.io_utils  import load_blocks
from pwhl_war.constants import HOCKEYTECH_API_KEY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("regen_2526")

SEASON          = "2025-26"
MIN_TOI         = 50.0
DEFENSE_WEIGHT  = -0.10
BLOCK_WEIGHT    = 0.04
REPLACEMENT_PCT = 25.0
GPW_FULL        = 4.6667   # 120-game actual: 560 goals / 120 games / 2 teams

OUT_CSV = Path("pwhl_war/data/war_2526.csv")


# ---------------------------------------------------------------------------
# Step 0: Fetch fresh blocks from HockeyTech API (season_id=3 for 2025-26)
# ---------------------------------------------------------------------------
def fetch_blocks_ht(season_id: int = 3) -> pd.DataFrame:
    """Fetch season-level blocked shots per player from HockeyTech API."""
    log.info("Fetching blocks from HockeyTech API (season_id=%d) ...", season_id)
    # HockeyTech uses JSONP; strip the callback wrapper by requesting with a
    # known callback name, then slicing the text to extract raw JSON.
    cb = "_pwhlBlk"
    params = {
        "feed": "statviewfeed", "view": "players",
        "season": str(season_id),
        "key": HOCKEYTECH_API_KEY, "client_code": "pwhl",
        "team": "all", "position": "skaters",
        "sort": "blocked", "limit": "500",
        "callback": cb,
    }
    try:
        r = requests.get(
            "https://lscluster.hockeytech.com/feed/index.php",
            params=params, timeout=20,
        )
        r.raise_for_status()
        import re, json as _json
        text = r.text.strip()
        # Strip callback wrapper: _pwhlBlk([...])  →  [...]
        m = re.match(r"[^(]+\((.*)\)\s*;?\s*$", text, re.DOTALL)
        raw = m.group(1) if m else text
        data = _json.loads(raw)
        rows = data[0]["sections"][0]["data"]
        blk = []
        for item in rows:
            row = item["row"]
            pid = int(row.get("player_id", 0) or 0)
            b   = int(row.get("shots_blocked_by_player", 0) or 0)
            if pid:
                blk.append({"PlayerID": pid, "blocks": b})
        df = pd.DataFrame(blk)
        log.info("HockeyTech blocks: %d players", len(df))
        return df
    except Exception as exc:
        log.warning("HockeyTech blocks fetch failed (%s) — falling back to cached CSV", exc)
        cached = load_blocks(SEASON)
        return cached if cached is not None else pd.DataFrame()


# ---------------------------------------------------------------------------
# Step 1: Get all 120 game IDs
# ---------------------------------------------------------------------------
log.info("=== Step 1: Game IDs for %s ===", SEASON)
loader = PWHLApiLoader()
game_ids, sched_map = loader.get_game_ids(SEASON)
log.info("Total games: %d", len(game_ids))


# ---------------------------------------------------------------------------
# Step 2: Build game_data_df from all game summaries
#   - One row per player per game (same format as hockeystats CSV)
#   - EV_ixG=0 (will be replaced by coord xG via pbp_df in box_war.fit)
# ---------------------------------------------------------------------------
log.info("=== Step 2: Build game_data_df from summaries (%d games) ===", len(game_ids))
records = []
for i, gid in enumerate(game_ids):
    if i % 20 == 0:
        log.info("  [%d/%d] game %d ...", i + 1, len(game_ids), gid)
    try:
        summ = loader.get_summary(gid)
    except Exception as exc:
        log.warning("  Game %d summary failed: %s — skipping", gid, exc)
        continue

    sm        = sched_map.get(gid, {})
    home_abbr = sm.get("home_abbr", "UNK")
    away_abbr = sm.get("away_abbr", "UNK")

    for side, team_abbr in [("homeTeam", home_abbr), ("visitingTeam", away_abbr)]:
        skaters = summ.get(side, {}).get("skaters", [])
        for sk in skaters:
            pid = _pid_from_url(sk.get("playerImageURL", ""))
            if not pid:
                continue
            toi = _parse_toi(sk.get("stats", {}).get("toi", ""))
            if toi <= 0:
                continue
            st = sk.get("stats", {})
            raw_pos = str(sk.get("position", "F")).upper()
            norm_pos = "D" if raw_pos in {"LD", "RD", "D"} else "F"
            records.append({
                "GameID":      gid,
                "Season":      "20252026",
                "SeasonStage": "Regular",
                "Team":        team_abbr,
                "PlayerID":    pid,
                "Name":        sk.get("name", str(pid)),
                "position":    norm_pos,
                "TOI":         toi,
                # xG columns start at 0; coord xG replaces total_ixG in box_war.fit
                "EV_ixG":      0.0,
                "PP_ixG":      0.0,
                "SH_ixG":      0.0,
                # Goals/assists as totals (only needed for output, not WAR math)
                "EV_G":   st.get("goals", 0), "PP_G": 0, "SH_G": 0, "EN_G": 0,
                "EV_A1":  st.get("assists", 0), "EV_A2": 0,
                "PP_A1":  0, "PP_A2": 0, "SH_A1": 0, "SH_A2": 0,
                "plusMinus": st.get("plusMinus", 0),
                "PIM":       st.get("penaltyMinutes", 0),
                "hits":      st.get("hits", 0),
            })

game_data_df = pd.DataFrame(records)
log.info(
    "game_data_df: %d rows, %d games, %d unique players",
    len(game_data_df),
    game_data_df["GameID"].nunique(),
    game_data_df["PlayerID"].nunique(),
)


# ---------------------------------------------------------------------------
# Step 3: Fetch coord PBP for current season (apply data)
# ---------------------------------------------------------------------------
log.info("=== Step 3: Fetch coord PBP for %s ===", SEASON)
coord_df = CoordLoader().fetch_pbp([SEASON])
log.info("Coord PBP: %d shot events", len(coord_df))


# ---------------------------------------------------------------------------
# Step 3b: Fetch training PBP (2023-24 + 2024-25) for CoordXGModel
#   Train on historical seasons → apply to 2025-26 (out-of-sample).
# ---------------------------------------------------------------------------
log.info("=== Step 3b: Fetch training PBP (2023-24 + 2024-25) ===")
train_coord_df = CoordLoader().fetch_pbp(["2023-24", "2024-25"])
log.info("Training PBP: %d shot events", len(train_coord_df))


# ---------------------------------------------------------------------------
# Step 4: Build fa_df (xGA per player, EV-only, via XGAWar + coord_df)
# ---------------------------------------------------------------------------
log.info("=== Step 4: Build xGA per player (EV-only, coord xG) ===")
coord_s = coord_df[coord_df["season"] == SEASON] if "season" in coord_df.columns else coord_df
fa_df = XGAWar.build_fa_season(SEASON, loader, coord_df=coord_s)
log.info("fa_df: %d players", len(fa_df))


# ---------------------------------------------------------------------------
# Step 5: Blocks and OZS%
# ---------------------------------------------------------------------------
log.info("=== Step 5: Blocks and OZS%% ===")
blocks_df = fetch_blocks_ht(season_id=3)
if blocks_df.empty:
    blocks_df = load_blocks(SEASON)
    log.info("Using cached blocks CSV")

ozs_path = Path("pwhl_war/data/raw/ozs_2025-26.csv")
if ozs_path.exists():
    ozs_df = pd.read_csv(ozs_path)
    log.info("OZS%%: %d players (partial season — acceptable)", len(ozs_df))
else:
    ozs_df = None
    log.warning("No OZS%% file found — skipping")


# ---------------------------------------------------------------------------
# Step 6: Run box_war model
# ---------------------------------------------------------------------------
log.info("=== Step 6: Run WAR model ===")
model = XGWar(
    min_toi_min     = MIN_TOI,
    replacement_pct = REPLACEMENT_PCT,
    defense_weight  = DEFENSE_WEIGHT,
    block_weight    = BLOCK_WEIGHT,
    goals_per_win   = GPW_FULL,
    team_adjust     = False,
)
model.fit(
    game_data_df,
    pbp_df        = coord_df,
    train_pbp_df  = train_coord_df,
    fa_df         = fa_df,
    ozs_df        = ozs_df,
    blocks_df     = blocks_df if (blocks_df is not None and not blocks_df.empty) else None,
)

war_df = model.get_war()
war_df["Season"] = SEASON

log.info("WAR results: %d qualified players (min_toi=%.0f min)", len(war_df), MIN_TOI)
log.info("Goals per win: %.4f", model.goals_per_win_)
log.info("Top-5 WAR:")
print(war_df[["name", "team", "pos", "gp", "toi_min", "oWAR", "dWAR", "WAR"]].head(5).to_string(index=False))


# ---------------------------------------------------------------------------
# Step 7: Save
# ---------------------------------------------------------------------------
log.info("=== Step 7: Save ===")
war_df.to_csv(OUT_CSV, index=False, encoding="utf-8")
log.info("Saved: %s (%d rows)", OUT_CSV, len(war_df))
log.info("Done.")
