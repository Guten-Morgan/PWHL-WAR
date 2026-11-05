"""
baseline_regular.py
-------------------
The current (TOI-share xGA) 2025-26 WAR pipeline, restricted to the 120
regular-season games. `regen_2526.py` takes every "Final" game, which
includes 13 playoff games; this is the like-for-like baseline for the
on-ice experiment (rpi/on-ice-defensive-metric-for-pwhl-war).

Writes rpi/on-ice-defensive-metric-for-pwhl-war/out/baseline_war_2526.csv and
leaves war_2526.csv alone.

Run from Projects/PWHL_war/:
    /c/Users/morga/AppData/Local/Python/bin/python.exe baseline_regular.py
"""

import sys
import logging
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from pwhl_war.xga_war import XGAWar, PWHLApiLoader, _pid_from_url, _parse_toi
from pwhl_war.coord_loader import CoordLoader
from pwhl_war.box_war import XGWar
from pwhl_war.io_utils import load_blocks
from pwhl_war.constants import HOCKEYTECH_API_KEY
from pwhl_war.scope import regular_season_game_ids

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("baseline_regular")

SEASON          = "2025-26"
MIN_TOI         = 50.0
DEFENSE_WEIGHT  = -0.10
BLOCK_WEIGHT    = 0.04
REPLACEMENT_PCT = 25.0
GPW_FULL        = 4.6667   # same as regen_2526.py: 560 goals / 120 games / 2 teams

OUT_DIR = Path("rpi/on-ice-defensive-metric-for-pwhl-war/out")
OUT_CSV = OUT_DIR / "baseline_war_2526.csv"

SCHEDULE_URL = ("https://raw.githubusercontent.com/sportsdataverse/"
                "fastRhockey-pwhl-data/main/pwhl/pwhl_schedule_master.parquet")
SCHEDULE_CACHE = Path("pwhl_war/data/raw/sdv/schedule_master.parquet")


def load_schedule_master() -> pd.DataFrame:
    """Schedule with game_type, downloaded once and cached (git-ignored)."""
    if not SCHEDULE_CACHE.exists():
        SCHEDULE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SCHEDULE_URL, SCHEDULE_CACHE)
    return pd.read_parquet(SCHEDULE_CACHE)


class RegularSeasonLoader(PWHLApiLoader):
    """PWHLApiLoader whose game list is limited to regular-season games, so
    XGAWar.build_fa_season (which calls get_game_ids itself) agrees with
    game_data_df."""

    def __init__(self, regular_ids, **kwargs):
        super().__init__(**kwargs)
        self._regular_ids = set(regular_ids)

    def get_game_ids(self, season):
        game_ids, sched_map = super().get_game_ids(season)
        keep = [g for g in game_ids if g in self._regular_ids]
        return keep, {g: sched_map[g] for g in keep}


# ---------------------------------------------------------------------------
# Step 1: Regular-season game IDs
# ---------------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
regular_ids = regular_season_game_ids(load_schedule_master(), SEASON)
log.info("=== Step 1: regular-season games per schedule master: %d ===", len(regular_ids))
loader = RegularSeasonLoader(regular_ids)
game_ids, sched_map = loader.get_game_ids(SEASON)
log.info("Regular-season games that are Final in the API: %d", len(game_ids))


# ---------------------------------------------------------------------------
# Step 2: Build game_data_df from game summaries (same as regen_2526.py)
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
                "EV_ixG":      0.0,
                "PP_ixG":      0.0,
                "SH_ixG":      0.0,
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
# Step 3: Coordinate PBP (regular season only, as in regen_2526.py) and the
#         2023-24 + 2024-25 training shots for CoordXGModel
# ---------------------------------------------------------------------------
log.info("=== Step 3: Fetch coord PBP for %s ===", SEASON)
coord_df = CoordLoader().fetch_pbp([SEASON])
log.info("Coord PBP: %d shot events", len(coord_df))
train_coord_df = CoordLoader().fetch_pbp(["2023-24", "2024-25"])
log.info("Training PBP: %d shot events", len(train_coord_df))


# ---------------------------------------------------------------------------
# Step 4: xGA per player (EV-only, TOI-share) over the regular-season games
# ---------------------------------------------------------------------------
log.info("=== Step 4: Build xGA per player (EV-only, coord xG) ===")
coord_s = coord_df[coord_df["season"] == SEASON] if "season" in coord_df.columns else coord_df
fa_df = XGAWar.build_fa_season(SEASON, loader, coord_df=coord_s)
log.info("fa_df: %d players", len(fa_df))


# ---------------------------------------------------------------------------
# Step 5: Blocks and OZS% (same sources as regen_2526.py)
# ---------------------------------------------------------------------------
def fetch_blocks_ht(season_id: int = 3) -> pd.DataFrame:
    """Season-level blocked shots per player from the HockeyTech API."""
    import re, json as _json
    import requests
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
        r = requests.get("https://lscluster.hockeytech.com/feed/index.php",
                         params=params, timeout=20)
        r.raise_for_status()
        text = r.text.strip()
        m = re.match(r"[^(]+\((.*)\)\s*;?\s*$", text, re.DOTALL)
        data = _json.loads(m.group(1) if m else text)
        rows = data[0]["sections"][0]["data"]
        blk = [{"PlayerID": int(it["row"].get("player_id", 0) or 0),
                "blocks": int(it["row"].get("shots_blocked_by_player", 0) or 0)}
               for it in rows if int(it["row"].get("player_id", 0) or 0)]
        return pd.DataFrame(blk)
    except Exception as exc:
        log.warning("HockeyTech blocks fetch failed (%s) — falling back to cached CSV", exc)
        cached = load_blocks(SEASON)
        return cached if cached is not None else pd.DataFrame()


log.info("=== Step 5: Blocks and OZS%% ===")
blocks_df = fetch_blocks_ht(season_id=3)
ozs_path = Path("pwhl_war/data/raw/ozs_2025-26.csv")
ozs_df = pd.read_csv(ozs_path) if ozs_path.exists() else None


# ---------------------------------------------------------------------------
# Step 6: Run the WAR model (parameters identical to regen_2526.py)
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


# ---------------------------------------------------------------------------
# Step 7: Save (never overwrites war_2526.csv)
# ---------------------------------------------------------------------------
war_df.to_csv(OUT_CSV, index=False, encoding="utf-8")
log.info("Saved: %s (%d rows)", OUT_CSV, len(war_df))
