"""
Phase 4: OZS% Extraction Pipeline
-----------------------------------
Parses faceoff events from all cached PBP JSON files to compute per-player
offensive zone start percentage (OZS%) for each season.

Algorithm:
  1. For each game: load pbp_{gid}.json + summ_{gid}.json
  2. Identify home team players from summ's playerImageURL fields
  3. Calibrate home team's attack direction per period from shot xLocations
  4. Classify each faceoff as OZ/DZ/NZ relative to each participant
  5. Aggregate to season totals; compute ozs_pct = oz_fo / (oz_fo + dz_fo)
  6. Impute team mean OZS% for players with 0 faceoffs (non-centers)
  7. Save to pwhl_war/data/raw/ozs_{season}.csv

Zone thresholds (from faceoff dot positions):
  x <= 175  →  left end zone
  175 < x < 425  →  neutral zone
  x >= 425  →  right end zone

Run: py -3 Projects/PWHL_war/extract_ozs.py
"""

import sys
import json
import logging
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from pwhl_war.xga_war import XGAWar, PWHLApiLoader, _pid_from_url

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("extract_ozs")

CACHE    = Path(__file__).parent / "pwhl_war" / "data" / "raw" / "pbp_cache"
OUT_DIR  = Path(__file__).parent / "pwhl_war" / "data" / "raw"
SEASONS  = ["2023-24", "2024-25", "2025-26"]
LOADER   = PWHLApiLoader(cache_dir=CACHE)

# Zone boundaries (faceoff dot x-coordinates: 100, 143, 457, 500 in end zones)
LEFT_END_MAX  = 175
RIGHT_END_MIN = 425
MID_POINT     = 300   # used for direction calibration (shot counts each side)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_home_player_ids(summ: dict) -> set:
    """Extract numeric player IDs for all home team players from summ JSON."""
    pids = set()
    home = summ.get("homeTeam", {})
    for skater in home.get("skaters", []):
        pid = _pid_from_url(skater.get("playerImageURL", ""))
        if pid:
            pids.add(pid)
    for goalie in home.get("goalies", []):
        pid = _pid_from_url(goalie.get("playerImageURL", ""))
        if pid:
            pids.add(pid)
    return pids


def classify_zone(x: int) -> str:
    """Absolute zone from faceoff x-coordinate."""
    if x <= LEFT_END_MAX:
        return "left_end"
    elif x >= RIGHT_END_MIN:
        return "right_end"
    return "neutral"


def calibrate_directions(pbp: list, home_pids: set) -> dict:
    """
    Return {period_id: "home_right" | "home_left"} by counting home team shots
    on each half of the ice per period.

    "home_right" → home team OZ is the right end zone (x >= 425).
    Gaps are filled using alternating logic (teams switch ends each period).
    """
    period_counts: dict[str, dict] = defaultdict(lambda: {"left": 0, "right": 0})

    for e in pbp:
        if e.get("event") != "shot":
            continue
        d = e.get("details", {})
        shooter_id = d.get("shooter", {}).get("id")
        if shooter_id not in home_pids:
            continue
        x      = d.get("xLocation", MID_POINT)
        period = str(d.get("period", {}).get("id", "1"))
        if x <= MID_POINT:
            period_counts[period]["left"] += 1
        else:
            period_counts[period]["right"] += 1

    # Determine direction for each calibrated period
    directions: dict[str, str | None] = {}
    for period, counts in period_counts.items():
        total = counts["left"] + counts["right"]
        if total == 0:
            directions[period] = None
        else:
            directions[period] = "home_right" if counts["right"] >= counts["left"] else "home_left"

    # Fill unknown periods using alternating logic
    # Gather all periods from faceoff events in case a period has no shots
    fo_periods = set(
        str(e["details"].get("period", {}).get("id", "1"))
        for e in pbp if e.get("event") == "faceoff"
    )
    all_periods = set(directions.keys()) | fo_periods
    sorted_periods = sorted(all_periods, key=lambda p: (int(p) if p.isdigit() else 99))

    # Forward-fill using alternation
    last_known: str | None = None
    for p in sorted_periods:
        if p not in directions or directions[p] is None:
            if last_known is not None:
                directions[p] = "home_left" if last_known == "home_right" else "home_right"
        if directions.get(p) is not None:
            last_known = directions[p]

    return directions


def parse_game(gid: int, loader: PWHLApiLoader, home_abbr: str = "", away_abbr: str = "") -> list[dict]:
    """
    Process one game's PBP + summary.

    Returns list of {player_id, name, team, position, oz, dz, nz} for each
    faceoff participant.  'team' is set from home_abbr/away_abbr (sched_map).
    """
    try:
        pbp  = loader.get_pbp(gid)
        summ = loader.get_summary(gid)
    except Exception as exc:
        log.warning("Game %d: load failed (%s) — skipped", gid, exc)
        return []

    home_pids = get_home_player_ids(summ)
    if not home_pids:
        log.debug("Game %d: empty home_pids — skipped", gid)
        return []

    # Build player metadata (name, team, position) from summ
    # Use sched_map abbreviations for real team names
    player_meta: dict[int, dict] = {}
    for side, abbr in [("homeTeam", home_abbr), ("visitingTeam", away_abbr)]:
        for skater in summ.get(side, {}).get("skaters", []):
            pid = _pid_from_url(skater.get("playerImageURL", ""))
            if pid:
                player_meta[pid] = {
                    "name":     skater.get("name", ""),
                    "team":     abbr,
                    "position": skater.get("position", ""),
                }

    directions = calibrate_directions(pbp, home_pids)

    # Accumulate faceoff counts per player in this game
    game_counts: dict[int, dict] = defaultdict(lambda: {"oz": 0, "dz": 0, "nz": 0})

    for e in pbp:
        if e.get("event") != "faceoff":
            continue
        d         = e.get("details", {})
        home_pid  = d.get("homePlayer",     {}).get("id")
        visit_pid = d.get("visitingPlayer", {}).get("id")
        x         = d.get("xLocation", MID_POINT)
        period    = str(d.get("period", {}).get("id", "1"))

        zone_abs  = classify_zone(x)
        direction = directions.get(period)

        if zone_abs == "neutral" or direction is None:
            home_zone = "nz"
        elif direction == "home_right":
            home_zone = "oz" if zone_abs == "right_end" else "dz"
        else:  # home_left
            home_zone = "oz" if zone_abs == "left_end" else "dz"

        visit_zone = "nz" if home_zone == "nz" else ("dz" if home_zone == "oz" else "oz")

        if home_pid:
            game_counts[home_pid][home_zone]  += 1
        if visit_pid:
            game_counts[visit_pid][visit_zone] += 1

    # Convert to list of records
    records = []
    for pid, counts in game_counts.items():
        meta = player_meta.get(pid, {"name": "", "team": "", "position": ""})
        records.append({
            "player_id": pid,
            "name":      meta["name"],
            "team":      meta["team"],
            "position":  meta["position"],
            "oz":        counts["oz"],
            "dz":        counts["dz"],
            "nz":        counts["nz"],
        })
    return records


def build_ozs_season(season: str, loader: PWHLApiLoader) -> pd.DataFrame:
    """
    Aggregate per-player faceoff counts across all games in a season.
    Computes ozs_pct = oz_fo / (oz_fo + dz_fo); NZ faceoffs excluded.
    Imputes team mean OZS% for players with 0 end-zone faceoffs.
    """
    # Minimum end-zone faceoffs for a player's OZS% to be trusted (not imputed)
    MIN_END_FO = 10

    game_ids, sched_map = loader.get_game_ids(season)
    print(f"  {season}: {len(game_ids)} games found")

    all_records = []
    for i, gid in enumerate(game_ids):
        if (i + 1) % 20 == 0:
            print(f"    Processed {i+1}/{len(game_ids)} games...")
        sm   = sched_map.get(gid, {})
        home = sm.get("home_abbr", "")
        away = sm.get("away_abbr", "")
        records = parse_game(gid, loader, home_abbr=home, away_abbr=away)
        all_records.extend(records)

    if not all_records:
        print(f"  WARNING: no faceoff records for {season}")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    # Aggregate across games per player (take most-frequent name/team/position)
    agg = df.groupby("player_id", as_index=False).agg(
        name     = ("name",     lambda x: x.mode().iloc[0] if len(x) else ""),
        team     = ("team",     lambda x: x.mode().iloc[0] if len(x) else ""),
        position = ("position", lambda x: x.mode().iloc[0] if len(x) else ""),
        oz_fo    = ("oz",       "sum"),
        dz_fo    = ("dz",       "sum"),
        nz_fo    = ("nz",       "sum"),
    )
    agg["n_fo"]       = agg["oz_fo"] + agg["dz_fo"] + agg["nz_fo"]
    agg["end_zone_fo"]= agg["oz_fo"] + agg["dz_fo"]

    # OZS% = oz / (oz + dz) — NZ excluded, mirrors NHL convention.
    # Only computed when player has >= MIN_END_FO end-zone faceoffs.
    agg["ozs_pct"] = np.where(
        agg["end_zone_fo"] >= MIN_END_FO,
        agg["oz_fo"] / agg["end_zone_fo"],
        np.nan,
    )

    # Impute team mean (then league mean as fallback) for sparse players
    team_mean   = agg.dropna(subset=["ozs_pct"]).groupby("team")["ozs_pct"].mean()
    league_mean = float(agg["ozs_pct"].mean())
    agg["ozs_pct"] = agg.apply(
        lambda r: r["ozs_pct"] if not np.isnan(r["ozs_pct"])
                  else float(team_mean.get(r["team"], league_mean)),
        axis=1,
    )
    n_imputed = agg["end_zone_fo"].lt(MIN_END_FO).sum()
    print(f"    Faceoff-trusted: {len(agg)-n_imputed}  |  Imputed (< {MIN_END_FO} end-zone fo): {n_imputed}")

    agg["season"] = season
    agg.drop(columns=["end_zone_fo"], inplace=True)
    return agg


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def verify_season(df: pd.DataFrame, season: str) -> None:
    print(f"\n  Verification — {season} (n={len(df)} players)")
    print(f"    OZS% distribution: mean={df['ozs_pct'].mean():.3f}  "
          f"std={df['ozs_pct'].std():.3f}  "
          f"min={df['ozs_pct'].min():.3f}  max={df['ozs_pct'].max():.3f}")

    # Top 10 OZS%
    top10 = df.nlargest(10, "ozs_pct")[["name", "position", "team", "oz_fo", "dz_fo", "ozs_pct"]]
    print(f"    Top-10 OZS%:")
    print(top10.to_string(index=False))

    # Bottom 10 OZS%
    bot10 = df.nsmallest(10, "ozs_pct")[["name", "position", "team", "oz_fo", "dz_fo", "ozs_pct"]]
    print(f"    Bottom-10 OZS%:")
    print(bot10.to_string(index=False))

    # Position breakdown
    pos_map = df["position"].str.upper().map(lambda p: "D" if p in {"LD","RD","D"} else "F")
    print(f"    Mean OZS% by position: F={df[pos_map=='F']['ozs_pct'].mean():.3f}  "
          f"D={df[pos_map=='D']['ozs_pct'].mean():.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for season in SEASONS:
        print(f"\nBuilding OZS% for {season}...")
        df = build_ozs_season(season, LOADER)
        if df.empty:
            continue

        out_path = OUT_DIR / f"ozs_{season}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"  Saved: {out_path} ({len(df)} players)")
        verify_season(df, season)

    print("\nDone.")
