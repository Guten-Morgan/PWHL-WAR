"""
csv_loader.py
-------------
Downloads and parses PWHL data from hockey-statistics.com.

The files are served as CSV despite the .xlsx URL extension.

Tables
------
  players   : player roster (PlayerID, Name, position, Team)
  game_data : per-player per-game stats — the richest table
              Includes individual xG (EV_ixG, PP_ixG, SH_ixG),
              on-ice xGA (EV_xGA, PP_xGA, SH_xGA), TOI, goals, assists
  play_by_play : event-level data (goals, shots, xG per event)
  teams     : team roster info
  schedule  : game schedule/results (GF, GA per team per game)

Usage
-----
  loader = PWHLCsvLoader()
  gd     = loader.get_game_data()     # richest — use this for WAR
  sched  = loader.get_schedule()
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

URLS = {
    "game_data":   "https://hockeyskytte-my.sharepoint.com/:x:/g/personal/lars_hockeyskytte_onmicrosoft_com/EYhR8oBpyEBJq91Qoltvz28BJDukxmKVJFZyjHTqg7vJ8w?download=1",
    "play_by_play":"https://hockeyskytte-my.sharepoint.com/:x:/g/personal/lars_hockeyskytte_onmicrosoft_com/EUInQwTGK6tIsWe1E-XNVp0BL-K4nEEHf6DpUO1JelWmrg?download=1",
    "players":     "https://hockeyskytte-my.sharepoint.com/:x:/g/personal/lars_hockeyskytte_onmicrosoft_com/EULfBkKobJhNoBvv8EIQWlIBip2_z16-myElHwPEY4PZcw?download=1",
    "teams":       "https://hockeyskytte-my.sharepoint.com/:x:/g/personal/lars_hockeyskytte_onmicrosoft_com/ETv4cFHHSshNlvkwrdN1OBYBiFHyQRxzWoJivTrZcRInIw?download=1",
    "schedule":    "https://hockeyskytte-my.sharepoint.com/:x:/g/personal/lars_hockeyskytte_onmicrosoft_com/EYyrLu6h3QVLuHzK_rDxNHIBJ41HdBugd4rExbQwbetULQ?download=1",
}

CACHE_DIR = Path(__file__).parent.parent / "pwhl_war" / "data" / "raw"

# Known season codes in the data
SEASON_CODES = {
    "20232024": "2023-24",
    "20242025": "2024-25",
    "20252026": "2025-26",
}


class PWHLCsvLoader:
    """Download, cache, and parse PWHL data from hockey-statistics.com."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers["User-Agent"] = (
            "Mozilla/5.0 (compatible; PWHL-WAR/1.0)"
        )

    # ------------------------------------------------------------------
    # Public getters
    # ------------------------------------------------------------------

    def get_game_data(
        self,
        season: str | None = None,
        stage: str = "Regular",
    ) -> pd.DataFrame:
        """
        Per-player per-game stats.

        Actual columns (49 total):
          Season, SeasonStage, GameID, Team, Venue, PlayerID, Name, position,
          Starting_G, TOI,
          EV_G, EV_A1, EV_A2, EV_Shots, EV_ixG, EV_GAx,
          PP_G, PP_A1, PP_A2, PP_Shots, PP_ixG, PP_GAx,
          SH_G, SH_A1, SH_A2, SH_Shots, SH_ixG, SH_GAx,
          EN_G, EN_A1, EN_A2,
          EV_GA, EV_SA, EV_xGA, EV_GSAx,
          PP_GA, PP_SA, PP_xGA, PP_GSAx,
          SH_GA, SH_SA, SH_xGA, SH_GSAx,
          PIM, plusMinus, Faceoffs, faceoffWins, hits, Game_No

        Parameters
        ----------
        season : '2023-24', '2024-25', or '2025-26' (None = all seasons)
        stage  : 'Regular', 'Playoff', or None for all stages
        """
        df = self._load_csv("game_data")

        # Coerce numerics
        num_cols = [c for c in df.columns if c not in
                    ("Season", "SeasonStage", "GameID", "Team", "Venue",
                     "Name", "position")]
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        # Filter season
        if season:
            code = {v: k for k, v in SEASON_CODES.items()}.get(season, season)
            df = df[df["Season"].astype(str) == str(code)]

        # Filter stage
        if stage:
            df = df[df["SeasonStage"].str.strip() == stage]

        # Exclude goalies from skater analysis (they're Starting_G = 1 or position = G)
        df = df[df["position"].str.upper() != "G"].copy()

        log.info(
            "game_data: %d player-game rows (season=%s, stage=%s)",
            len(df), season or "all", stage or "all",
        )
        return df

    def get_schedule(
        self,
        season: str | None = None,
        stage: str = "Regular",
    ) -> pd.DataFrame:
        """
        Game schedule and results.

        Columns: Season, SeasonStage, GameID, date, Venue, Team,
                 GameNumber, B2B, game_status, GF, GA
        """
        df = self._load_csv("schedule")
        df["GF"] = pd.to_numeric(df["GF"], errors="coerce")
        df["GA"] = pd.to_numeric(df["GA"], errors="coerce")

        if season:
            code = {v: k for k, v in SEASON_CODES.items()}.get(season, season)
            df = df[df["Season"].astype(str) == str(code)]

        if stage:
            df = df[df["SeasonStage"].str.strip() == stage]

        df = df[df["game_status"].str.strip() == "Final"]
        log.info("schedule: %d team-game rows", len(df))
        return df

    def get_players(self) -> pd.DataFrame:
        """Roster table: PlayerID, Name, position, Team."""
        return self._load_csv("players")

    def get_pbp(self) -> pd.DataFrame:
        """Play-by-play events (note: on-ice columns are empty in current data)."""
        return self._load_csv("play_by_play")

    def refresh(self, table: str | None = None) -> None:
        targets = [table] if table else list(URLS.keys())
        for name in targets:
            p = self._cache_path(name)
            if p.exists():
                p.unlink()
                log.info("Deleted cache: %s", p.name)

    def inspect(self, table: str) -> None:
        df = self._load_csv(table)
        print(f"\n{'='*60}")
        print(f"Table: {table}  |  {len(df)} rows  |  {len(df.columns)} cols")
        print(f"Columns: {list(df.columns)}")
        if len(df):
            print(f"\nFirst row:\n{df.iloc[0].to_dict()}")

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _cache_path(self, name: str) -> Path:
        return self.cache_dir / f"hockeystats_{name}.xlsx"

    def _load_csv(self, name: str) -> pd.DataFrame:
        path = self._cache_path(name)
        if not path.exists():
            self._download(name, path)
        # Files are CSV despite the .xlsx extension
        return pd.read_csv(path, low_memory=False)

    def _download(self, name: str, dest: Path) -> None:
        url = URLS[name]
        log.info("Downloading %s ...", name)
        try:
            resp = self._session.get(url, timeout=60, allow_redirects=True)
            resp.raise_for_status()
            # Sanity check: SharePoint sometimes returns HTML on auth failure
            content = resp.content
            if content[:15].lstrip().startswith(b"<"):
                raise ValueError(
                    f"Got HTML instead of data for '{name}'. "
                    "The SharePoint link may have expired."
                )
            dest.write_bytes(content)
            log.info("  Saved: %s (%.1f KB)", dest.name, len(content) / 1024)
        except requests.RequestException as exc:
            log.error("Download failed for %s: %s", name, exc)
            raise
