"""
coord_loader.py
---------------
Fetch play-by-play shot data from the PWHL hockey-statistics.com API and
return a tidy DataFrame with shot coordinates and per-event xG.

Endpoint
--------
  GET https://pwhl.hockey-statistics.com/api/data/pbp
  Query params: season (e.g. "2024/2025"), seasonState ("Regular Season")

Response schema (per-row fields used here)
------------------------------------------
  event        : str  — "Shot", "Goal", "Block", "Penalty"
  strength     : str  — "5v5", "5v4", "4v5", "PP", "SH", etc.
                        (normalised to "PP" / "SH" / "EV" / "EN" on load)
  x            : float | null
  y            : float | null
  xG           : float | null  (null for blocked shots)
  shooter      : str  — shooter name (mapped to "player" in output)
  game_id      : int | str
  season       : str  — "2024/2025" etc.

Usage
-----
  loader = CoordLoader()
  pbp    = loader.fetch_pbp(["2024-25"])   # returns pd.DataFrame
"""

from __future__ import annotations

import logging
import time
from typing import Sequence

import pandas as pd
import requests

from .constants import SEASON_YEARS

log = logging.getLogger(__name__)

API_PBP_URL = "https://pwhl.hockey-statistics.com/api/data/pbp"


class CoordLoader:
    """
    Fetch coordinate-level shot data from the PWHL PBP API.

    Parameters
    ----------
    delay : float
        Polite delay in seconds between API calls (default 0.5).
    """

    def __init__(self, delay: float = 0.5):
        self.delay = delay
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "Mozilla/5.0 (compatible; PWHL-WAR/1.0)"

    def fetch_pbp(
        self,
        seasons: Sequence[str],
        events: tuple[str, ...] = ("Shot", "Goal"),
        season_state: str = "Regular Season",
        drop_blocked: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch play-by-play shot data for the requested seasons.

        Parameters
        ----------
        seasons      : list of season labels e.g. ["2023-24", "2024-25"]
        events       : event types to include (default Shot + Goal)
        season_state : "Regular Season" or "Playoffs"
        drop_blocked : if True, drop rows where xG is null (blocked shots)

        Returns
        -------
        DataFrame with columns:
          season, game_id, event, strength, player, x, y, xG
        """
        frames: list[pd.DataFrame] = []

        for season in seasons:
            api_season = SEASON_YEARS.get(season, season)
            log.info("Fetching PBP: %s (%s) ...", season, api_season)
            try:
                df = self._fetch_season(api_season, season_state)
            except Exception as exc:
                log.warning("Failed to fetch PBP for %s: %s", season, exc)
                continue

            if df.empty:
                log.warning("No data returned for %s", season)
                continue

            # Filter to requested event types (API may use "event" or "Event")
            ev_col = "event" if "event" in df.columns else "Event" if "Event" in df.columns else None
            if ev_col and events:
                df = df[df[ev_col].isin(events)]

            # Standardise columns
            df = self._standardise(df, season)

            if drop_blocked:
                df = df[df["xG"].notna()]

            frames.append(df)
            time.sleep(self.delay)

        if not frames:
            return pd.DataFrame(columns=["season", "game_id", "event",
                                         "strength", "player", "x", "y", "xG"])

        result = pd.concat(frames, ignore_index=True)
        # Ensure numeric types
        for col in ["x", "y", "xG"]:
            result[col] = pd.to_numeric(result[col], errors="coerce")

        log.info("Fetched %d shot events across %d season(s)",
                 len(result), len(frames))
        return result

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _fetch_season(self, api_season: str, season_state: str) -> pd.DataFrame:
        """Make the API call and return raw DataFrame."""
        resp = self._session.get(
            API_PBP_URL,
            params={"season": api_season, "seasonState": season_state},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        # API may return list directly or wrapped in a key
        if isinstance(data, list):
            return pd.DataFrame(data)
        for key in ("rows", "data", "plays", "events", "pbp"):
            if key in data:
                return pd.DataFrame(data[key])
        return pd.DataFrame(data)

    def _standardise(self, df: pd.DataFrame, season: str) -> pd.DataFrame:
        """Rename API fields to canonical output schema.

        Handles two known API formats:
          Old / documented: Event, Strength, "Player 1", GameID  (Title Case)
          Current live API: event, strength, shooter, game_id    (snake_case)
        """
        rename = {
            # Title-case variants (original documented schema)
            "Event":    "event",
            "Strength": "strength",
            "Player 1": "player",
            "GameID":   "game_id",
            # snake_case variants (current live API)
            "shooter":  "player",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        # Ensure required columns exist
        for col in ["event", "strength", "player", "game_id", "x", "y", "xG"]:
            if col not in df.columns:
                df[col] = None

        # Normalise strength codes so CoordXGModel's PP flag works.
        # Live API uses numeric strength strings ("5v5", "5v4", "4v5", …).
        # We simplify: first number > second → "PP"; < → "SH"; equal → "EV".
        if df["strength"].dtype == object:
            df["strength"] = df["strength"].map(self._norm_strength).fillna("EV")

        df["season"] = season
        return df[["season", "game_id", "event", "strength", "player", "x", "y", "xG"]]

    @staticmethod
    def _norm_strength(s: str) -> str:
        """Map numeric strength codes to PP / SH / EV / EN."""
        if not isinstance(s, str):
            return "EV"
        s_up = s.upper()
        if s_up.startswith("EN"):
            return "EN"
        parts = s_up.split("V")
        if len(parts) == 2:
            try:
                a, b = int(parts[0]), int(parts[1])
                if a > b:
                    return "PP"
                if a < b:
                    return "SH"
                return "EV"
            except ValueError:
                pass
        # Already normalised ("PP", "SH", "EV") or unknown — return as-is
        return s
