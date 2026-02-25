"""
scraper.py
----------
Client for the PWHL HockeyTech API.

Base URL : https://lscluster.hockeytech.com/feed/
API key  : 446521baf8c38984  (community-documented public key)
Client   : pwhl

All responses are cached as JSON files in data/raw/ to avoid
re-fetching and to respect the unofficial nature of this API.
"""

import json
import time
import logging
from pathlib import Path

import requests

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URL    = "https://lscluster.hockeytech.com/feed/index.php"
API_KEY     = "446521baf8c38984"
CLIENT_CODE = "pwhl"
CACHE_DIR   = Path(__file__).parent.parent / "pwhl_war" / "data" / "raw"

# Polite delay between API calls (seconds)
REQUEST_DELAY = 0.5


class PWHLScraper:
    """
    Thin wrapper around the HockeyTech API for the PWHL.

    Usage
    -----
    scraper = PWHLScraper()
    seasons  = scraper.get_seasons()
    schedule = scraper.get_schedule(season_id="1")
    pbp      = scraper.get_game_pbp(game_id="12345")
    """

    def __init__(self, cache_dir: Path = CACHE_DIR, delay: float = REQUEST_DELAY):
        self.session   = requests.Session()
        self.cache_dir = Path(cache_dir)
        self.delay     = delay
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cache_path(self, name: str) -> Path:
        return self.cache_dir / f"{name}.json"

    def _load_cache(self, name: str):
        p = self._cache_path(name)
        if p.exists():
            log.debug("Cache hit: %s", name)
            return json.loads(p.read_text(encoding="utf-8"))
        return None

    def _save_cache(self, name: str, data) -> None:
        self._cache_path(name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _get(self, params: dict, cache_key: str):
        """
        Make a GET request to the HockeyTech API, returning parsed JSON.
        Results are cached; cached results are returned on subsequent calls.
        """
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        base_params = {
            "key":         API_KEY,
            "client_code": CLIENT_CODE,
            "fmt":         "json",
        }
        base_params.update(params)

        time.sleep(self.delay)
        try:
            resp = self.session.get(BASE_URL, params=base_params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            log.error("API request failed (%s): %s", cache_key, exc)
            raise

        self._save_cache(cache_key, data)
        log.debug("Fetched and cached: %s", cache_key)
        return data

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    def get_seasons(self) -> list[dict]:
        """
        Return a list of PWHL seasons.
        Each entry has at minimum: {'season_id', 'season_name'}.
        """
        data = self._get(
            {"feed": "modulekit", "view": "seasons"},
            cache_key="seasons",
        )
        # HockeyTech wraps everything in SiteKit
        seasons = data.get("SiteKit", {}).get("Seasons", [])
        log.info("Found %d seasons", len(seasons))
        return seasons

    def get_schedule(self, season_id: str) -> list[dict]:
        """
        Return all games for the given season_id.
        Each entry has: game_id, date_played, home_team, visiting_team,
        home_goal_count, visiting_goal_count, game_status, etc.
        """
        data = self._get(
            {"feed": "modulekit", "view": "schedule", "season_id": season_id},
            cache_key=f"schedule_s{season_id}",
        )
        games = data.get("SiteKit", {}).get("Schedule", [])
        log.info("Season %s: %d games in schedule", season_id, len(games))
        return games

    def get_game_pbp(self, game_id: str) -> dict:
        """
        Return the full (long-form) play-by-play for a single game.
        Tries the V2 endpoint first (more detail); falls back to V1.

        The returned dict has keys including:
          - 'details'       : game metadata
          - 'plays'         : list of play events
          - 'homeTeam'      : home team info
          - 'visitingTeam'  : visiting team info
          - 'home_team_roster'    : player list
          - 'visiting_team_roster': player list
        """
        # Try V2 first (more detail, may include on-ice players)
        try:
            data = self._get(
                {"feed": "statviewfeed", "view": "playbyplayV2", "game_id": game_id},
                cache_key=f"pbp_v2_g{game_id}",
            )
            # Validate we got useful data
            if self._has_play_data(data):
                return self._normalize_pbp(data, version=2)
        except Exception as exc:
            log.warning("V2 PbP failed for game %s: %s", game_id, exc)

        # Fallback to V1
        data = self._get(
            {"feed": "statviewfeed", "view": "playbyplay", "game_id": game_id},
            cache_key=f"pbp_v1_g{game_id}",
        )
        return self._normalize_pbp(data, version=1)

    def get_game_summary(self, game_id: str) -> dict:
        """
        Return the game summary, which often contains shift/TOI information
        not present in the play-by-play endpoint.
        """
        return self._get(
            {"feed": "statviewfeed", "view": "gameSummary", "game_id": game_id},
            cache_key=f"summary_g{game_id}",
        )

    def get_player_stats(self, season_id: str, position: str = "skaters") -> list[dict]:
        """
        Return league-wide player stats for a season.
        position: 'skaters' or 'goalies'
        """
        data = self._get(
            {
                "feed":     "statviewfeed",
                "view":     "players",
                "season":   season_id,
                "team":     "all",
                "position": position,
            },
            cache_key=f"playerstats_s{season_id}_{position}",
        )
        players = (
            data.get("SiteKit", {}).get("Players", [])
            or data.get("SiteKit", {}).get("Goalies", [])
        )
        log.info("Season %s %s: %d players", season_id, position, len(players))
        return players

    def get_team_stats(self, season_id: str) -> list[dict]:
        """Return per-team stats for the season."""
        data = self._get(
            {"feed": "statviewfeed", "view": "teams", "season": season_id},
            cache_key=f"teamstats_s{season_id}",
        )
        return data.get("SiteKit", {}).get("Teams", [])

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def get_all_pbp(
        self,
        season_id: str,
        status_filter: str = "Final",
        max_games: int | None = None,
    ) -> list[dict]:
        """
        Fetch play-by-play for every completed game in the season.

        Parameters
        ----------
        season_id     : HockeyTech season ID string
        status_filter : only fetch games with this game_status (default 'Final')
        max_games     : cap how many games to fetch (useful for testing)

        Returns
        -------
        List of normalised PbP dicts (one per game).
        """
        schedule = self.get_schedule(season_id)
        completed = [
            g for g in schedule
            if g.get("game_status", "").lower() == status_filter.lower()
        ]
        if max_games:
            completed = completed[:max_games]

        log.info("Fetching PbP for %d games...", len(completed))
        results = []
        for i, game in enumerate(completed, 1):
            gid = str(game["id"])
            log.info("[%d/%d] game_id=%s", i, len(completed), gid)
            try:
                pbp = self.get_game_pbp(gid)
                pbp["game_meta"] = game          # attach schedule metadata
                results.append(pbp)
            except Exception as exc:
                log.warning("Skipping game %s: %s", gid, exc)

        log.info("Successfully fetched %d game PbP files", len(results))
        return results

    # ------------------------------------------------------------------
    # Internal normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _has_play_data(data: dict) -> bool:
        """Check whether the API response contains usable play data."""
        sk = data.get("SiteKit", {})
        return bool(
            sk.get("Plays")
            or sk.get("PlayByPlay")
            or sk.get("plays")
        )

    @staticmethod
    def _normalize_pbp(data: dict, version: int) -> dict:
        """
        Normalise the raw HockeyTech PbP response into a consistent structure:

        {
            'game_id'          : str,
            'home_team_id'     : str,
            'visiting_team_id' : str,
            'home_roster'      : {player_id: {name, jersey, position, ...}},
            'visiting_roster'  : {player_id: {name, jersey, position, ...}},
            'plays'            : [
                {
                    'event'        : str,   # 'goal', 'shot', 'penalty', 'faceoff', ...
                    'period'       : int,
                    'time'         : str,   # 'MM:SS'
                    'time_secs'    : int,   # elapsed seconds in period
                    'team_id'      : str,
                    'player_id'    : str,
                    'home_on_ice'  : [player_id, ...],  # may be empty
                    'away_on_ice'  : [player_id, ...],  # may be empty
                    'strength'     : str,   # 'EV', 'PP', 'SH', 'EN'
                    'raw'          : dict,  # original event dict
                }
            ],
            'has_on_ice_data'  : bool,  # True if on-ice player lists are present
        }
        """
        sk = data.get("SiteKit", {})

        # --- Game details ---
        details     = sk.get("details", sk.get("Details", {}))
        game_id     = str(details.get("id", details.get("GameID", "")))
        home_id     = str(details.get("home_team_id", details.get("HomeID", "")))
        visiting_id = str(details.get("visiting_team_id", details.get("VisitorID", "")))

        # --- Rosters ---
        home_roster     = PWHLScraper._parse_roster(
            sk.get("homeTeam", sk.get("HomeTeam", {}))
        )
        visiting_roster = PWHLScraper._parse_roster(
            sk.get("visitingTeam", sk.get("VisitingTeam", {}))
        )

        # --- Play events ---
        raw_plays = (
            sk.get("Plays")
            or sk.get("PlayByPlay")
            or sk.get("plays")
            or []
        )

        plays           = []
        has_on_ice_data = False

        for raw in raw_plays:
            play = PWHLScraper._parse_play(raw)
            if play["home_on_ice"] or play["away_on_ice"]:
                has_on_ice_data = True
            plays.append(play)

        return {
            "game_id":          game_id,
            "home_team_id":     home_id,
            "visiting_team_id": visiting_id,
            "home_roster":      home_roster,
            "visiting_roster":  visiting_roster,
            "plays":            plays,
            "has_on_ice_data":  has_on_ice_data,
        }

    @staticmethod
    def _parse_roster(team_data: dict) -> dict:
        """
        Convert a HockeyTech team block into {player_id: info_dict}.
        HockeyTech uses various field names across versions; we try all.
        """
        players = (
            team_data.get("roster")
            or team_data.get("Roster")
            or team_data.get("players")
            or team_data.get("Players")
            or []
        )
        roster = {}
        for p in players:
            pid = str(p.get("player_id") or p.get("PlayerID") or p.get("id", ""))
            if not pid:
                continue
            roster[pid] = {
                "name":     (p.get("name") or p.get("Name") or
                             f"{p.get('first_name','')} {p.get('last_name','')}").strip(),
                "jersey":   str(p.get("jersey_number") or p.get("JerseyNumber") or ""),
                "position": str(p.get("position") or p.get("Position") or ""),
            }
        return roster

    @staticmethod
    def _parse_play(raw: dict) -> dict:
        """
        Normalise a single play event dict from the HockeyTech API.
        Field names vary between API versions and between event types.
        """
        # Event type
        event = (
            raw.get("event_type")
            or raw.get("EventType")
            or raw.get("type")
            or raw.get("Type")
            or "unknown"
        ).lower()

        # Period
        try:
            period = int(raw.get("period_id") or raw.get("PeriodID") or
                         raw.get("period") or raw.get("Period") or 0)
        except (ValueError, TypeError):
            period = 0

        # Time within period (MM:SS string)
        time_str = str(
            raw.get("time") or raw.get("Time") or
            raw.get("period_time") or raw.get("PeriodTime") or "0:00"
        )
        time_secs = PWHLScraper._time_to_secs(time_str)

        # Team and player
        team_id   = str(raw.get("team_id")   or raw.get("TeamID")   or "")
        player_id = str(raw.get("player_id") or raw.get("PlayerID") or "")

        # On-ice players — HockeyTech uses several naming conventions
        home_on_ice = PWHLScraper._extract_on_ice(raw, "home")
        away_on_ice = PWHLScraper._extract_on_ice(raw, "away")

        # Strength (EV / PP / SH / EN)
        strength = (
            raw.get("strength") or raw.get("Strength") or
            raw.get("situation") or raw.get("Situation") or "EV"
        ).upper()
        # Normalise common variants
        strength_map = {
            "EVEN": "EV", "EVEN STRENGTH": "EV",
            "POWER PLAY": "PP", "SHORT HANDED": "SH",
            "SHORTHANDED": "SH", "EMPTY NET": "EN",
        }
        strength = strength_map.get(strength, strength)

        return {
            "event":       event,
            "period":      period,
            "time":        time_str,
            "time_secs":   time_secs,
            "team_id":     team_id,
            "player_id":   player_id,
            "home_on_ice": home_on_ice,
            "away_on_ice": away_on_ice,
            "strength":    strength,
            "raw":         raw,
        }

    @staticmethod
    def _extract_on_ice(raw: dict, side: str) -> list[str]:
        """
        Try every known HockeyTech field name for on-ice player lists.
        Returns a list of player_id strings (may be empty).
        """
        keys = [
            f"{side}_on_ice",
            f"{side}OnIce",
            f"{side}_players",
            f"{side}Players",
            # Some versions nest under 'players'
        ]
        if side == "home":
            keys += ["HomeOnIce", "home_skaters", "homeSkaters"]
        else:
            keys += ["AwayOnIce", "visiting_on_ice", "VisitingOnIce",
                     "away_skaters", "awaySkaters"]

        for k in keys:
            val = raw.get(k)
            if val:
                if isinstance(val, list):
                    return [str(p) for p in val if p]
                if isinstance(val, str):
                    return [v.strip() for v in val.split(",") if v.strip()]
        return []

    @staticmethod
    def _time_to_secs(time_str: str) -> int:
        """Convert 'MM:SS' to total seconds."""
        try:
            parts = str(time_str).split(":")
            return int(parts[0]) * 60 + int(parts[1])
        except (IndexError, ValueError):
            return 0
