"""
processor.py
------------
Converts raw PWHL play-by-play data into "stints" — the fundamental
unit of analysis for RAPM.

A stint is a contiguous period of play where the exact same skaters
are on the ice for both teams.  For each stint we record:
  - duration (seconds)
  - the 5 (or 4/6 for PP/SH) skater IDs for each team
  - goals for / against (from the home team's perspective)
  - strength (EV / PP / SH)

If on-ice player data is absent from the PbP (a known PWHL data gap),
the module falls back to a box-score approximation using +/- and
individual point totals per game.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

PERIOD_LENGTH_SECS = 20 * 60   # 20-minute periods
OT_LENGTH_SECS     = 5  * 60   # 5-minute OT in PWHL regular season


@dataclass
class Stint:
    """One contiguous on-ice shift for a fixed group of skaters."""
    game_id:        str
    period:         int
    start_secs:     int           # seconds elapsed within period
    end_secs:       int
    home_skaters:   tuple[str, ...]
    away_skaters:   tuple[str, ...]
    home_goals:     int = 0
    away_goals:     int = 0
    strength:       str = "EV"   # EV, PP, SH, EN

    @property
    def duration_secs(self) -> int:
        return max(0, self.end_secs - self.start_secs)

    @property
    def duration_min(self) -> float:
        return self.duration_secs / 60.0

    def to_dict(self) -> dict:
        return {
            "game_id":      self.game_id,
            "period":       self.period,
            "start_secs":   self.start_secs,
            "end_secs":     self.end_secs,
            "duration_secs":self.duration_secs,
            "home_skaters": list(self.home_skaters),
            "away_skaters": list(self.away_skaters),
            "home_goals":   self.home_goals,
            "away_goals":   self.away_goals,
            "strength":     self.strength,
        }


class PWHLProcessor:
    """
    Converts a list of normalised game PbP dicts (from PWHLScraper) into
    a DataFrame of Stints ready for RAPM regression.

    Usage
    -----
    proc   = PWHLProcessor()
    stints = proc.build_stints(all_pbp_list)   # list from scraper.get_all_pbp()
    df     = proc.stints_to_dataframe(stints)
    """

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_stints(self, games: list[dict]) -> list[Stint]:
        """
        Process all games and return a flat list of Stint objects.

        If on-ice player data is present (has_on_ice_data=True) the full
        event-driven stint builder is used.  Otherwise a game-level box-score
        fallback is used and a warning is logged.
        """
        all_stints: list[Stint] = []
        full_data_count = 0

        for game in games:
            gid = game.get("game_id", "unknown")
            if game.get("has_on_ice_data"):
                stints = self._build_stints_from_pbp(game)
                full_data_count += 1
            else:
                log.warning(
                    "Game %s has no on-ice player data; using box-score fallback.", gid
                )
                stints = self._build_stints_fallback(game)

            all_stints.extend(stints)

        log.info(
            "Built %d stints from %d games (%d with full PbP, %d with fallback).",
            len(all_stints),
            len(games),
            full_data_count,
            len(games) - full_data_count,
        )
        return all_stints

    def stints_to_dataframe(self, stints: list[Stint]) -> pd.DataFrame:
        """
        Convert a list of Stint objects into a long-form DataFrame.
        Each row is one stint; skaters are stored as lists in the
        'home_skaters' and 'away_skaters' columns.
        """
        if not stints:
            return pd.DataFrame()
        rows = [s.to_dict() for s in stints]
        df   = pd.DataFrame(rows)
        df   = df[df["duration_secs"] > 0].copy()
        log.info("Stint DataFrame shape: %s", df.shape)
        return df

    def get_player_toi(self, stints: list[Stint]) -> pd.DataFrame:
        """
        Compute total time-on-ice (seconds) for each player across all stints.

        Returns a DataFrame with columns: player_id, toi_secs, toi_min.
        """
        toi: defaultdict[str, int] = defaultdict(int)
        for s in stints:
            for pid in s.home_skaters + s.away_skaters:
                toi[pid] += s.duration_secs

        if not toi:
            return pd.DataFrame(columns=["player_id", "toi_secs", "toi_min"])

        df = pd.DataFrame(
            [{"player_id": k, "toi_secs": v} for k, v in toi.items()]
        )
        df["toi_min"] = df["toi_secs"] / 60
        return df.sort_values("toi_min", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Stint builder: full PbP path
    # ------------------------------------------------------------------

    def _build_stints_from_pbp(self, game: dict) -> list[Stint]:
        """
        Build stints from event-level on-ice data.

        Strategy
        --------
        Walk through plays in chronological order.  Whenever the set of
        on-ice players changes (detected from the home_on_ice / away_on_ice
        fields on each event), close the current stint and open a new one.
        Goals within a stint are tallied.
        """
        gid        = game["game_id"]
        home_id    = game["home_team_id"]
        plays      = sorted(
            game.get("plays", []),
            key=lambda p: (p["period"], p["time_secs"])
        )

        stints: list[Stint] = []

        # Current stint state
        cur_period:   int            = 0
        cur_start:    int            = 0
        cur_home:     tuple[str,...] = ()
        cur_away:     tuple[str,...] = ()
        cur_strength: str            = "EV"
        home_goals:   int            = 0
        away_goals:   int            = 0

        def close_stint(end_secs: int) -> None:
            if cur_home and cur_away and end_secs > cur_start:
                stints.append(Stint(
                    game_id      = gid,
                    period       = cur_period,
                    start_secs   = cur_start,
                    end_secs     = end_secs,
                    home_skaters = cur_home,
                    away_skaters = cur_away,
                    home_goals   = home_goals,
                    away_goals   = away_goals,
                    strength     = cur_strength,
                ))

        for play in plays:
            period    = play["period"]
            t         = play["time_secs"]
            event     = play["event"]
            strength  = play.get("strength", "EV")

            h_on = tuple(sorted(play["home_on_ice"]))
            a_on = tuple(sorted(play["away_on_ice"]))

            # Detect lineup change or period change
            lineup_changed = (
                (h_on and h_on != cur_home)
                or (a_on and a_on != cur_away)
                or period != cur_period
            )

            if lineup_changed:
                close_stint(t)
                # Reset
                cur_period   = period
                cur_start    = t
                cur_home     = h_on if h_on else cur_home
                cur_away     = a_on if a_on else cur_away
                cur_strength = strength
                home_goals   = 0
                away_goals   = 0

            # Tally goals
            if event == "goal":
                if play["team_id"] == home_id:
                    home_goals += 1
                else:
                    away_goals += 1

        # Close final stint at end of last period
        period_end = PERIOD_LENGTH_SECS if cur_period <= 3 else OT_LENGTH_SECS
        close_stint(period_end)

        log.debug("Game %s: %d stints from full PbP", gid, len(stints))
        return stints

    # ------------------------------------------------------------------
    # Stint builder: box-score fallback
    # ------------------------------------------------------------------

    def _build_stints_fallback(self, game: dict) -> list[Stint]:
        """
        When on-ice player lists are unavailable, create a single synthetic
        'stint' per game using box-score data.

        This is far less accurate than real stints but allows partial results
        until better data is available.  The game counts as one 60-minute
        unit with goals for/against.

        Skater lists come from the roster (everyone who played); we can't
        distinguish line combinations, so we use all skaters ranked by
        (estimated) ice time.
        """
        gid     = game["game_id"]
        meta    = game.get("game_meta", {})

        try:
            home_goals = int(meta.get("home_goal_count", 0))
            away_goals = int(meta.get("visiting_goal_count", 0))
        except (ValueError, TypeError):
            home_goals = away_goals = 0

        # Use roster players as proxy — top-5 forwards/D by jersey number
        # (a rough approximation; better than nothing)
        home_skaters = tuple(list(game.get("home_roster", {}).keys())[:5])
        away_skaters = tuple(list(game.get("visiting_roster", {}).keys())[:5])

        if not home_skaters or not away_skaters:
            log.warning("Game %s: empty rosters in fallback.", gid)
            return []

        return [Stint(
            game_id      = gid,
            period       = 0,             # 0 = whole game
            start_secs   = 0,
            end_secs     = 3600,          # 60 minutes
            home_skaters = home_skaters,
            away_skaters = away_skaters,
            home_goals   = home_goals,
            away_goals   = away_goals,
            strength     = "EV",
        )]

    # ------------------------------------------------------------------
    # Design matrix builder (used directly by rapm.py)
    # ------------------------------------------------------------------

    @staticmethod
    def build_design_matrix(
        stints_df: pd.DataFrame,
        player_list: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the sparse RAPM design matrix.

        For each stint (row), each player column is:
          +1  if the player is a home-team skater in that stint
          -1  if the player is an away-team skater in that stint
           0  otherwise

        Parameters
        ----------
        stints_df   : DataFrame from stints_to_dataframe()
        player_list : ordered list of player_ids (defines column order)

        Returns
        -------
        X       : (n_stints, n_players) float array
        y       : (n_stints,) float — home goals per 60 min per stint
        weights : (n_stints,) float — stint duration in minutes (regression weight)
        """
        player_idx = {pid: i for i, pid in enumerate(player_list)}
        n_stints   = len(stints_df)
        n_players  = len(player_list)

        X       = np.zeros((n_stints, n_players), dtype=np.float32)
        weights = np.zeros(n_stints,              dtype=np.float32)
        y       = np.zeros(n_stints,              dtype=np.float32)

        for row_i, (_, row) in enumerate(stints_df.iterrows()):
            dur_min = row["duration_secs"] / 60.0
            if dur_min <= 0:
                continue
            weights[row_i] = dur_min

            # Goals per 60 for this stint (home perspective)
            net_goals       = row["home_goals"] - row["away_goals"]
            y[row_i]        = (net_goals / dur_min) * 60.0

            for pid in row["home_skaters"]:
                if pid in player_idx:
                    X[row_i, player_idx[pid]] = 1.0

            for pid in row["away_skaters"]:
                if pid in player_idx:
                    X[row_i, player_idx[pid]] = -1.0

        return X, y, weights
