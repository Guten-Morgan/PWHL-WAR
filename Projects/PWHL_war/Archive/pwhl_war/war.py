"""
war.py
------
Converts RAPM values (goals above average per 60) into
Wins Above Replacement (WAR) for PWHL skaters.

Pipeline
--------
  1. Goals Above Average (GAA)
       GAA = RAPM  × TOI_min / 60

  2. Replacement Level
       The replacement-level player is defined as the average player
       who would be called up from the PWHL's available talent pool
       (roughly: a player in the bottom quartile by RAPM, or an
       estimate based on the CWHL / college hockey pool).
       Empirically we set replacement at -X goals per 60 below average,
       where X is estimated from the RAPM distribution.

  3. Goals Above Replacement (GAR)
       GAR = (RAPM - replacement_level) × TOI_min / 60

  4. Goals → Wins conversion
       The Pythagorean expectation for hockey gives:

           win% = GF² / (GF² + GA²)

       At league average (GF ≈ GA ≈ G):
           dW/dGF ≈ 1 / (2G)

       So  goals_per_win ≈ 2 × (league avg goals per team per game)

       We estimate this empirically from the season schedule data.

  5. WAR = GAR / goals_per_win

Components
----------
  oWAR : wins contributed via offense
  dWAR : wins contributed via defense
  WAR  : total = oWAR + dWAR

References
----------
- Macdonald (2012) "Adjusted Plus-Minus for NHL Players"
- Thomas et al. "Wins Above Replacement for NHL Players"
- Evolving-Hockey WAR methodology notes
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .constants import DEFAULT_REPLACEMENT_PCT

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------

# Replacement level percentile: players at or below this RAPM percentile
# are considered "replacement level".  Previously 20.0; aligned to 25.0
# to match box_war and xga_war defaults.
REPLACEMENT_PERCENTILE: float = DEFAULT_REPLACEMENT_PCT  # previously 20.0

# Minimum TOI (minutes) required to be considered a full-season player
# for the replacement-level calculation.
REPLACEMENT_MIN_TOI: float = 50.0


class WARCalculator:
    """
    Convert RAPM output into WAR.

    Parameters
    ----------
    replacement_percentile : float
        RAPM percentile defining the "replacement" player (default 20th).
    replacement_min_toi    : float
        Minimum TOI (min) for a player to be included in the replacement-level
        estimation.  Very low-TOI players are excluded as they may be injuries
        or emergency call-ups rather than true replacement-level players.
    goals_per_win          : float or None
        Override the empirically derived goals-per-win value.  If None (default),
        it is computed from the season schedule data passed to calculate_war().
    """

    def __init__(
        self,
        replacement_percentile: float = REPLACEMENT_PERCENTILE,
        replacement_min_toi:    float = REPLACEMENT_MIN_TOI,
        goals_per_win:          float | None = None,
    ):
        self.replacement_percentile = replacement_percentile
        self.replacement_min_toi    = replacement_min_toi
        self._goals_per_win_override = goals_per_win

        # Fitted values (set after calculate_war)
        self.replacement_level_: float | None = None
        self.goals_per_win_:     float | None = None

    # ------------------------------------------------------------------
    # Main method
    # ------------------------------------------------------------------

    def calculate_war(
        self,
        rapm_df:   pd.DataFrame,
        schedule:  list[dict] | None = None,
    ) -> pd.DataFrame:
        """
        Compute WAR for every player in rapm_df.

        Parameters
        ----------
        rapm_df  : DataFrame from RAPMModel.get_rapm()
                   Required columns: player_id, oRAPM, dRAPM, RAPM, toi_min
        schedule : list of game dicts from PWHLScraper.get_schedule()
                   Used to compute goals_per_win from actual season data.
                   If None, falls back to a league-average estimate.

        Returns
        -------
        DataFrame with columns:
          player_id, toi_min, oRAPM, dRAPM, RAPM,
          oGAA, dGAA, GAA,        ← Goals Above Average (total, not per-60)
          replacement_level,       ← scalar, same for all rows
          goals_per_win,           ← scalar, same for all rows
          oGAR, dGAR, GAR,         ← Goals Above Replacement
          oWAR, dWAR, WAR          ← Wins Above Replacement
        """
        df = rapm_df.copy()

        # 1. Goals per win
        gpw = self._compute_goals_per_win(schedule)
        self.goals_per_win_ = gpw
        log.info("Goals per win: %.3f", gpw)

        # 2. Replacement level
        repl = self._compute_replacement_level(df)
        self.replacement_level_ = repl
        log.info("Replacement level RAPM: %.3f goals/60", repl)

        # 3. Goals Above Average (volume stats — not per-60)
        toi = df["toi_min"]
        df["oGAA"] = (df["oRAPM"] * toi / 60).round(3)
        df["dGAA"] = (df["dRAPM"] * toi / 60).round(3)
        df["GAA"]  = (df["RAPM"]  * toi / 60).round(3)

        # 4. Goals Above Replacement
        #    GAR = (RAPM - replacement_level) × TOI / 60
        #    Split: oGAR uses oRAPM vs. half of replacement,
        #           dGAR uses dRAPM vs. half of replacement.
        #    (Half-split mirrors the Evolving-Hockey convention.)
        half_repl   = repl / 2.0
        df["oGAR"]  = ((df["oRAPM"] - half_repl) * toi / 60).round(3)
        df["dGAR"]  = ((df["dRAPM"] - half_repl) * toi / 60).round(3)
        df["GAR"]   = (df["oGAR"] + df["dGAR"]).round(3)

        # 5. Wins Above Replacement
        df["oWAR"] = (df["oGAR"] / gpw).round(3)
        df["dWAR"] = (df["dGAR"] / gpw).round(3)
        df["WAR"]  = (df["GAR"]  / gpw).round(3)

        # Attach scalar context columns for reference
        df["replacement_level"] = round(repl, 4)
        df["goals_per_win"]     = round(gpw,  4)

        col_order = [
            "player_id", "toi_min",
            "oRAPM", "dRAPM", "RAPM",
            "oGAA",  "dGAA",  "GAA",
            "replacement_level", "goals_per_win",
            "oGAR",  "dGAR",  "GAR",
            "oWAR",  "dWAR",  "WAR",
        ]
        return df[col_order].sort_values("WAR", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Goals-per-win estimation
    # ------------------------------------------------------------------

    def _compute_goals_per_win(self, schedule: list[dict] | None) -> float:
        """
        Estimate how many marginal goals equals one win using the
        Pythagorean hockey formula.

        At league average (GF = GA = G per team per game):
            goals_per_win = 2 × G

        If schedule data is unavailable, fall back to 6.0 (typical NHL value).
        """
        if self._goals_per_win_override is not None:
            return self._goals_per_win_override

        if not schedule:
            log.warning("No schedule data; using default goals_per_win = 6.0")
            return 6.0

        gf_list, ga_list = [], []
        for g in schedule:
            try:
                hg = int(g.get("home_goal_count",    g.get("HomeGoals",    0)) or 0)
                ag = int(g.get("visiting_goal_count", g.get("VisitorGoals", 0)) or 0)
                if g.get("game_status", "").lower() == "final":
                    gf_list.append(hg)
                    ga_list.append(ag)
            except (ValueError, TypeError):
                continue

        if not gf_list:
            log.warning("Could not parse schedule; using default goals_per_win = 6.0")
            return 6.0

        total_goals = sum(gf_list) + sum(ga_list)
        n_games     = len(gf_list)
        avg_goals_per_team_per_game = total_goals / (2 * n_games)

        # Pythagorean: goals_per_win ≈ 2 × avg_goals_per_team_per_game
        gpw = 2.0 * avg_goals_per_team_per_game
        log.info(
            "Avg goals/team/game: %.3f  →  goals_per_win: %.3f",
            avg_goals_per_team_per_game, gpw
        )
        return gpw

    # ------------------------------------------------------------------
    # Replacement level estimation
    # ------------------------------------------------------------------

    def _compute_replacement_level(self, rapm_df: pd.DataFrame) -> float:
        """
        Define the replacement-level RAPM as the Nth percentile of
        qualified players' RAPM values.

        Interpretation: a replacement-level player is as good as the
        bottom N% of rostered players who logged meaningful ice time.
        """
        qualified = rapm_df[rapm_df["toi_min"] >= self.replacement_min_toi]["RAPM"]

        if len(qualified) < 5:
            log.warning(
                "Fewer than 5 qualified players for replacement level (%d). "
                "Using 0.0 as fallback.",
                len(qualified),
            )
            return 0.0

        level = float(np.percentile(qualified, self.replacement_percentile))
        log.info(
            "Replacement level (p%.0f of %d qualified): %.4f goals/60",
            self.replacement_percentile, len(qualified), level
        )
        return level

    # ------------------------------------------------------------------
    # Diagnostics / utilities
    # ------------------------------------------------------------------

    def summary(self, war_df: pd.DataFrame, top_n: int = 20) -> None:
        """Print a formatted WAR leaderboard."""
        print(f"\n{'='*72}")
        print(f"  PWHL WAR Leaderboard (top {top_n})")
        print(f"  Replacement level : {self.replacement_level_:+.3f} goals/60")
        print(f"  Goals per win     : {self.goals_per_win_:.3f}")
        print(f"{'='*72}")
        cols = ["player_id", "toi_min", "oRAPM", "dRAPM", "RAPM",
                "oWAR", "dWAR", "WAR"]
        print(war_df[cols].head(top_n).to_string(index=False))
        print(f"{'='*72}\n")

    @staticmethod
    def season_totals(war_df: pd.DataFrame) -> dict:
        """Return a dict of aggregate season-level WAR stats."""
        return {
            "total_WAR":   war_df["WAR"].sum().round(2),
            "median_WAR":  war_df["WAR"].median().round(3),
            "max_WAR":     war_df["WAR"].max().round(3),
            "min_WAR":     war_df["WAR"].min().round(3),
            "n_players":   len(war_df),
            "goals_per_win": war_df["goals_per_win"].iloc[0] if len(war_df) else None,
            "replacement_level": war_df["replacement_level"].iloc[0] if len(war_df) else None,
        }
