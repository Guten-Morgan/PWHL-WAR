"""
box_war.py
----------
xG-based Wins Above Replacement for PWHL skaters.

Data source: game_data table from hockey-statistics.com, which provides
per-player per-game stats including individual expected goals (ixG) and
plus/minus.

Data limitation
---------------
The EV_xGA / PP_xGA / SH_xGA columns in the source data are only populated
for goalies, not skaters.  The defensive component therefore uses a
residual plus/minus approach.

Method
------
1. Aggregate game_data to season-level per player.

2. Offensive value: individual expected goals per 60 min
     o_xG60 = (EV_ixG + PP_ixG + SH_ixG) / toi_min × 60

3. Defensive value: residual plus/minus per 60 min

     Raw plus/minus conflates offense and defense — high scorers are on
     the ice for more goals, so they accumulate high +/- regardless of
     defensive ability.  We remove the linear relationship with o_xG60
     via OLS regression on qualified players:

       pm60      = plusMinus / toi_min × 60
       β         = OLS slope of pm60 ~ o_xG60 (qualified players)
       pm60_resid = pm60 − (α + β × o_xG60)

     The residual captures "on-ice goal differential that can't be
     explained by the player's own shot generation" — a proxy for
     two-way/defensive value.

     League-mean adjustment:
       league_resid = TOI-weighted average of pm60_resid
       d_adj60      = pm60_resid − league_resid
       d_value60    = d_adj60 × defense_weight

     defense_weight = 0.1: empirically optimised — team WAR vs GD Spearman
     rs peaks at 0.9429 (p=0.005) for weights in [0.02, 0.14]; drops to
     0.8286 above 0.15.  Residual +/- is noisier than xG and still
     partially reflects team quality, so a light touch is appropriate.

4. Total value per 60:
     value60 = o_xG60 + d_value60

5. League average value60 (weighted by TOI):
     avg_value60 = Σ(value60 × toi) / Σ(toi)

6. Goals Above Average:
     GAA = (value60 − avg_value60) × toi_min / 60

7. Separate replacement levels (p-th percentile of each component):
     o_repl = np.percentile(o_xG60,   replacement_pct)
     d_repl = np.percentile(d_value60, replacement_pct)

8. Goals Above Replacement:
     oGAR = (o_xG60   − o_repl) × toi_min / 60
     dGAR = (d_value60 − d_repl) × toi_min / 60
     GAR  = oGAR + dGAR

9. Wins Above Replacement:
     oWAR  = oGAR / goals_per_win
     dWAR  = dGAR / goals_per_win         (signed — positive = above repl)
     WAR   = oWAR + dWAR                  (signed: good defense adds, bad subtracts)
     war60 = WAR / toi_min × 60           (on-ice quality, ice-time-independent)
     goals_per_win = 2 × avg_goals_per_team_per_game  (Pythagorean)

Limitations
-----------
- Defensive metric is a residual proxy, not RAPM.  It captures variance
  in +/- unexplained by individual shot generation, but team quality
  effects remain partially present.
- Intended as a solid starting point; RAPM would require shift-level data
  with populated on-ice player IDs (currently empty in the PWHL data).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

log = logging.getLogger(__name__)

DEFAULT_MIN_TOI      = 50.0    # minutes; ~5 full games
DEFAULT_REPLACEMENT  = 25.0    # percentile


class XGWar:
    """
    xG-based WAR from the PWHL game_data table.

    Parameters
    ----------
    min_toi_min        : Minimum TOI (minutes) to qualify (default 50).
    replacement_pct    : Percentile of value60 that defines replacement level.
    goals_per_win      : Override the Pythagorean estimate if desired.
    defense_weight     : Weight applied to the defensive component [0, 1].
                         Default 0.1 (empirically optimised vs team GD).
                         0.0 = offense only.
    """

    def __init__(
        self,
        min_toi_min:     float = DEFAULT_MIN_TOI,
        replacement_pct: float = DEFAULT_REPLACEMENT,
        goals_per_win:   float | None = None,
        defense_weight:  float = 0.1,
    ):
        self.min_toi_min     = min_toi_min
        self.replacement_pct = replacement_pct
        self._gpw_override   = goals_per_win
        self.defense_weight  = defense_weight

        self.results_:             pd.DataFrame | None = None
        self.o_replacement_val60_: float | None        = None
        self.d_replacement_val60_: float | None        = None
        self.league_avg_val60_:    float | None        = None
        self.goals_per_win_:       float | None        = None

    # ------------------------------------------------------------------
    # Main fit
    # ------------------------------------------------------------------

    def fit(
        self,
        game_data_df: pd.DataFrame,
        schedule_df:  pd.DataFrame | None = None,
    ) -> "XGWar":
        """
        Parameters
        ----------
        game_data_df : from PWHLCsvLoader.get_game_data()
        schedule_df  : from PWHLCsvLoader.get_schedule() — for goals_per_win
        """
        df = self._aggregate(game_data_df)

        # Goals per win
        gpw = self._compute_gpw(schedule_df)
        self.goals_per_win_ = gpw

        # --- Offensive xG per 60 ---
        df["total_ixG"] = df["EV_ixG"] + df["PP_ixG"] + df["SH_ixG"]
        df["o_xG60"]    = df["total_ixG"] / df["toi_min"].clip(lower=0.1) * 60

        # --- Defensive: residual plus/minus per 60 ---
        #
        # EV_xGA / PP_xGA / SH_xGA are only populated for goalies in this
        # dataset; they are zero for all skaters.  We use plus/minus as a
        # proxy, but first remove the linear relationship with o_xG60 via
        # OLS regression on qualified players.
        #
        # This prevents high scorers from being rewarded twice: once in
        # o_xG60 (shot generation) and again in d_value60 (being on ice
        # when their shots become goals).  The residual captures on-ice
        # goal differential unexplained by individual offensive output.
        df["pm60"] = df["plusMinus"] / df["toi_min"].clip(lower=0.1) * 60

        qual_mask = df["toi_min"] >= self.min_toi_min
        qual_fit  = df[qual_mask]

        reg = LinearRegression().fit(
            qual_fit[["o_xG60"]].values,
            qual_fit["pm60"].values,
        )
        df["pm60_resid"] = df["pm60"] - (
            reg.intercept_ + reg.coef_[0] * df["o_xG60"]
        )
        log.info(
            "pm60 ~ o_xG60: intercept=%.3f slope=%.3f",
            reg.intercept_, reg.coef_[0],
        )

        # League-mean-adjust the residuals
        league_resid = np.average(
            df.loc[qual_mask, "pm60_resid"],
            weights=df.loc[qual_mask, "toi_min"].clip(lower=0.1),
        )
        df["d_adj_pm60"] = df["pm60_resid"] - league_resid
        df["d_value60"]  = df["d_adj_pm60"] * self.defense_weight

        # --- Combined value ---
        df["value60"] = df["o_xG60"] + df["d_value60"]

        # --- League average (TOI-weighted, qualified players only) ---
        qual = df[df["toi_min"] >= self.min_toi_min]
        toi_sum = qual["toi_min"].sum()
        if toi_sum == 0:
            raise ValueError("No qualified players. Lower min_toi_min.")

        avg_val60 = (qual["value60"] * qual["toi_min"]).sum() / toi_sum
        self.league_avg_val60_ = avg_val60
        log.info("League average value60: %.4f", avg_val60)

        # Separate offense/defense league averages
        avg_o = (qual["o_xG60"]  * qual["toi_min"]).sum() / toi_sum
        avg_d = (qual["d_value60"] * qual["toi_min"]).sum() / toi_sum

        # --- Separate replacement levels (following Evolving-Hockey convention) ---
        #
        # Offensive and defensive distributions have completely different shapes:
        #   o_xG60    : positive-valued, roughly bell-shaped (~0.3–1.5 range)
        #   d_value60 : league-adjusted +/- deviation, centred near 0
        #
        # Each component gets its own p-th percentile cutoff.
        o_repl = float(np.percentile(qual["o_xG60"],   self.replacement_pct))
        d_repl = float(np.percentile(qual["d_value60"], self.replacement_pct))
        self.o_replacement_val60_ = o_repl
        self.d_replacement_val60_ = d_repl
        log.info("Offensive replacement (p%.0f): %.4f xG/60", self.replacement_pct, o_repl)
        log.info("Defensive replacement (p%.0f): %.4f d_val/60", self.replacement_pct, d_repl)

        # --- Goals Above Average ---
        df["oGAA"] = ((df["o_xG60"]   - avg_o) * df["toi_min"] / 60).round(3)
        df["dGAA"] = ((df["d_value60"] - avg_d) * df["toi_min"] / 60).round(3)
        df["GAA"]  = (df["oGAA"] + df["dGAA"]).round(3)

        # --- Goals Above Replacement ---
        df["oGAR"] = ((df["o_xG60"]   - o_repl) * df["toi_min"] / 60).round(3)
        df["dGAR"] = ((df["d_value60"] - d_repl) * df["toi_min"] / 60).round(3)
        df["GAR"]  = (df["oGAR"] + df["dGAR"]).round(3)

        # --- WAR ---
        # dWAR is kept as a signed value so players can see their defensive
        # rating relative to replacement.  Total WAR uses |dWAR| so that
        # defensive contribution is always additive — the magnitude of a
        # player's defensive impact (good or bad) adds to their total value.
        df["oWAR"]  = (df["oGAR"] / gpw).round(3)
        df["dWAR"]  = (df["dGAR"] / gpw).round(3)
        df["WAR"]   = (df["oWAR"] + df["dWAR"]).round(3)
        # WAR per 60 minutes — quality per ice time, independent of role depth
        df["war60"] = ((df["WAR"] / df["toi_min"].clip(lower=0.1)) * 60).round(3)

        # Attach context scalars
        df["o_replacement_val60"] = round(o_repl, 4)
        df["d_replacement_val60"] = round(d_repl, 4)
        df["goals_per_win"]       = round(gpw, 4)
        df["league_avg_val60"]    = round(avg_val60, 4)

        self.results_ = df.sort_values("WAR", ascending=False).reset_index(drop=True)
        return self

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def get_war(self, min_toi: float | None = None) -> pd.DataFrame:
        if self.results_ is None:
            raise RuntimeError("Call fit() first.")

        df  = self.results_.copy()
        cut = min_toi if min_toi is not None else self.min_toi_min
        df  = df[df["toi_min"] >= cut].reset_index(drop=True)

        cols = [c for c in [
            "Name", "PlayerID", "Team", "position", "GP", "toi_min",
            # Traditional stats
            "G", "A1", "A2", "plusMinus", "PIM",
            # xG + defensive proxy metrics
            "total_ixG", "pm60", "pm60_resid", "d_adj_pm60",
            "o_xG60", "d_value60", "value60",
            # WAR components
            "oGAA", "dGAA", "GAA",
            "oGAR", "dGAR", "GAR",
            "oWAR", "dWAR", "WAR", "war60",
            "o_replacement_val60", "d_replacement_val60", "goals_per_win",
        ] if c in df.columns]
        return df[cols]

    def summary(self, top_n: int = 25) -> None:
        df = self.get_war()
        name_col = "Name" if "Name" in df.columns else "PlayerID"
        print(f"\n{'='*72}")
        print(f"  PWHL xG-WAR Leaderboard  (top {top_n})")
        print(f"  Method              : xG offense + residual +/- defense")
        print(f"  Off. replacement    : {self.o_replacement_val60_:+.3f} xG/60 (p{self.replacement_pct:.0f})")
        print(f"  Def. replacement    : {self.d_replacement_val60_:+.3f} pm_adj/60 (p{self.replacement_pct:.0f})")
        print(f"  League avg val60    : {self.league_avg_val60_:+.3f}")
        print(f"  Goals per win       : {self.goals_per_win_:.3f}")
        print(f"{'='*72}")
        show = [c for c in [name_col, "Team", "GP", "toi_min", "G",
                             "total_ixG", "o_xG60", "d_value60",
                             "oWAR", "dWAR", "WAR"]
                if c in df.columns]
        print(df[show].head(top_n).to_string(index=False))
        print(f"{'='*72}\n")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Roll up per-game rows to one row per player (season total).
        """
        sum_cols = [
            "TOI",
            "EV_G", "EV_A1", "EV_A2", "EV_Shots", "EV_ixG",
            "PP_G", "PP_A1", "PP_A2", "PP_Shots", "PP_ixG",
            "SH_G", "SH_A1", "SH_A2", "SH_Shots", "SH_ixG",
            "EN_G",
            "PIM", "plusMinus", "hits",
        ]
        # Only keep columns that actually exist
        sum_cols  = [c for c in sum_cols  if c in df.columns]
        group_keys = [c for c in ["PlayerID", "Name", "Team", "position"]
                      if c in df.columns]

        agg = df.groupby(group_keys, as_index=False)[sum_cols].sum()

        # Games played = count of non-zero TOI rows per player
        gp = (
            df[df["TOI"] > 0]
              .groupby(group_keys, as_index=False)
              .size()
              .rename(columns={"size": "GP"})
        )
        agg = agg.merge(gp, on=group_keys, how="left")
        agg["GP"] = agg["GP"].fillna(0).astype(int)

        # Rename TOI to toi_min (already in minutes in this dataset)
        agg = agg.rename(columns={"TOI": "toi_min"})

        # Convenience combined stats
        agg["G"]  = agg.get("EV_G", 0) + agg.get("PP_G", 0) + agg.get("SH_G", 0) + agg.get("EN_G", 0)
        agg["A1"] = agg.get("EV_A1", 0) + agg.get("PP_A1", 0) + agg.get("SH_A1", 0)
        agg["A2"] = agg.get("EV_A2", 0) + agg.get("PP_A2", 0) + agg.get("SH_A2", 0)

        log.info("Aggregated %d player-game rows → %d players", len(df), len(agg))
        return agg

    def _compute_gpw(self, schedule_df: pd.DataFrame | None) -> float:
        if self._gpw_override is not None:
            return self._gpw_override

        if schedule_df is None or schedule_df.empty:
            log.warning("No schedule — using goals_per_win = 6.0")
            return 6.0

        try:
            gf = pd.to_numeric(schedule_df["GF"], errors="coerce").dropna()
            ga = pd.to_numeric(schedule_df["GA"], errors="coerce").dropna()
            total   = gf.sum() + ga.sum()
            n_games = len(gf)   # each row is one team's perspective
            avg_per_team = total / (2 * n_games)
            gpw = 2.0 * avg_per_team
            log.info("Avg goals/team/game=%.3f → goals_per_win=%.3f", avg_per_team, gpw)
            return gpw
        except Exception as exc:
            log.warning("goals_per_win failed (%s) — using 6.0", exc)
            return 6.0
