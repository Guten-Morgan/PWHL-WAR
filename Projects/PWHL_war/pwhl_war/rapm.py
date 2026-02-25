"""
rapm.py
-------
Regularized Adjusted Plus/Minus (RAPM) for PWHL skaters.

RAPM uses ridge regression to isolate each player's individual
contribution to goal differential while controlling for the quality
of their teammates and opponents.

Model
-----
  y_i  = Σ_j X_ij β_j  +  ε_i

where
  y_i   = home net goals per 60 min for stint i
  X_ij  = +1 if player j is a home skater in stint i
           -1 if player j is an away skater in stint i
            0 otherwise
  β_j   = RAPM coefficient for player j (goals above average per 60)
  ε_i   = error term

The regression is weighted by stint duration (longer stints carry
more information) and regularised with an L2 penalty (ridge) that
shrinks noisy estimates toward zero.

Two components are estimated separately:
  Offensive RAPM (oRAPM): contribution to goals FOR
  Defensive RAPM (dRAPM): contribution to goals AGAINST
  Total RAPM             : oRAPM + dRAPM

References
----------
- Macdonald, B. (2011). "A Regression-Based Adjusted Plus-Minus Statistic
  for NHL Players." Journal of Quantitative Analysis in Sports.
- Evolving-Hockey RAPM methodology: evolving-hockey.com
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from .processor import PWHLProcessor

log = logging.getLogger(__name__)


class RAPMModel:
    """
    Fit offensive and defensive RAPM for all PWHL skaters.

    Parameters
    ----------
    alphas : array-like
        Ridge regularisation strengths to cross-validate over.
        Larger alpha → stronger shrinkage → values closer to zero.
        Defaults cover a wide range; the CV will select the best.
    min_toi_min : float
        Minimum total ice-time (minutes) for a player to be included
        in the regression.  Players below this threshold are excluded
        as insufficient sample.
    strength_filter : list[str] or None
        If provided, only stints with these strength codes are used.
        e.g. ['EV'] for even-strength only.  None means all strengths.
    """

    def __init__(
        self,
        alphas: list[float] | None = None,
        min_toi_min: float = 50.0,
        strength_filter: list[str] | None = None,
    ):
        self.alphas          = alphas or [0.01, 0.1, 1, 10, 100, 1000, 10000]
        self.min_toi_min     = min_toi_min
        self.strength_filter = strength_filter

        # Fitted attributes
        self.player_list_: list[str]           = []
        self.rapm_df_:     pd.DataFrame | None = None
        self._offense_model: RidgeCV | None    = None
        self._defense_model: RidgeCV | None    = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, stints_df: pd.DataFrame, toi_df: pd.DataFrame) -> "RAPMModel":
        """
        Fit the RAPM model.

        Parameters
        ----------
        stints_df : DataFrame from PWHLProcessor.stints_to_dataframe()
        toi_df    : DataFrame from PWHLProcessor.get_player_toi()
                    Used to filter out low-TOI players.

        Returns
        -------
        self
        """
        # --- Filter by strength ---
        df = stints_df.copy()
        if self.strength_filter:
            df = df[df["strength"].isin(self.strength_filter)]
        if df.empty:
            raise ValueError("No stints remain after strength filtering.")

        # --- Filter players by minimum TOI ---
        qualified = set(
            toi_df.loc[toi_df["toi_min"] >= self.min_toi_min, "player_id"]
        )
        # Only keep stints where at least some qualified players appear
        # (we still need all players for a valid design matrix, but we'll
        #  only report results for qualified players)
        self.player_list_ = sorted(qualified)
        log.info("Qualified players (≥ %.0f min TOI): %d", self.min_toi_min, len(self.player_list_))

        if len(self.player_list_) < 2:
            raise ValueError(
                f"Only {len(self.player_list_)} qualified players; need at least 2. "
                "Try lowering min_toi_min."
            )

        # --- Build design matrix ---
        X, y_net, weights = PWHLProcessor.build_design_matrix(df, self.player_list_)
        log.info("Design matrix shape: %s", X.shape)

        # Remove zero-weight rows
        mask = weights > 0
        X, y_net, weights = X[mask], y_net[mask], weights[mask]

        # Offensive target: home goals-for per 60
        y_off = np.array([
            (row["home_goals"] / (row["duration_secs"] / 60)) * 60
            for _, row in df[mask.tolist() + [False] * (len(df) - mask.sum())].iterrows()
        ], dtype=np.float32)

        # Re-derive offense and defense targets correctly from filtered df
        df_filtered = df[mask].reset_index(drop=True)
        dur_min     = df_filtered["duration_secs"] / 60.0

        y_off = (df_filtered["home_goals"]  / dur_min * 60).values.astype(np.float32)
        y_def = (df_filtered["away_goals"]  / dur_min * 60).values.astype(np.float32)

        # For defensive RAPM we flip the sign so that "good defence → positive dRAPM"
        # The design matrix is the same; we just predict away goals from home perspective
        # (away goals from home player columns = home defence allowed)
        y_def_home_perspective = -y_def  # negative = fewer goals allowed = better

        # --- Fit ridge regression with cross-validation ---
        log.info("Fitting offensive RAPM...")
        self._offense_model = RidgeCV(alphas=self.alphas, fit_intercept=False)
        self._offense_model.fit(X, y_off, sample_weight=weights)
        log.info("  Best alpha (offense): %.4g", self._offense_model.alpha_)

        log.info("Fitting defensive RAPM...")
        self._defense_model = RidgeCV(alphas=self.alphas, fit_intercept=False)
        self._defense_model.fit(X, y_def_home_perspective, sample_weight=weights)
        log.info("  Best alpha (defense): %.4g", self._defense_model.alpha_)

        # --- Assemble results ---
        self.rapm_df_ = self._build_results(toi_df)
        log.info("RAPM fitting complete.")
        return self

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def get_rapm(self) -> pd.DataFrame:
        """
        Return the RAPM results DataFrame.

        Columns
        -------
        player_id  : str
        toi_min    : float  — total qualifying ice time in minutes
        oRAPM      : float  — offensive RAPM (goals-for above average per 60)
        dRAPM      : float  — defensive RAPM (goals-against prevented per 60)
        RAPM       : float  — total RAPM = oRAPM + dRAPM
        """
        if self.rapm_df_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self.rapm_df_.copy()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_results(self, toi_df: pd.DataFrame) -> pd.DataFrame:
        o_coefs = dict(zip(self.player_list_, self._offense_model.coef_))
        d_coefs = dict(zip(self.player_list_, self._defense_model.coef_))

        rows = []
        for pid in self.player_list_:
            o = o_coefs.get(pid, 0.0)
            d = d_coefs.get(pid, 0.0)
            rows.append({
                "player_id": pid,
                "oRAPM":     round(float(o), 4),
                "dRAPM":     round(float(d), 4),
                "RAPM":      round(float(o + d), 4),
            })

        rapm_df = pd.DataFrame(rows)
        rapm_df = rapm_df.merge(
            toi_df[["player_id", "toi_min"]],
            on="player_id",
            how="left",
        )
        rapm_df["toi_min"] = rapm_df["toi_min"].fillna(0).round(1)
        return rapm_df.sort_values("RAPM", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def r_squared(
        self,
        stints_df: pd.DataFrame,
        target: str = "net",
    ) -> float:
        """
        Compute weighted R² of the fitted model on the training data.
        target: 'net' (total RAPM), 'offense', or 'defense'
        """
        if self.rapm_df_ is None:
            raise RuntimeError("Model not fitted.")

        df = stints_df.copy()
        if self.strength_filter:
            df = df[df["strength"].isin(self.strength_filter)]

        X, y_net, weights = PWHLProcessor.build_design_matrix(df, self.player_list_)
        mask = weights > 0
        X, y_net, weights = X[mask], y_net[mask], weights[mask]

        if target == "net":
            coefs = self._offense_model.coef_ + self._defense_model.coef_
            y     = y_net
        elif target == "offense":
            coefs = self._offense_model.coef_
            df_f  = df[mask].reset_index(drop=True)
            y     = (df_f["home_goals"] / (df_f["duration_secs"] / 60) * 60).values
        else:
            coefs = self._defense_model.coef_
            df_f  = df[mask].reset_index(drop=True)
            y     = -(df_f["away_goals"] / (df_f["duration_secs"] / 60) * 60).values

        y_pred    = X @ coefs
        ss_res    = np.average((y - y_pred) ** 2, weights=weights)
        ss_tot    = np.average((y - np.average(y, weights=weights)) ** 2, weights=weights)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def summary(self) -> None:
        """Print a quick summary of the RAPM results to stdout."""
        df = self.get_rapm()
        print(f"\n{'='*60}")
        print(f"  PWHL RAPM Model Summary")
        print(f"  Players:  {len(df)}")
        if self._offense_model:
            print(f"  α (off):  {self._offense_model.alpha_:.4g}")
        if self._defense_model:
            print(f"  α (def):  {self._defense_model.alpha_:.4g}")
        print(f"{'='*60}")
        print(df.head(20).to_string(index=False))
        print(f"{'='*60}\n")
