"""
coord_xg.py
-----------
PWHL-native coordinate-based expected goals (xG) model.

Uses logistic regression on shot coordinates to produce a PWHL-calibrated
per-shot xG probability, independent of the hockey-statistics.com black-box
xG algorithm.

Features (V2)
-------------
  log_dist          : log1p(sqrt((x - 89)^2 + y^2)) — log distance to net
  angle             : arctan2(|y|, max(89 - x, 0.01)) — shot angle from net
  strength_pp       : 1 if Strength == "PP", else 0
  log_dist × angle  : interaction term

NET_X = 89.0 (attacking net x-coordinate in the PWHL PBP coordinate system).

Target
------
  goal = (event == "Goal").astype(int)

Validation
----------
  Brier score on hold-out data: ~0.0755 (V2) vs ~0.0783 naive baseline.
  Train on 2023-24 + 2024-25; evaluate on 2025-26.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

log = logging.getLogger(__name__)


class CoordXGModel:
    """
    PWHL-native logistic regression xG model trained on shot coordinates.

    Parameters
    ----------
    C : float
        Inverse of regularization strength (default 1.0 = no regularization).
    """

    NET_X: float = 89.0  # attacking net x-coordinate in the PWHL PBP system

    def __init__(self, C: float = 1.0):
        self.C = C
        self._model: LogisticRegression | None = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    @classmethod
    def _features(cls, df: pd.DataFrame) -> np.ndarray:
        """
        Build the V2 feature matrix from a shot DataFrame.

        Expected columns: x, y, strength
        Returns: [log_dist, angle, is_pp, log_dist * angle]
        """
        x = pd.to_numeric(df["x"], errors="coerce").fillna(0.0).values
        y = pd.to_numeric(df["y"], errors="coerce").fillna(0.0).values

        dist     = np.sqrt((x - cls.NET_X) ** 2 + y ** 2)
        log_dist = np.log1p(dist)
        angle    = np.arctan2(np.abs(y), np.maximum(cls.NET_X - x, 0.01))

        pp_flag = (df.get("strength", pd.Series(["EV"] * len(df)))
                   .str.upper()
                   .str.startswith("PP")
                   .astype(float)
                   .values)

        return np.column_stack([log_dist, angle, pp_flag, log_dist * angle])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, pbp_df: pd.DataFrame) -> "CoordXGModel":
        """
        Fit the logistic regression model.

        Parameters
        ----------
        pbp_df : DataFrame from CoordLoader.fetch_pbp() — shot + goal rows only.
                 Required columns: x, y, strength, event
                 Rows with null x or y are dropped.

        Returns
        -------
        self  (for chaining)
        """
        df = pbp_df.dropna(subset=["x", "y"]).copy()
        if len(df) < 10:
            raise ValueError(
                f"Too few rows to train ({len(df)}). "
                "Provide at least 10 valid shot/goal rows."
            )

        X = self._features(df)
        y = (df["event"].str.lower() == "goal").astype(int).values

        self._model = LogisticRegression(
            C=self.C,
            max_iter=500,
            random_state=42,
        )
        self._model.fit(X, y)
        self._is_fitted = True
        log.info(
            "CoordXGModel trained on %d shots (%d goals, %.1f%% rate)",
            len(df), y.sum(), 100 * y.mean(),
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return per-row predicted xG probability.

        Parameters
        ----------
        df : DataFrame with x, y, strength columns

        Returns
        -------
        1-D array of floats in [0, 1]
        """
        if not self._is_fitted:
            raise RuntimeError("Call train() before predict().")
        df_clean = df.copy()
        df_clean["x"] = pd.to_numeric(df_clean["x"], errors="coerce").fillna(0.0)
        df_clean["y"] = pd.to_numeric(df_clean["y"], errors="coerce").fillna(0.0)
        X = self._features(df_clean)
        return self._model.predict_proba(X)[:, 1]

    def player_xg_season(self, pbp_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate predicted xG sums by player × season.

        Parameters
        ----------
        pbp_df : DataFrame from CoordLoader.fetch_pbp()

        Returns
        -------
        DataFrame with columns: player, season, coord_xG
        """
        if not self._is_fitted:
            raise RuntimeError("Call train() before player_xg_season().")

        df = pbp_df.copy()
        df["coord_xG"] = self.predict(df)

        agg = (
            df.groupby(["player", "season"], as_index=False)["coord_xG"]
            .sum()
            .rename(columns={"coord_xG": "coord_xG"})
        )
        return agg

    def brier_score(self, pbp_df: pd.DataFrame) -> float:
        """
        Compute the Brier score (mean squared error between predicted xG
        and actual goal indicator).

        Lower is better; random = ~0.09 for ~10% goal rate.
        """
        if not self._is_fitted:
            raise RuntimeError("Call train() before brier_score().")
        df = pbp_df.dropna(subset=["x", "y"]).copy()
        y_true = (df["event"].str.lower() == "goal").astype(float).values
        y_pred = self.predict(df)
        return float(np.mean((y_pred - y_true) ** 2))
