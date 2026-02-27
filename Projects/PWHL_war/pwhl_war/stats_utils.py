"""
stats_utils.py
--------------
Shared statistical utilities for the PWHL WAR project.

Contains the 4-step OLS-residual → team-adjust → league-adjust → scale
defensive pipeline used by box_war.py, xga_war.py, tune_fa_weight.py,
and compare_fa_xga.py.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

log = logging.getLogger(__name__)


def compute_defensive_value60(
    df: pd.DataFrame,
    qual_mask: pd.Series,
    raw_col: str,
    off_col: str,
    team_col: str,
    defense_weight: float,
    sign: int = 1,
    team_adjust: bool = True,
) -> pd.DataFrame:
    """
    4-step defensive adjustment pipeline.

    Steps
    -----
    1. OLS residual: remove linear correlation between raw defensive metric
       and individual offensive output (players in offensive zone see fewer
       shots against; high scorers have inflated +/-).

    2. Team-quality adjust: subtract each team's TOI-weighted mean residual
       so the metric reflects individual vs. teammate comparison, not
       team-quality effects.  Skipped when ``team_adjust=False``.

    3. League-mean adjust: subtract the league-wide TOI-weighted mean of
       team-adjusted residuals to produce a zero-centred metric.

    4. Scale: multiply by `defense_weight` (and `sign`) to produce `d_value60`.

    Parameters
    ----------
    df             : player-level season DataFrame (must have `toi_min`)
    qual_mask      : boolean Series indexing qualified players (toi_min >= threshold)
    raw_col        : name of the raw defensive metric column (e.g. "pm60", "FA60")
    off_col        : name of the individual offensive rate column (e.g. "o_xG60", "ixG60")
    team_col       : name of the team column (e.g. "Team", "team")
    defense_weight : scaling factor applied to the adjusted metric
    sign           : +1 for pm60 (higher = better defense), -1 for FA60 (fewer = better)
    team_adjust    : if True (default), subtract each team's TOI-weighted mean residual
                     before league centering (Step 2).  Set False to preserve
                     between-team signal for diagnostic experiments.

    Returns
    -------
    df with `d_value60` column added (and intermediate `_def_resid`, `_def_adj` columns).
    """
    df = df.copy()
    qual_fit = df[qual_mask]

    # Step 1: OLS residual
    reg = LinearRegression().fit(
        qual_fit[[off_col]].values,
        qual_fit[raw_col].values,
    )
    df["_def_resid"] = df[raw_col] - (reg.intercept_ + reg.coef_[0] * df[off_col])
    log.info(
        "%s ~ %s: intercept=%.3f slope=%.3f",
        raw_col, off_col, reg.intercept_, reg.coef_[0],
    )

    # Step 2: Team-quality adjustment (optional — skip to preserve between-team signal)
    if team_adjust:
        team_resid = (
            df[qual_mask]
            .groupby(team_col)["_def_resid"]
            .apply(lambda g: np.average(
                g, weights=df.loc[g.index, "toi_min"].clip(lower=0.1)
            ))
        )
        df["_team_resid"] = df[team_col].map(team_resid).fillna(0)
        df["_def_resid"]  = df["_def_resid"] - df["_team_resid"]
        log.info("Team %s adjustments: %s", raw_col, team_resid.round(3).to_dict())

    # Step 3: League-mean adjust
    league_resid = np.average(
        df.loc[qual_mask, "_def_resid"],
        weights=df.loc[qual_mask, "toi_min"].clip(lower=0.1),
    )
    df["_def_adj"] = df["_def_resid"] - league_resid

    # Step 4: Scale (sign converts FA60 direction: fewer shots = positive)
    df["d_value60"] = sign * df["_def_adj"] * defense_weight

    return df
