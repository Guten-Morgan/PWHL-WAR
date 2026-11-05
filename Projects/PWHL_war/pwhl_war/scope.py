"""
scope.py
--------
Which games belong to a regular season.

The sportsdataverse schedule master tags every game with `game_type`
("regular" or "playoffs") and a `season` end year (2026 = 2025-26).
"""

from __future__ import annotations

import pandas as pd

_REQUIRED = ("game_id", "season", "game_type")


def regular_season_game_ids(schedule_df: pd.DataFrame, season: str) -> list[int]:
    """Sorted, unique regular-season game ids for a season label like "2025-26"."""
    missing = [c for c in _REQUIRED if c not in schedule_df.columns]
    if missing:
        raise ValueError(f"schedule is missing required column(s): {', '.join(missing)}")

    end_year = int(season.split("-")[0]) + 1
    keep = (schedule_df["season"].astype(int) == end_year) & (schedule_df["game_type"] == "regular")
    return sorted({int(g) for g in schedule_df.loc[keep, "game_id"]})
