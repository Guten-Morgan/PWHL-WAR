"""
io_utils.py
-----------
Shared I/O utilities for the PWHL WAR project.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .constants import BLOCKS_FILES

log = logging.getLogger(__name__)


def load_blocks(season: str | None) -> pd.DataFrame | None:
    """
    Load blocks CSV for the given season.

    Returns a DataFrame with columns [PlayerID, blocks], or None if unavailable.
    Raw observed counts are used as-is; the WAR model normalises by TOI.

    Parameters
    ----------
    season : e.g. '2024-25', or None (combined-season mode — not yet supported)
    """
    if season is None:
        return None   # combined-season mode not yet supported
    path = BLOCKS_FILES.get(season)
    if path is None or not path.exists():
        log.warning("No blocks file for season %s — skipping block component.", season)
        return None

    blk = pd.read_csv(path)
    blk["PlayerID"] = blk["PlayerID"].astype(int)
    return blk[["PlayerID", "blocks"]]
