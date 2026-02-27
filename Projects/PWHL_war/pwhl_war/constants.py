"""
constants.py
------------
Single source of truth for all shared constants in the PWHL WAR project.

This module has zero imports from pwhl_war to prevent circular imports.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Season mappings
# ---------------------------------------------------------------------------

# CSV season codes (as stored in hockey-statistics.com data) → human label
SEASON_CODES = {
    "20232024": "2023-24",
    "20242025": "2024-25",
    "20252026": "2025-26",
}

# Human season label → API year format used by pwhl.hockey-statistics.com
SEASON_YEARS = {
    "2023-24": "2023/2024",
    "2024-25": "2024/2025",
    "2025-26": "2025/2026",
}

# Human season label → HockeyTech season IDs for standings / schedule API
SEASON_IDS = {
    "2023-24": "1",
    "2024-25": "5",
    "2025-26": "8",
}

# Pythagorean goals-per-win constants (2 × avg_goals_per_team_per_game)
# Derived from the completed regular season for each year.
GPW = {
    "2023-24": 4.569,
    "2024-25": 5.049,
    "2025-26": 4.508,
}

# ---------------------------------------------------------------------------
# Team name → abbreviation
# ---------------------------------------------------------------------------

TEAM_MAP = {
    "Boston Fleet":         "BOS",
    "Minnesota Frost":      "MIN",
    "Montreal Victoire":    "MTL",
    "Montréal Victoire":    "MTL",   # accented form from schedule API
    "New York Sirens":      "NY",
    "Ottawa Charge":        "OTT",
    "Toronto Sceptres":     "TOR",
    "Seattle Torrent":      "SEA",
    "Vancouver Goldeneyes": "VAN",
}

# ---------------------------------------------------------------------------
# API credentials
# ---------------------------------------------------------------------------

# Community-documented public API key — not a secret.
# Visible in cached JSON filenames; centralised here for DRY, not security.
HOCKEYTECH_API_KEY = "446521baf8c38984"

# ---------------------------------------------------------------------------
# Data file paths
# ---------------------------------------------------------------------------

_RAW_DIR = Path(__file__).parent / "data" / "raw"

BLOCKS_FILES = {
    "2023-24": _RAW_DIR / "blocks_2324.csv",
    "2024-25": _RAW_DIR / "blocks_2425.csv",
    "2025-26": _RAW_DIR / "blocks_2526.csv",
}

# ---------------------------------------------------------------------------
# Model hyperparameter defaults
# ---------------------------------------------------------------------------

DEFAULT_MIN_TOI         = 50.0    # minutes; ~5 full games
DEFAULT_REPLACEMENT_PCT = 25.0    # percentile defining replacement level
GPW_FALLBACK            = 6.0     # goals-per-win when schedule data unavailable
DEFAULT_DEFENSE_WEIGHT  = 0.79    # Exp-3 true optimum: normalised FA60, team_adjust=False (r=0.602, p=0.014)
DEFAULT_BLOCK_WEIGHT    = 0.04    # xG value per position+team-adjusted block/60
