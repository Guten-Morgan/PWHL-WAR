"""
PWHL WAR (Wins Above Replacement)
==================================
xG-based WAR models for the Professional Women's Hockey League.

Active models
-------------
  csv_loader.py  -> download & cache game_data from hockey-statistics.com
  box_war.py     -> individual xG (offense) + residual pm60 (defense)
  xga_war.py     -> individual xG (offense) + Fenwick FA/xGA (defense)
  run_war.py     -> CLI runner for box_war
  run_xga_war.py -> CLI runner for xga_war
  run_combined_war.py -> blend box oWAR + xGA dWAR (BOX_WEIGHT=0.0 intentional)

Shared utilities
----------------
  constants.py   -> single source of truth for all constants (season maps,
                    API key, BLOCKS_FILES, model hyperparameter defaults)
  stats_utils.py -> 4-step OLS -> team-adjust -> league-adjust -> scale pipeline
  io_utils.py    -> load_blocks() and other shared I/O helpers

Dormant RAPM lineage (see RAPM_ARCHIVE.md)
------------------------------------------
  scraper.py     -> HockeyTech API client (fetches raw PBP + schedules)
  processor.py   -> parse PBP into shift stints for RAPM
  rapm.py        -> ridge-regression RAPM (goals above average per 60)
  war.py         -> convert RAPM + TOI to Wins Above Replacement

  The RAPM pipeline is dormant because the PWHL's current scale (6-8 teams,
  ~24 games/season) provides too few unique stints for ridge regression to
  reliably separate teammate effects.  See RAPM_ARCHIVE.md for the rationale
  and conditions under which RAPM could be reactivated.
"""
