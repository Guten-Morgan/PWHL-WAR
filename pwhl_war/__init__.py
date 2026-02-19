"""
PWHL WAR (Wins Above Replacement)
==================================
RAPM-based WAR model for the Professional Women's Hockey League.

Pipeline:
  scraper.py   -> fetch raw data from HockeyTech API
  processor.py -> parse play-by-play into shift stints
  rapm.py      -> ridge-regression RAPM (goals above average per 60)
  war.py       -> convert RAPM + TOI to Wins Above Replacement
"""
