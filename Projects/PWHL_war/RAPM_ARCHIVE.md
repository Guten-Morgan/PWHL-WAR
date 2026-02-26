# RAPM Archive

This document explains the purpose, current status, and reactivation conditions
for the Regularized Adjusted Plus/Minus (RAPM) pipeline in the PWHL WAR project.

## What these files implement

| File | Role |
|---|---|
| `pwhl_war/scraper.py` | HockeyTech API client — fetches raw play-by-play, schedules, and roster data |
| `pwhl_war/processor.py` | Converts PBP events into shift stints; builds the RAPM design matrix |
| `pwhl_war/rapm.py` | Ridge-regression RAPM — isolates each player's goals-above-average per 60 while controlling for teammates/opponents |
| `pwhl_war/war.py` | Converts RAPM coefficients + TOI into Wins Above Replacement (GAA → GAR → WAR) |

## Why RAPM is dormant

RAPM uses ridge regression on a player × stint design matrix where each stint is one row and each player column is +1 (home), −1 (away), or 0 (off ice). Ridge regression needs **many independent stints** to separate each player's contribution from the confounding effects of teammates and opponents.

The PWHL currently has:
- **6–8 teams** (vs. 32 in the NHL)
- **~24 regular-season games per team per season**
- **Limited roster depth** — a small player pool means the same players appear together very frequently

At this scale, the ridge penalty dominates and shrinks all coefficients toward zero, making RAPM estimates unreliable and largely uninformative. Year-to-year repeatability tests confirmed that RAPM-derived dWAR is not significantly more stable than random noise at the individual level for the current PWHL sample sizes.

By contrast, the **box WAR** (pm60-based) and **xGA/Fenwick WAR** (FA-based) models are appropriately calibrated for small-league analytics and produce team-level validation r values of 0.7–0.9 vs. goal differential.

## Conditions for reactivation

The RAPM pipeline could be reactivated when one or more of the following conditions are met:

1. **League scale ≥ 12 teams** — more teams → more opponents faced → less collinearity in the design matrix
2. **Multi-year pooled stints** — combining 3+ full seasons (~72+ games per team) would provide sufficient stint counts for ridge regression to work reliably
3. **Shift-level data becomes available** — the PWHL API's `on_ice_home` / `on_ice_away` fields are currently empty for all events; if per-shift on-ice player IDs become published, the processor could build accurate stints rather than relying on TOI-share attribution

## Notes on the current code

- `scraper.py` and `processor.py` are still functional — they can fetch and parse data correctly
- `rapm.py` uses `RidgeCV` with cross-validation; the architecture is sound but the data volume is insufficient
- `war.py` uses the 25th percentile replacement level (aligned to `box_war.py` and `xga_war.py` after Phase 2 refactor; was previously 20th percentile)
- None of these files are imported by the active models (`box_war.py`, `xga_war.py`) — they can be safely ignored without affecting WAR output
