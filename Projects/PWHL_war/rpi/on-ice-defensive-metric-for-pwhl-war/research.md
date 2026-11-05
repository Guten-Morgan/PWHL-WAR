# On-Ice Defensive Metric for PWHL WAR — Research

Date: 2026-11-05 · Full EDA: [eda_report.md](eda_report.md)

## Problem Context

**Original framing:** replace the plus/minus-residual dWAR with an on-ice xGA / RAPM metric built from the sportsdataverse `fastRhockey-pwhl-data` files (`on_ice_home`/`on_ice_away`, shifts, strength state).

**Corrected framing (from the code analysis):** production dWAR is *not* plus/minus any more. `XGWar.fit` uses an xGA path: the team's even-strength xGA in each game is split among skaters **by share of ice time** (`xga_war.py:586`), then residualised on `o_xG60`, `is_D` and optionally `ozs_pct`, centred, and scaled by `defense_weight = -0.10` (`constants.py:87`, `stats_utils.py:119`). Plus/minus (pm60) is only the fallback when `fa_df` is missing. The real weakness is the **TOI-share attribution**: every skater on a team gets the same xGA rate per minute in a game regardless of who was actually on the ice. Real on-ice data replaces that guess.

**Desired behaviour:** per-player xGA based on who was actually on the ice, delivered in the existing `fa_df` shape (`player_id`, `xGA`) so the downstream math is unchanged, and shown to be at least as repeatable as the current metric.

**Constraints:**
- The PWHL is small: 8 teams, 30 regular-season games each. The earlier RAPM attempt was archived for this reason.
- `CLAUDE.md` requires staying on the linear model.
- The published page is rebuilt daily from this repo, so any new input must be reproducible in CI.
- Positions must stay F/D only.

## Affected Files

```
pwhl_war/box_war.py:252-290        # fa_df path: xGA60, normalise, covariates, compute_defensive_value60
pwhl_war/box_war.py:257            # fa_map = fa_df.set_index("player_id")["xGA"] (last duplicate wins)
pwhl_war/stats_utils.py:22-121     # compute_defensive_value60 (OLS residual, team/league adjust, weight)
pwhl_war/xga_war.py:380-406        # XGAWar.build_fa_season -> [player_id, xGA, toi_min]
pwhl_war/xga_war.py:548-586        # TOI-share attribution of team EV xGA (the thing to replace)
pwhl_war/xga_war.py:122-134        # get_game_ids: season + "Final" only, no regular/playoff filter
pwhl_war/coord_loader.py:133       # seasonState="Regular Season" for shots
pwhl_war/constants.py:87           # DEFAULT_DEFENSE_WEIGHT = -0.10
regen_2526.py:99-139               # game_data_df from all Final games, SeasonStage hard-coded "Regular"
regen_2526.py:40                   # DEFENSE_WEIGHT = -0.10
export_js.py                       # reads war_*.csv -> pwhl-data.js
Archive/RAPM_ARCHIVE.md            # why RAPM was archived + reactivation conditions
Archive/pwhl_war/processor.py:293-343  # archived stint design matrix
Archive/pwhl_war/rapm.py:71-162    # archived RidgeCV RAPM
Archive/tests/validation/tune_fa_weight.py:414-439   # team Spearman vs goal differential
Archive/tests/validation/permutation_tests.py:71-111 # permutation test + year-to-year stability
tests/test_war_formula.py:56-102   # formula-identity tests to extend
```

- A new metric can enter as a drop-in `fa_df`. `XGWar` only reads `player_id` and `xGA`; `toi_min` is unused.
- Players missing from `fa_df` get `xGA = 0`, which reads as perfect defence.
- Traded players appear twice in `fa_df` (keyed `pid|team`) and only the last row survives.

## Code Examples

Current attribution, the step an on-ice source would replace (`xga_war.py:586`):

```python
p["xGA"]     += team_xGA * (toi / team_total_toi)
```

Current scaling, so the sign is set only by the weight (`stats_utils.py:119`):

```python
df["d_value60"] = df["_def_adj"] * defense_weight
```

Existing optional-input pattern to copy for a new on-ice input (`box_war.py:161-170` signature; `extract_ozs.py` for the precompute-to-CSV pattern):

```python
fit(game_data_df, schedule_df=None, pbp_df=None, fa_df=None, ..., blocks_df=None, ozs_df=None)
```

Archived stint design matrix (`Archive/pwhl_war/processor.py:293-343`): home skaters +1, away skaters −1, weight = stint minutes. For xGA the response would be xG against per 60 per stint, with separate offence and defence columns.

## External Research Findings

- **RAPM method.** Macdonald (2012, arXiv 1201.0317) used ridge regression because of teammate collinearity. Shot-based targets were more repeatable year to year than goals. Evolving Hockey and HockeyStats run a stint-weighted ridge on xGF/60, with score state and strength state as covariates (evolving-hockey.com glossary; hockeystats.com/methodology/rapm).
- **Defence is the hard half.** HockeyStats says single-season RAPM is unstable and adds priors from the previous year (defence prior coefficient about 0.28 versus 0.45 for offence). Evolving Hockey's WAR builds defence from 11 years of pooled RAPM, then predicts it from box-score features.
- **No published reliability figures** for xGA RAPM at PWHL scale or at any small sample. Whether it beats the current metric is untested in public sources.
- **Goalies.** Evolving Hockey leaves goalies out of the xGA regression, because xG against does not depend on saves. In the PWHL data, a single goalie covers 54–94% of a team's shots, so goalie and team effects are only partly separable.
- **Tooling.** Sparse `Ridge` with `sample_weight`, and `GridSearchCV` with `GroupKFold` by `game_id` for lambda. `RidgeCV` leave-one-out does not suit sparse input. These tooling notes are from background knowledge, not verified this session.
- **Data source.** `sportsdataverse/fastRhockey-pwhl-data` (MIT, daily updates in season, shifts and play-by-play for 2023-24 to 2025-26). No public PWHL RAPM or WAR was found.

## EDA Findings

Source: [eda_report.md](eda_report.md), 2025-26 files. I independently confirmed the game-id split (regular season ids 210–329, playoffs 338–351) and that `war_2526.csv` has a player with 39 games, which is more than the 30-game regular season.

| Severity | Finding |
|---|---|
| **Critical** | The shipped `on_ice_home`/`on_ice_away` lists are built with a rule that drops the players on the ice at a goal. Only 46% of scoring-team and 37% of conceding-team players (per the goal rows' own plus/minus columns) are in the list. An alternate rule recovers 89%. On-ice sets must be rebuilt from shifts and validated against those columns. |
| **Critical** | Goals appear twice: as a `shot` row with `goal=True` (601) and as a `goal` row (602). Summing both inflates counts by about 8%. |
| **Critical** | No xG column. Available features are distance, angle, coordinates, shot type and `empty_net`. `shot_quality` leaks the outcome and must not be used. |
| **Critical** | The model's own inputs are inconsistent. `war_2526.csv` includes all 133 games (playoffs); `game_data_2526.csv` has only 61 games with zero-filled xGA columns; 10 player ids are duplicated (traded players). |
| Moderate | Overtime lists match shifts only 29% of the time; 421 shot/block rows have impossible strength states; the defending goalie is missing from 9.5% of shot lists. |
| Moderate | 25 teammate pairs share more than 70% of ice time (for example Jenner–Leslie 90%). The stint design matrix is rank-deficient (371 of 372 columns). |
| Estimability | 155 skaters have 100+ five-on-five minutes. At the median (302 min) a skater sees about 138 shots against but only about 11 goals against, so goals-only RAPM is far too thin and xGA is required. |
| Alignment | Player ids agree across play-by-play, shifts, box score and the model files. Box TOI matches model TOI within 0.05 min for 2,156 of 2,306 player-games. |

## FAR Scale Output

### Factual Score: 4
Code claims carry file:line references and the key ones were re-checked (line 586, 119, 87, 257, game-id ranges, gp maximum). The EDA numbers come from a subagent that ran the data. The web findings are partly background knowledge: Macdonald's lambda selection and goalie treatment, and the tooling notes, were not confirmed from primary text.

### Actionable Score: 4
There is a clear insertion point (`fa_df`), a rebuild recipe for on-ice sets, reusable validation scripts and test patterns, and a short list of data fixes.

### Relevant Score: 5
It addresses the core problem and corrects its framing.

### Mean: 4.33
### Result: PASS (mean ≥ 4.0)

Gaps that should not block planning: no reliability evidence for small-sample xGA RAPM, and an untested xG model for the PWHL data.

## Testing Strategy

- **Unit:** the on-ice reconstruction, on a synthetic game with hand-built shifts (pattern: `test_coord_integration.py` mocks). Also the goal de-duplication, and `fa_df` shape and sign.
- **Validation of the rebuild:** compare reconstructed sets against `plus_player_*`/`minus_player_*` on goal rows, and against the shipped lists on shots. Target at least the 89% recovered by the alternate rule.
- **Metric validation:**
  - Split-half by alternating games, and 2023-24 vs 2024-25 vs 2025-26 repeatability, using the permutation-test scripts.
  - Compare against current dWAR on the same splits.
- **Important caveat:** the existing team-level check (Spearman against goal differential, r = 0.718) cannot test individual attribution. Team totals are the same under TOI-share and under on-ice attribution, so only player-level repeatability can discriminate between them.
- **Characterisation tests first** for `compute_defensive_value60` and the `fa_df` sign convention, because the sign is easy to flip (docs say 0.36, code uses −0.10).

## Potential Design Pattern Recommendations

1. **Drop-in `fa_df` replacement (recommended first).** Compute per-player on-ice xGA from rebuilt shift sets and xG from the existing `CoordXGModel`. This keeps `XGWar` unchanged and isolates the effect of better attribution. Pattern: `extract_ozs.py` (precompute to CSV, optional DataFrame argument).
2. **Shrunk RAPM as a second experiment.** Weighted ridge on stints with team-season and strength terms, shrinking toward the on-ice-xGA estimate (a box-score-style prior) instead of toward zero. Pool the three seasons with a season offset. Pattern: archived `processor.py`/`rapm.py`, but with sparse matrices and `GroupKFold`.
3. **Pick by repeatability, not by team correlation.** Keep the current metric if neither new option repeats better.

## Assumptions

1. Shift data (`start_s`/`end_s`) is accurate enough to rebuild on-ice sets at about 89% or better on goals. *Check: compare against goal-row plus/minus columns for all three seasons.*
2. The 2023-24 and 2024-25 files have the same defects and structure as 2025-26. *Check: run the same EDA; only 2025-26 was profiled.*
3. Game ids 210–329 are the regular season in every season's files. *Check: confirm per season from dates and team game counts.*
4. Replacing TOI-share with real on-ice attribution changes player rankings enough to matter. *Check: rank correlation between the two on 2025-26.*
5. Player-level repeatability can be measured with three seasons. *Check: carry out the test; low power is itself a finding.*
6. `CoordXGModel` is acceptable as the xG source for on-ice shots. *Check: confirm it scores the sportsdataverse shots, or fall back to a model fit on their distance/angle columns.*
7. The playoff games in `war_2526.csv` bias xGA60 downward (playoff minutes with no matching shots). *Check: rerun regen with regular-season ids only and compare.*

## Out of Scope

- Non-linear models (`CLAUDE.md` rule).
- Goalie WAR, and any goalie adjustment beyond noting the confound.
- Offensive component changes, including the xG model itself.
- Power-play and penalty-kill splits (deferred until even-strength works).
- Fixing `run_xga_war.py`/`run_combined_war.py`, doc cleanup (0.36 vs −0.10), and the GPW inconsistency, except where they block the experiment. They are listed in the analysis and worth a separate cleanup task.
- Changing the published page or the daily build.

## Immediate Findings Worth Acting On Regardless

- **Playoff mismatch.** `war_2526.csv` (and so the published 2025-26 chart) mixes in 13 playoff games, while its shot data is regular season only. Verified: a player has 39 games against a 30-game regular season.
- **Stale and conflicting documentation.** `CLAUDE.md` and `box_war.py` docstrings still say defence weight 0.36, while production uses −0.10 on a different quantity.
