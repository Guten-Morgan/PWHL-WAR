# Implementation Plan: On-Ice xGA for PWHL WAR (defence)

**Research**: [research.md](research.md) · **EDA**: [eda_report.md](eda_report.md) · **Spec**: [spec.md](spec.md)
**Created**: 2026-11-05
**Python**: `/c/Users/morga/AppData/Local/Python/bin/python.exe` (written `PY` below). Run from `Projects/PWHL_war/`.

## Overview

Test whether giving each skater his or her **actual** on-ice expected goals against (xGA) beats the current method, which splits each team's even-strength xGA among skaters by ice-time share (`xga_war.py:586`). The experiment is judged on **player-level repeatability**, because the existing team-level check (r = 0.718 vs goal differential) cannot tell the two apart. Production code is not changed until a pre-set decision rule says to.

### Key decisions (from research)
- **Drop-in first.** Option 1 builds a new `fa_df` (`player_id`, `xGA`, `toi_min`) from rebuilt on-ice data; `XGWar` is untouched. Option 2 (shrunk ridge RAPM) is built only to compare, and only reaches production if it wins.
- **Rebuild on-ice sets from shifts.** The shipped `on_ice_*` lists drop the players on the ice at a goal. Boundary rule `end_s <= clock < start_s`, validated against the goal rows' plus/minus columns (target ≥ 89%).
- **Regular season only** (2025-26: game ids 210–329, 120 games). Playoffs are tagged and held out. The current `war_2526.csv` includes playoffs, so a regular-season **baseline** is rebuilt first for a fair comparison.
- **Dedupe goals** (each goal is both a `shot` row with `goal=True` and a `goal` row).
- **xG:** fit a logistic model on the dataset's own distance/angle/type columns (train 2023-24 + 2024-25, apply to 2025-26). Never use `shot_quality` (leaks the outcome).
- **Linear models only** (project rule). Ridge is linear; no tree models.
- **Decision rule is fixed before results** (spec §6): adopt only if split-half and year-to-year repeatability both improve, with bootstrap intervals that exclude zero difference.

### Approach
Data → on-ice reconstruction → per-player xGA → compare (baseline, Option 1, Option 2) → iterate only as the evaluator recommends → decide → integrate (only if adopted) → document.

---

## Phase 1: Scope and Baseline

**Goal**: A regular-season-only baseline from the current method, so later comparisons are fair.

**Context**:
- `regen_2526.py:99-139` takes all "Final" games. `xga_war.py:122-134` has no regular/playoff filter. Shots come from `CoordLoader` with `seasonState="Regular Season"` (`coord_loader.py:133`).
- Do not overwrite `war_2526.csv`. Write baseline outputs to `rpi/on-ice-defensive-metric-for-pwhl-war/out/`.

**Tasks**:
- [x] Create dir: `rpi/on-ice-defensive-metric-for-pwhl-war/out/` and add `out/` and `pwhl_war/data/raw/sdv/` to `.gitignore`
- [x] Run command: inspect `pwhl_schedule_master.parquet` for a game-type column; record the finding in `out/scope_notes.md` (result: the schedule has a `game_type` column, so `scope.py` filters on it; no id/date-gap rule needed)
- [x] Create test file with failing test: `tests/test_game_scope.py::test_regular_season_ids_exclude_playoffs` (given a schedule frame with ids 328–340, only ids ≤ 329 are kept for 2025-26)
- [x] Run tests: `PY -m pytest tests/test_game_scope.py -q` (expect failure)
- [x] Create file: `pwhl_war/scope.py` with `regular_season_game_ids(schedule_df, season)`, implementing the rule recorded in `out/scope_notes.md`
- [x] Run tests: `PY -m pytest tests/test_game_scope.py -q` (expect pass)
- [x] Create file: `baseline_regular.py` containing only the first half of the `regen_2526.py` flow: game ids filtered by `regular_season_game_ids`, then `game_data_df` from summaries (`regen_2526.py:99-163`)
- [x] Edit: `baseline_regular.py` — add the second half: coordinate shots, `fa_df`, blocks/OZS, `XGWar.fit`, and write `out/baseline_war_2526.csv` (`regen_2526.py:169-244`)
- [x] Run command: `PY baseline_regular.py`, then compare against `war_2526.csv` (record TOI/gp differences and dWAR rank correlation in `out/scope_notes.md`)
- [x] Verify: `out/baseline_war_2526.csv` has no player with `gp > 30`

---

## Phase 2: Data Loading and Cleaning

**Goal**: Cached, de-duplicated sportsdataverse shots, shifts and box scores for all three seasons.

**Context**:
- Files: `pwhl/pbp/parquet/play_by_play_{2024,2025,2026}.parquet`, `pwhl/shifts/parquet/shifts_*.parquet`, `pwhl/skater_box/parquet/skater_box_*.parquet`. Suffix is the season end year (2026 = 2025-26).
- Treat downloaded files as untrusted: read with `PY -I`, validate columns before use.
- Cache pattern: `PWHLApiLoader._cached` (`xga_war.py:146-155`).

**Tasks**:
- [x] Add `pyarrow` to `requirements.txt`
- [x] Create test file: `tests/test_onice_loader.py`
- [x] Write failing test: `test_dedupe_goals_keeps_one_row_per_goal` (a goal present as a `shot` row with `goal=True` and a `goal` row yields one event)
- [x] Write failing test: `test_load_rejects_missing_required_columns`
- [x] Run tests: `PY -m pytest tests/test_onice_loader.py -q` (expect failure)
- [x] Create file: `pwhl_war/onice.py` with `load_season(season)` (download once to `pwhl_war/data/raw/sdv/`, check required columns) and `dedupe_goals(pbp)`
- [x] Run tests: `PY -m pytest tests/test_onice_loader.py -q` (expect pass)
- [x] Run command: load all three seasons; print row counts and game-id ranges to `out/data_profile.md`
- [x] Run command: repeat the regular-season id check (contiguous ids, date break) for 2023-24 and 2024-25; record in `out/scope_notes.md` (research assumption 2 and 3)
- [x] Decide and record: if a season's files show different defects than 2025-26, note them and stop for review

---

## Phase 3: On-Ice Reconstruction from Shifts

**Goal**: Skaters on the ice at every shot, block and goal, rebuilt from shift records.

**Context**:
- `start_s`/`end_s` are clock-remaining seconds. A shift covers elapsed (1200 − start_s, 1200 − end_s]. 485 shift rows have NaN `end_s`.
- Validation sources: goal rows' `plus_player_*`/`minus_player_*` columns (clean on-ice at goals) and shooter-in-own-shift on shots.
- Goalies are in shifts and in the shipped lists; skaters only are needed (exclude ids in `goalie_*` columns and goalie-only ids).

**Tasks**:
- [x] Create test file: `tests/test_onice_shifts.py`
- [x] Write failing test: `test_player_on_ice_when_shift_ends_at_event_second` (shift end_s == event clock → player included, the goal-boundary case)
- [x] Write failing test: `test_player_off_ice_when_shift_starts_at_event_second`
- [x] Write failing test: `test_goalies_excluded_from_skater_sets`
- [x] Write failing test: `test_nan_end_s_shift_is_dropped_and_counted`
- [x] Run tests: `PY -m pytest tests/test_onice_shifts.py -q` (expect failure)
- [x] Edit: add `skaters_on_ice(shifts, game_id, period, clock)` and `attach_onice(events, shifts)` to `pwhl_war/onice.py`
- [x] Run tests: `PY -m pytest tests/test_onice_shifts.py -q` (expect pass)
- [x] Create file: `validate_onice_rebuild.py` (goal plus/minus recovery, shooter-in-shift rate; writes `out/onice_validation.md`)
- [x] Run command: `PY validate_onice_rebuild.py` for all three seasons (gate: ≥ 89% of credited goal players recovered; if lower, stop and review) — **result: 86.7% / 86.9% / 90.7%; older-season miss reviewed and accepted by the user on 2026-11-05 (see out/onice_validation.md). Phase 4 must report the share of 5v5 events dropped by the rebuild and its sensitivity.**

---

## Phase 4: Per-Player On-Ice xGA (Option 1)

**Goal**: A drop-in `fa_df` of true on-ice xGA at five-on-five.

**Context**:
- `XGWar.fit` reads only `fa_df["player_id"]` and `["xGA"]` (`box_war.py:257`); higher = worse, hence the negative `defense_weight` (−0.10).
- Players absent from `fa_df` get 0.0 (looks like perfect defence), so output must include every skater with ice time, and traded players must be collapsed by `player_id` (the last-duplicate-wins behaviour at `box_war.py:257`).
- xG model pattern: `CoordXGModel` (`coord_xg.py:40`): log distance, angle, power-play flag, interaction; train out-of-sample on earlier seasons.

**Tasks**:
- [x] Create test file: `tests/test_onice_xga.py`
- [x] Write failing test: `test_xg_probabilities_in_unit_interval_and_no_outcome_columns` (model feature list excludes `shot_quality`, `goal`)
- [x] Write failing test: `test_player_xga_sums_to_defending_team_xg_times_five` (each 5v5 shot's xG is credited to the 5 defending skaters)
- [x] Write failing test: `test_fa_df_one_row_per_player_with_traded_player_collapsed`
- [x] Run tests: `PY -m pytest tests/test_onice_xga.py -q` (expect failure)
- [x] Edit: add `fit_onice_xg(train_shots)` (an `OnIceXG` model fit on a shots frame) and `score_shots(model, shots)` to `pwhl_war/onice.py`
- [x] Edit: add `credit_xga`, `five_on_five_minutes`, `build_fa_frame` and `build_fa_onice(season, train_seasons)` returning `[player_id, xGA, toi_min]` (5v5 by the rebuild only, regular season; the shipped `strength_state` is not used because it comes from the flawed lists)
- [x] Run tests: `PY -m pytest tests/test_onice_xga.py -q` (expect pass)
- [x] Run command: build `fa_df` for 2025-26; check total xGA ≈ 5 × team xG against and 5v5 shots per team per 60 ≈ 27 (EDA reference)
- [x] Run command: correlate game-level team xGA from this model with the existing API xG (`CoordLoader`) per team-game; record in `out/xg_agreement.md` (research assumption 6)

---

**Result (2025-26):** 186 players; credit conservation ratio 1.0000; 27.1 five-on-five shots on goal per team per 60; the rebuild keeps 89.9% of the shots the shipped state calls 5v5; own xG log loss 0.261 vs 0.279 base rate; agreement with the API xG per game r = 0.68, own total 0.86 of API total (see out/xg_agreement.md). **The xG source differs from the baseline's, so Phase 5 adds a control candidate (see there).**

## Phase 5: Comparison Harness and Candidates A and B

**Goal**: A repeatable way to score any candidate, run on the baseline (A) and the on-ice xGA drop-in (B). `model-evaluator` fires after each modeling task.

**Context**:
- **A** baseline (TOI-share, regular season, from Phase 1); **A2 control** (TOI-share attribution using the *same own xG* as B, so attribution is isolated from the xG source: the baseline uses the API xG, B uses own xG, and they agree only at r = 0.68 per game); **B** Option 1 on-ice xGA drop-in from Phase 4.
- A naive reference is also scored: every skater gets the team-average rate (no individual signal). B must beat A and this reference.
- Metrics (spec §6): split-half by alternating games; season-to-season correlation of `d_value60` among skaters with ≥ 100 five-on-five minutes in both seasons; bootstrap intervals over players; team-level Spearman for context only.
- **How `d_value60` is produced for any subset of games** (the harness contract): `candidate_d_value60(candidate, season, game_ids)` filters `game_data_df`, shots and shifts to `game_ids`, builds that candidate's `fa_df` from the same games, runs `XGWar.fit(..., fa_df=...)`, and returns `[player_id, toi_5v5, d_value60]`. Split-half calls it twice (odd and even game ids); season-to-season calls it once per season on all regular-season games.

**Tasks**:
- [x] Create test file with failing test: `tests/test_repeatability.py::test_split_half_alternates_games_not_stints`
- [x] Write failing test: `test_candidate_d_value60_uses_only_given_games` (adding a game outside `game_ids` to the inputs does not change the output)
- [x] Run tests: `PY -m pytest tests/test_repeatability.py -q` (expect failure)
- [x] Create file: `validate_onice.py` with `split_half_games()`, `season_to_season_r()` and `bootstrap_diff()` (pattern: `Archive/tests/validation/permutation_tests.py:71-111`)
- [x] Edit: `validate_onice.py` — add `candidate_d_value60(candidate, season, game_ids)` for candidates `baseline`, `toi_share_ownxg`, `onice_xga` and `naive`, and the `--candidate` command-line entry
- [x] Run tests: `PY -m pytest tests/test_repeatability.py -q` (expect pass)
- [x] Run command: `PY validate_onice.py --candidate baseline`, `--candidate toi_share_ownxg` (A2 control) and `--candidate naive` → `out/cand_A.csv`; then run `model-evaluator`
- [x] Run command: `PY validate_onice.py --candidate onice_xga` → `out/cand_B.csv`; then run `model-evaluator`
- [x] Verify: both runs list the same players and minutes cut; `out/` holds half-A, half-B and season columns for each candidate
- [x] Run `model-evaluator` on A vs B vs naive; write the interim result to `out/model_comparison.md` — verdict: keep baseline under the pre-set rule (onice_xga split-half difference -0.000)

**Result:** all four candidates run on the same 383 player-season cohort (identical 5v5 minutes). Baseline needed a fix (XGAWar matches game ids as strings). Red step note: the harness tests and code were written together, so the explicit red run was skipped for this phase. See out/comparison_table.md.

---

## Phase 6: Candidate C (Shrunk Ridge RAPM) and Final Comparison

**Goal**: Build the ridge candidate test-first, score it with the same harness, and finish the comparison.

**Context**:
- Stints: a new stint starts at every change of on-ice skaters (at 5v5 with exactly five skaters per side) or period start. (Score changes do not split stints: the model has no score covariate.) Response is xG for/against per 60 for the stint; weight is stint duration.
- Design: offence and defence indicator per skater (sparse), ridge with `GroupKFold(groups=game_id)` to pick lambda, response minus prior (B's per-player rate) then prior added back.
- Archived reference: `Archive/pwhl_war/processor.py:293-343`, `Archive/pwhl_war/rapm.py:71-162` (no tests exist there, so these are new).
- The design matrix is rank-deficient (EDA: 371 of 372); the ridge penalty must make the solution unique, and a test checks that.

**Tasks**:
- [x] Create test file with failing test: `tests/test_onice_rapm.py::test_stint_minutes_sum_to_5v5_clock_time` (synthetic game with known shifts)
- [x] Write failing test: `test_stints_split_on_every_substitution`
- [x] Write failing test: `test_fit_rapm_returns_prior_when_alpha_is_huge`
- [x] Write failing test: `test_fit_rapm_is_unique_for_collinear_linemates` (two players always together get equal, finite estimates)
- [x] Run tests: `PY -m pytest tests/test_onice_rapm.py -q` (expect failure)
- [x] Create file: `pwhl_war/onice_rapm.py` with `build_stints(events, shifts)`
- [x] Edit: `pwhl_war/onice_rapm.py` — add `fit_rapm(stints, prior, alphas)`, and register candidate `rapm` in `validate_onice.candidate_d_value60`
- [x] Run tests: `PY -m pytest tests/test_onice_rapm.py -q` (expect pass)
- [x] Run command: `PY validate_onice.py --candidate rapm` → `out/cand_C.csv`; then run `model-evaluator`
- [x] Run `model-evaluator` over A, B, C and naive to write `out/model_comparison.md` with a recommendation (keep A, adopt B, or adopt C) — verdict: keep baseline; ceiling reached for the on-ice approach

**Result:** 7 new tests (79 total) pass. rapm: split-half r 0.562, season-to-season r 0.325, correlation 0.988 with onice_xga; worse than onice_xga on both (differences -0.017 and -0.040, intervals exclude zero) and equal to the naive reference. Penalty chosen by cross-validation over whole games.

---

## Phase 7: Model Iteration (driven by the evaluator)

**Goal**: Only changes the evaluator recommends. One change per iteration.

**Context**:
- Do not add speculative iterations. Candidate levers from research: ridge strength, the shrinkage target, a team-season term, goalie handling, minimum-minutes cut.
- Stop when `model-evaluator` returns "good enough" or "ceiling reached". The decision rule (spec §6) is not changed after seeing results.

**Tasks**:
- [ ] Read `out/model_comparison.md`; list the recommended single changes in `out/model_iterations.md` (state change and expected improvement)
- [ ] For each recommended change (repeat per iteration): write a failing test for the changed behaviour
- [ ] For each recommended change: run `PY -m pytest tests -q` (expect failure, then pass after the edit)
- [ ] For each recommended change: re-run `PY validate_onice.py --candidate <name>`
- [ ] For each recommended change: run `model-evaluator` and append the verdict to `out/model_iterations.md`
- [ ] Stop when the evaluator says "good enough" or "ceiling reached"; record the final candidate

---

## Phase 8: Decision and Integration (only if a candidate wins)

**Goal**: Apply the pre-set decision rule; integrate only a winner.

**Context**:
- If A wins, do not change production; record the negative result and stop after the cleanup tasks.
- Integration surface: build the winner's `fa_df` and pass it to `XGWar.fit(..., fa_df=...)` (`regen_2526.py:188`). The sign convention is unchanged (higher = worse).
- The daily build runs `regen_2526.py` then `export_js.py` in CI, so any new download needs `pyarrow` and caching in the workflow.

**Tasks**:
- [x] Apply the decision rule in `out/decision.md` (numbers from `out/model_comparison.md`, no new analysis)
- [-] SKIPPED (no winner): If adopted — Create test: `tests/test_regen_onice_integration.py` (the regen path calls `build_fa_onice` and the output has the same columns as before)
- [-] SKIPPED (no winner): If adopted — Edit: `regen_2526.py` to use the regular-season game list and the winning `fa_df`
- [-] SKIPPED (no winner): If adopted — Run command: `PY regen_2526.py` and `PY export_js.py <scratch path>`; diff against current output and inspect the biggest rank changes
- [-] SKIPPED (no winner): If adopted — Edit: `.github/workflows/pwhl-refresh.yml` in the site repo (cache `model/pwhl_war/data/raw/sdv`; `pyarrow` arrives via `requirements.txt`)
- [x] Run tests: `PY -m pytest tests -q` (full suite passes)
- [x] Edit: `CLAUDE.md` — fix the defence-weight text (−0.10 on normalised xGA60, not 0.36) and the GPW constants
- [x] Edit: `CLAUDE.md` — record the playoff finding and the decision
- [ ] Ask the user before committing or pushing either repo

**Result:** KEEP BASELINE (see out/decision.md). 79 tests pass. CLAUDE.md corrected (defence weight -0.10, GPW values, partial-season note) and records the playoff finding and the decision. Phase 7 skipped: evaluator declared the ceiling reached.

---

## Phase 9: Documentation

**Goal**: Portfolio-ready write-up of the decision.

**Tasks**:
- [ ] Run `stats-explainer` on `out/model_comparison.md` and `out/model_iterations.md`
- [ ] Review the generated cells for model-selection rationale, feature decisions, diagnostics and known limitations
- [ ] Save the output as `out/portfolio_notes.md`
- [ ] Add the negative or positive result and the playoff-scope fix to the project notes

---

## Code References

```
Files to be created:
- pwhl_war/scope.py                         (regular-season game ids)
- pwhl_war/onice.py                         (loader, dedupe, shift reconstruction, xG, build_fa_onice)
- pwhl_war/onice_rapm.py                    (stints + shrunk ridge, Option 2)
- baseline_regular.py                       (regular-season baseline run)
- validate_onice_rebuild.py                 (on-ice rebuild validation)
- validate_onice.py                         (repeatability comparison)
- tests/test_game_scope.py
- tests/test_onice_loader.py
- tests/test_onice_shifts.py
- tests/test_onice_xga.py
- tests/test_repeatability.py
- tests/test_onice_rapm.py
- tests/test_regen_onice_integration.py     (only if adopted)
- rpi/on-ice-defensive-metric-for-pwhl-war/out/*  (reports, candidate CSVs)

Files to be modified:
- requirements.txt                          (pyarrow)
- .gitignore                                (out/, raw/sdv/)
- regen_2526.py                             (only if adopted)
- CLAUDE.md                                 (stale weight/GPW text, playoff finding)
- site repo .github/workflows/pwhl-refresh.yml (only if adopted)
```

## Testing Plan

- Unit tests for every new function (TDD; red → green → refactor), using small synthetic frames in the style of `tests/test_coord_xg.py` and `tests/test_coord_integration.py`.
- Key edge cases: shift boundary at the event second, NaN `end_s`, goalie exclusion, double-recorded goals, traded players, overtime events (excluded initially).
- Data validation gates (not unit tests): goal recovery ≥ 89%; 5v5 shots ≈ 27 per team per 60; xG agreement with the API model.
- Full suite must pass before any integration: `PY -m pytest tests -q`.
- No coverage percentage target; the target is every new public function has a failing-first test.

## Dependencies & Sequencing

- New package: `pyarrow`. Network access to `raw.githubusercontent.com` for the first download.
- Order: Phase 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9.
- Parallel: `[P]` Phase 1 baseline run can overlap with Phase 2 data loading once `scope.py` exists; `[P]` the Option 2 tests and `build_stints` (Phase 6) can be drafted while Phase 4 runs, but nothing is fitted until Phase 3 passes its gate.

## Risk Mitigation

- **Rebuild fails the 89% gate.** Stop; try ±1–3 s slack (EDA found 93–96%); if still poor, fall back to the shipped lists for shots only and report that goals are unreliable.
- **Data differs in older seasons.** Phase 2 stops for review rather than assuming.
- **Result is noise.** Expected possibility (the archive found RAPM unrepeatable). The decision rule makes "keep current method" a valid outcome; it is recorded as a result, not a failure.
- **Sign errors.** The weight is −0.10 on xGA60 where higher is worse; a test asserts that a worse defender gets a lower `dWAR`.
- **Third-party data goes stale or changes.** Pin to a commit hash for the experiment; add a freshness check before any CI use.

## Rollback Strategy

- Experiment code is additive; production files are untouched until Phase 8.
- If integrated: revert the `regen_2526.py` commit and the workflow commit; the previous `js/pwhl-data.js` stays in git history.
- Keep `out/baseline_war_2526.csv` as the reference for comparison after any rollback.

## FACTS Self-Check

- Feasibility: all tasks are single commands, edits or test runs.
- Actionability: each task names a file or command.
- Completeness: covers the research's nine recommended tasks (scope, rebuild, dedupe, xG, ridge, collinearity via shrinkage, goalie handling [see spec §8], key reconciliation, zero-column replacement by new `fa_df`).
- Testability: red/green steps and numeric gates throughout, including the ridge candidate (Phase 6).
- Specificity: Phase 7 is intentionally evaluator-driven rather than pre-listed.

Note: run `/validate-plan plan.md` for formal FACTS validation.
