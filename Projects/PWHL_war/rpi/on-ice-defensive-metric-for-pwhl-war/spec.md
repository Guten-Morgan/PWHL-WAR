# Specification: On-Ice xGA Defensive Metric for PWHL WAR

**Version**: 1.0
**Date**: 2026-11-05
**Status**: Draft (ready for implementation once the plan is validated)

---

## 1. Purpose

Test whether using each skater's actual on-ice expected goals against (xGA) gives a more repeatable defensive rating than the current method, which shares each team's even-strength xGA among its skaters in proportion to ice time. If it does, replace the current input to the defensive component of the PWHL Wins Above Replacement (WAR) model; if not, keep the current method and record the result.

---

## 2. Goals and Non-Goals

**Goals**
- Build, per season, a table of one row per skater with total five-on-five xGA while that skater was on the ice and the skater's five-on-five minutes, in the shape the existing model already accepts.
- Rebuild "who was on the ice" for every shot, block and goal from shift records, and validate the rebuild against independent data.
- Compare the current method, the on-ice method, and a shrunk ridge regression (RAPM) on the same players, using player-level repeatability.
- Apply a decision rule fixed before looking at results.
- Produce a regular-season-only baseline of the current method, since the published 2025-26 data includes playoff games.

**Non-Goals**
- Changing the offensive component or the offensive xG model.
- Power-play and penalty-kill defence; goalie ratings; non-linear models.
- Changing the published page, its design, or the daily build, unless a candidate is adopted.
- Fixing `run_xga_war.py` / `run_combined_war.py` or general documentation drift beyond the items listed in plan Phase 8.

---

## 3. Background and Context

**The model.** The WAR model (Python package `pwhl_war`) rates each skater as offence plus defence. Offence is individual expected goals per 60 minutes (`o_xG60`). Defence starts from a per-player `xGA` total supplied in a table `fa_df`. `XGWar.fit` divides by minutes (xGA60), divides by the standard deviation across qualified players (≥ 50 minutes), removes the part explained by `o_xG60`, a defenseman flag and optionally offensive-zone-start share by ordinary least squares, subtracts the league mean, and multiplies by `defense_weight = -0.10`. The weight is negative because higher xGA means worse defence. Only `fa_df["player_id"]` and `fa_df["xGA"]` are read; any player missing from `fa_df` is treated as having xGA 0, which looks like perfect defence. If a player_id appears twice (a traded player), the last row wins.

**How xGA is built today.** For each game, the opposing team's even-strength xG (from an API that gives xG per shot) is summed, then credited to each skater in proportion to his or her share of the team's ice time: `xGA += team_xGA * (toi / team_total_toi)`. The API has no on-ice player lists, so every skater on a team gets the same xGA per minute in a game. Plus/minus is only a fallback.

**Why it might be improved.** A new public dataset (sportsdataverse `fastRhockey-pwhl-data`, MIT licence, updated daily in season) has shift records and on-ice player lists for 2023-24, 2024-25 and 2025-26.

**What the data looks like (2025-26; other seasons not yet profiled).**
- Play-by-play: 23,838 rows. Events include `shot` (7,438, of which 601 are goals), `goal` (602), `blocked_shot` (3,197), `hit`, `faceoff`, `penalty`, `goalie_change`, `shootout`. Goals are recorded **twice**: once as a `shot` row with `goal=True` and once as a `goal` row.
- There is no xG column. Usable shot fields: `shot_distance`, `shot_angle`, `x_coord`, `y_coord`, shot type (`event_type`), `empty_net`, power-play flags. `shot_quality` includes outcome categories and leaks the result: never use it as a feature.
- `on_ice_home` and `on_ice_away` are comma-separated player ids including the goalie (six ids at five-on-five); `strength_state` is home-versus-away and derived from the same lists; `strength_state_valid` is true for 94.9% of shots.
- **The shipped lists are wrong at goals.** They match shifts built with the rule `end_s < clock <= start_s`, which drops a player whose shift ends exactly at the event second. The shift feed ends shifts at the goal's clock second, so only 46% of scoring-team and 37% of conceding-team players (as credited in the goal rows' `plus_player_*`/`minus_player_*` columns) appear in the lists. The rule `end_s <= clock < start_s` recovers 89% of those players (about 10% remains clock jitter; ±3 seconds of slack gives 93–96%).
- Shifts: 104,487 rows. `start_s`/`end_s` are clock-remaining seconds within the period; a shift covers elapsed time (1200 − start_s, 1200 − end_s]. 485 rows have NaN `end_s`. Shift minutes sum to box-score minutes within 0.05 per player-game. Shifts include goalies.
- Games: regular season is game ids 210–329 (120 games, no gaps, last on 2026-04-25); playoffs are ids 338–351 (13 games, from 2026-04-30). There is no playoff flag. Game 282 is dated 04-03 but its id sits among March ids.
- Volume at five-on-five: 4,817 shots on goal, about 27.4 per team per 60 minutes, goal rate 8.1% of shots on goal. 155 skaters have at least 100 five-on-five minutes (103 forwards, 52 defenders). A median skater (302 minutes) sees about 138 shots and about 11 goals against, so goals alone are too sparse.
- Collinearity: 25 teammate pairs share more than 70% of their ice time (for example 90% for Jenner–Leslie); the stint design matrix is rank-deficient.
- Goalies: a single goalie covers 54–94% of a team's shots; 21 goalies in the season.
- Player ids agree across play-by-play, shifts, box scores and the model's files. Team names are corrupted for Montréal; join on ids.

**Problems in the current pipeline that shape the work.**
- `war_2526.csv` (the source of the published 2025-26 data) is built from all 133 games, playoffs included (a player has 39 games against a 30-game regular season). Its shot data is regular season only, so playoff minutes may carry no matching xGA. This is unconfirmed and is measured in plan Phase 1.
- The model's `game_data_2526.csv` stops at game 270 and has all-zero on-ice defensive columns, which is why the model fell back on the TOI-share method.
- Ten player ids appear twice in `war_2526.csv` (traded players).
- An earlier ridge RAPM attempt was archived because it was no more repeatable than noise: too few teams (8), 30 games each, and no usable on-ice data. Reactivation conditions were at least 12 teams, three full seasons pooled, and real on-ice data; the third now holds, the first does not.
- The existing validation, a team-level Spearman correlation with goal differential (r = 0.718), cannot separate the methods because team totals are the same under both.

**Prior art.** Evolving Hockey and HockeyStats use stint-weighted ridge regression on xGF/60 with score and strength covariates, and shrink toward priors because single-season defensive estimates are unstable; Evolving Hockey's WAR uses 11 years of pooled RAPM. No published reliability figures exist for xGA RAPM at PWHL scale, so the outcome is genuinely uncertain.

---

## 4. Technical Design

### 4.1 Architecture / Approach

```
sportsdataverse parquet (pbp, shifts, box)
   -> load + validate columns + cache        (pwhl_war/onice.py)
   -> de-duplicate goals; keep periods 1-3; drop events without usable lists
   -> rebuild on-ice skaters from shifts     (boundary rule end_s <= clock < start_s)
   -> score each shot with an xG model       (train 2023-24 + 2024-25, apply 2025-26)
   -> per-player 5v5 xGA and minutes         = Option 1  -> fa_df
   -> stints + weighted ridge, shrunk to Option 1       = Option 2 -> fa_df
   -> validate_onice.py: compare A (current), B (Option 1), C (Option 2)
   -> decision rule -> (if adopted) regen_2526.py passes the winner as fa_df to XGWar.fit
```

Candidates:
- **A (baseline):** current TOI-share xGA, regular-season games only.
- **B (Option 1):** on-ice xGA from rebuilt on-ice sets.
- **C (Option 2):** weighted ridge on five-on-five stints, offence and defence coefficient per skater, shrunk toward B's per-player rate instead of zero.

All three feed the same downstream math so only the xGA input differs.

### 4.2 Interfaces and APIs

All in `pwhl_war/onice.py` unless noted. Types are pandas DataFrames unless stated.

```python
def regular_season_game_ids(schedule_df: DataFrame, season: str) -> list[int]
    # pwhl_war/scope.py. season like "2025-26". Returns regular-season ids only.
    # Raises ValueError if the schedule frame lacks game_id or date columns.

def load_season(season: str, refresh: bool = False) -> dict[str, DataFrame]
    # Returns {"pbp": ..., "shifts": ..., "box": ...}. Downloads once to
    # pwhl_war/data/raw/sdv/, then reads the cache. Raises ValueError if a required
    # column (see 4.3) is missing. Never executes downloaded content.

def dedupe_goals(pbp: DataFrame) -> DataFrame
    # Removes the `goal` event rows that duplicate a `shot` row with goal=True,
    # keeping one row per goal. Does not change non-goal rows.

def skaters_on_ice(shifts: DataFrame, game_id: int, period: int, clock: float) -> dict
    # clock = seconds remaining in the period at the event.
    # Returns {"home": set[int], "away": set[int]} of skater ids (no goalies) with
    # end_s <= clock < start_s. Rows with NaN end_s are ignored.

def attach_onice(events: DataFrame, shifts: DataFrame) -> DataFrame
    # Adds columns home_skaters, away_skaters (tuple[int]) and n_home, n_away.
    # Also adds `rebuild_ok` (False when no shift covers the event).

def fit_onice_xg(train_shots: DataFrame) -> OnIceXG
    # Logistic regression on shots on goal: log distance, angle, shot type,
    # power-play / short-handed flags, interaction. Excludes shot_quality, goal, scoring_chance.

def score_shots(model: OnIceXG, shots: DataFrame) -> Series   # xG per shot, in [0, 1]

def build_fa_onice(season: str, train_seasons: list[str]) -> DataFrame
    # Returns columns [player_id:int, xGA:float, toi_min:float], one row per skater
    # with any five-on-five ice time, regular season only, traded players
    # summed across teams. Higher xGA = worse defence.
```

`pwhl_war/onice_rapm.py`:

```python
def build_stints(events: DataFrame, shifts: DataFrame) -> DataFrame
    # One row per interval with constant on-ice skaters at 5v5, with duration
    # in seconds, xG for and against, and the two skater sets.

def fit_rapm(stints: DataFrame, prior: DataFrame, alphas: list[float]) -> DataFrame
    # Weighted ridge (sample_weight = duration), sparse design, offence and defence
    # coefficient per skater, response minus prior, prior added back.
    # Lambda chosen by GridSearchCV with GroupKFold(groups=game_id).
    # Returns [player_id, xGA, toi_min] in the same units as build_fa_onice.
```

`validate_onice.py` (command line):

```
python validate_onice.py --candidate {baseline,onice_xga,rapm} [--season 2025-26]
   -> writes rpi/on-ice-defensive-metric-for-pwhl-war/out/cand_{A,B,C}.csv
python validate_onice_rebuild.py
   -> writes out/onice_validation.md (goal recovery, shooter-in-shift rate)
```

Exit code is nonzero when a gate fails (4.5).

### 4.3 Data Models and Schemas

Required columns, checked on load (names exact):
- `pbp`: `game_id, event, period_of_game, time_of_period, goal, goalie_id, player_id, plus_player_one_id … plus_player_five_id, minus_player_one_id … minus_player_five_id, shot_distance, shot_angle, event_type, empty_net, power_play, short_handed, strength_state, strength_state_valid, home_team_id, away_team_id, team_id, game_date`
- `shifts`: `game_id, player_id, home, period, start_s, end_s`
- `box`: `game_id, player_id, position, team_id, toi`

Season file suffix is the end year: `2024` = 2023-24, `2025` = 2024-25, `2026` = 2025-26.

Output table (`fa_df`):

| column | type | meaning | example |
|---|---|---|---|
| `player_id` | int | HockeyTech player id | `24` |
| `xGA` | float | total five-on-five xG against while on ice, regular season | `9.84` |
| `toi_min` | float | five-on-five minutes on ice | `302.4` |

Candidate result CSV (`out/cand_*.csv`): `player_id, season, pos, toi_5v5, xGA, xGA60, d_value60, half_a_d_value60, half_b_d_value60`.

### 4.4 Configuration and Environment

- Python: `/c/Users/morga/AppData/Local/Python/bin/python.exe`; run from `Projects/PWHL_war/`.
- New dependency: `pyarrow` (add to `requirements.txt`).
- Network: first run needs `raw.githubusercontent.com`; later runs use the cache in `pwhl_war/data/raw/sdv/` (git-ignored).
- Constants: `defense_weight = -0.10`, minimum qualified minutes for ranking 100 five-on-five, replacement percentile 25, regular-season ids per season recorded in `out/scope_notes.md`.
- Read downloaded data with `python -I`; do not run code from the cache directory.

### 4.5 Dependencies

pandas, numpy, scikit-learn (logistic regression, `Ridge`, `GridSearchCV`, `GroupKFold`), scipy (sparse matrices, `spearmanr`), pyarrow; existing `pwhl_war.box_war.XGWar`, `pwhl_war.stats_utils.compute_defensive_value60`, `pwhl_war.coord_xg.CoordXGModel` (pattern only). pytest for tests.

---

## 5. Key Decisions and Rationale

1. **Decision:** drop-in `fa_df`, not a new defence formula.
   - Alternatives: rewrite the defence formula around RAPM coefficients.
   - Rationale: isolates the effect of better attribution, keeps all other model settings, and `XGWar` only reads two columns.
2. **Decision:** rebuild on-ice sets from shifts with `end_s <= clock < start_s`.
   - Alternatives: trust the shipped lists.
   - Rationale: the shipped lists are wrong for about half of the players at goals; the rebuild recovers 89% against independent data.
3. **Decision:** regular season only; playoffs tagged and held out.
   - Alternatives: all 133 games.
   - Rationale: the shot data used elsewhere is regular season only; mixing scopes biases rates; playoff minutes are higher-intensity and a different sample.
4. **Decision:** xG from a logistic model fit on the dataset's own distance, angle and shot type, trained on earlier seasons and applied out of sample.
   - Alternatives: the dataset's `shot_quality` label; the hockey-statistics API xG.
   - Rationale: `shot_quality` leaks the outcome; the API xG has no on-ice players. Agreement between the two is checked (§7, scenario 9).
5. **Decision:** five-on-five only, periods 1–3.
   - Alternatives: all strengths, overtime.
   - Rationale: the existing defence input is even-strength; overtime lists match shifts only 29% of the time.
6. **Decision:** xGA, not goals against, as the response.
   - Rationale: about 11 goals against per skater at median minutes is too sparse; shot-based targets are far more repeatable.
7. **Decision:** no goalie term in the first version.
   - Alternatives: goalie fixed effects.
   - Rationale: xG depends on shot features, not saves, so goalie quality does not enter xGA; a goalie term would be nearly collinear with team for several teams. Revisit only if the evaluator recommends it.
8. **Decision:** judge by player-level repeatability, with team-level correlation reported for context only.
   - Rationale: team totals are identical under TOI-share and on-ice attribution, so the team check cannot discriminate.
9. **Decision:** linear models only.
   - Rationale: project rule; ridge is linear.

---

## 6. Acceptance Criteria

1. Given the 2025-26 files, `dedupe_goals` returns exactly one event per goal (601 goals, not 1,203 rows).
2. The on-ice rebuild recovers at least 89% of the players credited on goal rows (`plus_player_*`, `minus_player_*`) for each of the three seasons, and the shooter is in his or her own rebuilt set on at least 88% of shots.
3. `build_fa_onice("2025-26", …)` returns one row per `player_id`; `xGA` and `toi_min` are non-negative; the sum of `xGA` over skaters equals five times the sum of the defending teams' five-on-five xG within 1%.
4. League five-on-five shots on goal are between 25 and 30 per team per 60 minutes.
5. No feature used by the xG model is `shot_quality`, `goal`, or `scoring_chance` (asserted by a test).
6. A defender with higher on-ice xGA than a teammate with the same minutes, other things equal, receives a lower `dWAR` (sign test through `XGWar`).
7. `baseline_regular.py` output contains no player with more games than the 30-game regular season.
8. `validate_onice.py` reports, for each candidate, the split-half correlation (odd versus even game ids) and the season-to-season correlation of `d_value60` among skaters with at least 100 five-on-five minutes in both seasons, each with a bootstrap 95% interval over players (2,000 resamples, seed 42).
9. **Decision rule (fixed now):** a candidate replaces the current method only if both its split-half correlation and its season-to-season correlation exceed the baseline's, and the bootstrap interval of each difference excludes zero. Otherwise the current method stays and the result is recorded.
10. Nothing in production (`regen_2526.py`, `export_js.py`, the site repo) is modified before criterion 9 is evaluated; nothing is committed or pushed without the user's approval.
11. The full test suite passes (`pytest tests -q`).

---

## 7. Test Scenarios

| # | Scenario | Input | Expected output | Notes |
|---|---|---|---|---|
| 1 | Goal duplicate | one `shot` row (`goal=True`) and one `goal` row for the same goal | one goal event | Prevents the 8% inflation |
| 2 | Missing column | parquet without `end_s` | `ValueError` naming the column | Third-party schema drift |
| 3 | Shift ends at the event second | skater with `end_s == clock` | skater is on the ice | The goal-boundary bug |
| 4 | Shift starts at the event second | skater with `start_s == clock` | skater is off the ice | Half-open interval |
| 5 | Goalie in shifts | shift rows for a goalie id | goalie absent from skater sets | Sets are skaters only |
| 6 | NaN end | shift with `end_s` NaN | ignored, counted in the validation report | 485 such rows |
| 7 | Traded player | player with rows for two teams | one `fa_df` row, xGA and minutes summed | Avoids last-row-wins |
| 8 | xG bounds | all scored shots | every value in [0, 1] | |
| 9 | Credit conservation | a 5v5 shot with xG 0.10 | each of 5 defenders gains 0.10; total 0.50 | |
| 10 | Overtime | event in period 4 | excluded | Lists unreliable |
| 11 | Impossible strength | shot with a 5v2 list | excluded | 421 such rows |
| 12 | Split-half | games ids 1..10 | odd ids in one half, even in the other, stints never split across halves | |
| 13 | Sign | two defenders, equal minutes, xGA 8 vs 12 | the 12 player has the lower `dWAR` | Weight is −0.10 |
| 14 | Regular-season scope | schedule with ids 328–340 | ids ≤ 329 kept | Boundary from this season |
| 15 | Decision rule | candidate with a higher split-half r but an interval including zero | keep current method | Rule is conservative |

---

## 8. Constraints and Known Limitations

- Small league: 8 teams, 30 games each, about 155 skaters with at least 100 five-on-five minutes. The earlier RAPM attempt failed on repeatability; "no improvement" is a likely and acceptable outcome.
- Teammate collinearity: 25 pairs share over 70% of ice time, so credit between such pairs will be split almost evenly under any method.
- About 10% of on-ice reconstruction at goals is clock jitter even with the best rule.
- Overtime, shootouts and power-play/penalty-kill are excluded.
- Goalie and team effects cannot be fully separated (one goalie covers 54–94% of a team's shots); no goalie term in the first version.
- Only 2025-26 has been profiled; the older seasons may have different defects (checked in plan Phase 2).
- Third-party data: no guarantee of schema stability; a freshness and schema check is required before any CI use. The dataset's own xG model is not used.
- The existing team-level validation cannot distinguish the methods and is reported for context only.

---

## 9. Out-of-Scope Follow-ups

- Power-play and penalty-kill defence; a blended all-strength defence.
- Goalie ratings and goalie-adjusted xGA.
- Offensive on-ice impact (RAPM for offence) and replacing individual xG.
- Fixing `run_xga_war.py` and `run_combined_war.py`, which do not run as written.
- Reconciling goals-per-win values (4.6667 in `regen_2526.py`, 4.508 in `constants.py`, 6.0 fallback).
- A multi-season pooled prior in the style of Evolving Hockey once more seasons exist.
- Adding the 2026-27 season to the build when the schedule appears.

---

## 10. Glossary

- **xG / xGA:** expected goals; expected goals against.
- **5v5:** five skaters per side (goalies excluded).
- **On-ice:** the skaters on the ice at an event's time.
- **Stint:** a stretch of play with a constant set of skaters at five-on-five.
- **RAPM:** regularized adjusted plus-minus; here, a ridge regression of stint xG on skater indicators.
- **Ridge / shrinkage:** a penalty that pulls estimates toward a prior value, reducing noise from sparse data.
- **Split-half:** correlate a metric computed from odd-numbered games with the same metric from even-numbered games.
- **TOI-share attribution:** the current method of crediting team xGA by ice-time share.
- **`fa_df`:** the per-player table `[player_id, xGA, toi_min]` the model reads for defence.
- **Clock remaining:** seconds left in the period; `start_s` and `end_s` in shifts use this convention.
