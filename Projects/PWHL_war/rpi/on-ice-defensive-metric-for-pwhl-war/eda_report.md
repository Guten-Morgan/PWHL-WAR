# EDA Report: On-ice defensive metric (xGA / RAPM) for PWHL WAR

**Date**: 2026-11-05
**Datasets profiled** (sportsdataverse fastRhockey-pwhl-data, 2025-26, read with `python -I`, nothing written to the data folder):
- `play_by_play_2026.parquet` (23,838 rows x 104 cols)
- `shifts_2026.parquet` (104,487 x 14)
- `skater_box_2026.parquet` (5,041 x 22)
- Model side: `pwhl_war/data/war_2526.csv` (182 x 37), `pwhl_war/data/raw/game_data_2526.csv` (2,550 x 32)

Scratch scripts live in the session scratchpad (p2..p26.py); none are in the repo.

---

## 1. Games: regular season vs playoffs

- 133 distinct game_ids (210-351). **No playoff flag column exists**; split inferred.
- **Regular season = game_id 210-329 = 120 games** (all 120 ids present, no gaps). Last regular game dated 2026-04-25 (ids 326-329).
- **Playoffs = 13 games, ids 338-351** (330-337 and 346 absent from the file), first dated 2026-04-30, last 2026-05-20. Series shape fits 4 + 5 + 4 games (BOS-OTT 338/339/343/344; MTL-MIN 340/341/342/345/347; OTT-MTL 348-351).
- **Confidence: high** (id contiguity, date break 04-25 to 04-30, 13 = 120 + 13 = 133, and team game counts in box: MTL 39, OTT 38, BOS 35, MIN 34 vs 30 for the other four).
- Schedule: 1 game (282) is dated 2026-04-03 though its id sits between 03-10 and 03-11 (rescheduled; id order is not date order). Gap 2026-01-28 (id 270) to 2026-02-26 (id 271) = Olympic break.
- Playoff volume: 13 games, 623 of 5,278 5v5 min (12%), 727 of 7,438 `shot` rows (9.8%).
- OT/shootout: playoff games have up to 6 periods (game 341: 1-6 plus labels OT1/OT2).

## 2. Completeness of on-ice lists

Format: `on_ice_home/away` = comma-separated player ids **including the goalie** (6 ids at 5v5), skaters_home/away = **skaters only**.

| Check | Result |
|---|---|
| Events with on-ice list | 23,485 / 23,838 (98.5%). Null: shootout 112 (all), blocked_shot 68, goalie_change 43, shot 42, faceoff 35, goal 21, penalty 17, hit 15 |
| Shots/goals/blocks with list | shot 99.4%, blocked 97.9%, goal 96.5% (131 events null, concentrated in playoff games 341 (19), 351 (9), 344 (6), plus 247 (5), 350 (5)) |
| `strength_state_valid` True | 22,263 (93.4%), False 1,230 (5.2%), None 345 (1.4%). Shots 94.9%, blocked 95.0%, goals 86.2% |
| `strength_state` orientation | **Home-vs-away (5v4 = home 5, away 4), not shooter-relative** |
| strength_state vs list-derived skater counts | 99.9% agree: strength_state is derived from the same lists (not independent validation) |
| Invalid states | 421 shot/block rows with impossible counts (5v2 80, 2v5 49, 5v7 24, 7v5 23, 8v5 17, 5v1 17, 5v0 16 ...): list size is wrong in these rows; none of them is truly 5v5 |
| Lists sized correctly at 5v5 | 99.9% of 5v5-labelled shots have exactly 5 skaters per side |
| Goalie present in list | defending list has no goalie in 9.5% of shots faced at 5 defending skaters (763 of 7,993); varies by game (median 0%, up to 91%) |
| `skaters_home` junk | 1.3% of rows have skaters_home 0 or >6 (max 11) |
| Unknown ids | none; every id in the lists is in shifts or box. No duplicates in a list, no player on both sides |

Shot/event counts (play_by_play): shot 7,438 (incl. 601 goals; `goal==True`), goal 602 rows (**goals are double-recorded: each goal has a `shot` row AND a `goal` row**), blocked_shot 3,197, shootout 112. There is **no xG column**; available shot features: shot_distance, shot_angle, x/y coords, shot_quality (4-level text), scoring_chance, event_type (shot type), empty_net, power_play/short_handed. Shot features are complete (shot_distance null 0% on shots/blocks, 0.2% on goals).

## 3. On-ice lists vs shifts

**Shifts clock convention**: `start_s`/`end_s` are **clock-remaining seconds** within the period (`start_s > end_s` in 99.5% of rows; 485 rows (0.46%) have `end_s` NaN). A shift covers elapsed (1200-start_s, 1200-end_s]. Tested: shooter is in his own shift at event clock for 92% (inclusive) vs 32% when treated as elapsed. Shift TOI sums match box `time_on_ice` to within 0.05 min (4,997 player-games; 44 box rows lack shifts). Shifts include goalies (21 goalie ids) and are not in box.

Reconstructing skater sets at each shot/goal/block time (11,106 events; goalies removed):
- **Provided lists = shifts with rule `end_s < clock <= start_s`**: exact match 94.9% (5v5: 96.2%; goals 95.7%; blocks 96.5%; shots 94.2%). Mismatches (563) are almost all large (symdiff >= 6: list garbage) rather than 1-player differences. Period 1 matches 91.7%, P2/P3 98%, OT (period 4) 28.6%. 7 games below 50%; worst games 313 (59%), 235 (61%), 240 (63%), 265, 210.
- OT: 57 blocked_shot rows carry `period_of_game` = "OT1"/"OT2" (with per-period clock) and are not in the numeric-OT stream; `sec_from_start` resets to 0 in playoff period 6.

**Boundary bug (important)**: the rule above drops a player whose shift ends exactly at the event second. The shift feed ends shifts at the goal clock, so at goals the lists are wrong:
- Credited goal on-ice players (plus_player/minus_player columns, 5,784 slots on 577 goals): only **46% of scoring-team plus players and 37% of conceding-team minus players are in the list**; 52% of these players have `end_s == goal clock`. The scorer is in the list on only ~41% of goals (shooter-in-own-list for ordinary shots 79.8%).
- Shift rule `end_s <= clock < start_s` recovers 89% of plus/minus slots (inclusive both ends 92%, but with extra incoming players); for shots it lifts shooter-in-own-shift from 79.4% to 88.7%, blocks 88.5% to 89.4%. Residual ~10% is event/shift clock jitter (+-3s slack gives 93-96%).
- Implication: **for the quantity RAPM/xGA cares most about (goals against), the shipped lists put the wrong defenders on the ice roughly half the time.** For shots against the error is milder (~20% of shooters not in the list). Recommendation: rebuild stints from `shifts_2026` with explicit boundary rules and validate against plus/minus columns.
- Goal rows carry `plus_player_*` (scoring team) and `minus_player_*` (conceding team): 5 ids in 556/602 rows (4 in 30, 3 in 15 etc.), minus 5 ids in 464 rows. These are a clean independent source for on-ice at goals.
- `goalie_id` is **null on all 602 goal rows** (and 178 of 561 goalie_change rows); null on 0 shot/block rows.

## 4. player_id alignment with the model data

- sdv ids are consistent across pbp, shifts and box (box covers 187 skaters; 133 games). `war_2526.csv` ids and names match sdv ids (all 182 war rows' ids exist in box).
- **war_2526.csv covers all 133 games (regular + playoffs)**, not 120: `gp` up to 39 (box max 39), summed TOI 77,389 min vs box all-133 = 77,804 vs box regular-only = 69,381. 149 of 182 war rows are within 0.5 min TOI and the same `gp` as box all-games; vs regular-only box only 80 match. Playoffs are therefore blended into the current WAR.
- 13 unique players differ by more than 0.5 min vs box (war lower by 0.5 to 40 min: Clair DeGeorge 98 vs 139, Samoskevich 153 vs 171, Hannah Brandt 392 vs 405, Peyton Hemp 536 vs 547). 15 box skaters are absent from war (TOI 0 to 47 min, 1-9 GP).
- 10 player ids appear on 2 rows in war (traded players, one row per team, e.g. Eldridge 36 SEA/BOS, Krizova 104 MIN/NY, Cava 115 OTT/VAN). player_id is not unique in war_2526.csv; the RAPM (one coefficient per player) must be joined/merged accordingly.
- **`raw/game_data_2526.csv` is stale**: 61 games only (ids 210-270, all SeasonStage "Regular", through 2026-01-28); 2,306 non-goalie player-games match box exactly one-to-one on (game, player) with TOI within 0.05 min for 2,156, and 150 differ by >0.05 (up to 12.7 min). Games 271-351 (72) missing.
- **The on-ice defensive columns in `game_data_2526.csv` are all zero**: EV_SA, EV_xGA, EV_GA, PP_GA, SH_GA sum to 0 across 2,306 rows (EV_Shots sum 3,490, EV_ixG 274.7 are populated). Anything depending on EV_xGA from that file is silently empty; this is the reason the model fell back to plus/minus residual.
- Team labels in pbp are mojibake ("Montr�al Victoire") whenever Montreal plays; use team_id (1 BOS, 2 MIN, 3 MTL, 4 NY, 5 OTT, 6 TOR, 8 SEA, 9 VAN per matches in box/pbp).

## 5. Rates, sparsity, estimability (5v5)

Shot counts by strength (pbp `strength_state`, shot rows incl. goals):

| Strength | SOG (shot rows) | Goals | Blocked | Goal % of SOG |
|---|---|---|---|---|
| 5v5 | 4,817 | 391 | 2,065 | 8.1% |
| 5v4 / 4v5 | 1,451 | 70 | 626 | 4.8% |
| 4v4 | 131 | 8 | 73 | 6.1% |
| 3v3 | 90 | 5 | 2 | 5.6% |
| Other (5v3, 6v5, 5v6 ...) | 902 | 106 | 363 | 11.8% |
| null | 47 | 22 | 68 | - |

(Goal column = `goal` rows; these goals are already inside the SOG column as `shot` rows with goal=True, so do not add the two.)

- Overall: SOG goal% = 601/7,438 = 8.1%. 5v5 SOG/60 (both teams) = 4,817 / (5,278 min / 60) = **54.8, i.e. 27.4 per team**; 5v5 goals 4.4/60 combined (2.2 per team); blocked 23.5/60 combined.
- 5v5 clock time from shifts (exactly 5 skaters per side): **5,278 min = 39.7 min/game (regular 4,655 min, 38.8/game)**. This is a lower bound on true 5v5 time because seconds with 6-7 skaters during changes are dropped. Median stint 5s, mean 9.0s; 35,204 stints, 32,193 unique (home,away) lineup pairs, 71% under 10s.
- 5v5 minutes per skater (186 players with any): mean 284, median 302, p25 173, p75 377, max 611; D median 362, F median 290. Skaters with >= 50 / 100 / 200 / 300 min: 171 / 155 / 133 / 95 (>= 100 min: 103 F + 52 D). Regular season only: >= 100 min 153, >= 200 min 127, >= 50 min 169; median 277.
- Expected volume per skater at median 5v5 TOI (302 min): ~138 shots against, ~11 goals against. At 100 min: ~46 SOG against, ~3.7 goals against. Goals against per skater is far too thin for a goals-only RAPM; shot-based (xGA) targets are required, as in the plan.
- **Collinearity**: stint-by-player design (home +1/away -1) is 35,204 x 186 with rank 185 (one dependency: sum constraint); offense/defense design 70,408 x 372 has rank 371 and smallest singular value 0, condition number essentially infinite -> ridge (or an explicit constraint) is mandatory. Among teammate pairs with both >= 50 min (1,923 pairs): 81 share >50% of the lesser player's TOI, 25 share >70%, 9 share >80%; F-F median overlap 9%, D-D median 9% (206 pairs, 20 over 50%, 9 over 70%). Top pairs: Jenner-Leslie 430 min (90% of the lesser's TOI), Flaherty-Tabin 459 min (86%), Poulin-Stacey 291 (84%), Poulin-Roque 290 (84%), Kadirova-Mrazova 295 (81%), Boreen-Rosenthal 218 (83%), Habisch-Langseth 59 (84%), Krizova-Petrie 75 (84%). These pairs will be near-inseparable; expect ridge shrinkage to split credit evenly.

## 6. Goalies

- goalie_id on shot rows = the **defending** goalie (never in the shooting team's list). Present on 100% of shot/block rows; 0% on goal rows. It is in the defending on-ice list for 90.2% of shots (the rest: list lacks the goalie).
- 21 goalie ids; 1 goalie appears for 2 teams. Goalies per team: 2 (NY), 3 for most teams, 4 (OTT).

| def team | goalies | shots against | top goalie share |
|---|---|---|---|
| 1 BOS | 3 | 878 | 86% |
| 2 MIN | 3 | 969 | 58% |
| 3 MTL | 3 | 1,006 | 86% |
| 4 NY | 2 | 784 | 89% |
| 5 OTT | 4 | 1,163 | 94% |
| 6 TOR | 3 | 850 | 76% |
| 8 SEA | 3 | 932 | 54% |
| 9 VAN | 3 | 856 | 64% |

- **A single goalie covers a team 54-94% of the time** (4 of 8 teams >= 86%; SEA, MIN, VAN are true timeshares: 54% / 58% / 64%). Only 11 of 266 team-games (4.1%) saw a mid-game goalie change by shot attribution, but there are 561 `goalie_change` rows (4.2 per game, many are start-of-period or both-goalies-in-net bookkeeping).
- Because the starting goalie is confounded with team defence, team-by-goalie adjustment needs goalie fixed effects or goalie quality in the xG model; with 2-4 goalies per team and one dominant goalie in half the teams, a goalie term will be nearly collinear with team in those teams.

---

## Data quality issues

### CRITICAL
- **On-ice lists drop players whose shift ends at the event second** -> at goals only 46% (plus) / 37% (minus) of credited on-ice players are in the list; shooters are in own list for 79.8% of shots. Fix: rebuild stints/event on-ice from shifts with explicit boundary rules (end <= t < start for goals, validate with plus/minus columns).
- **Model files are out of sync**: `war_2526.csv` includes playoffs (all 133 games), `game_data_2526.csv` stops at game 270 (61 games), and its EV_SA/EV_xGA/EV_GA/PP_GA/SH_GA are all 0. Decide season scope (120 regular vs 133) and rebuild from sdv.
- **Goals double-recorded** (`shot` row with goal=True and a `goal` row): counting both inflates shots/xGA by 601 events (8% of SOG). Dedupe to one row per goal.
- **No xG in the data**; xGA requires fitting an xG model on shot_distance/angle/shot type/quality (shot_quality is a post-hoc label incl. "goal" categories: leaks the outcome for goals).

### MODERATE
- strength_state is derived from the same flawed lists (99.9% agreement); 5.2% of events (421 shots/blocks) have impossible strengths (5v2, 5v7 ...). Filter to `strength_state_valid==True` and list-size == 5/5 for 5v5.
- 131 shot/goal/block events have no list (playoff games 341, 351, 344 worst); 112 shootout rows have none by design.
- Goalie missing from the defending list in 9.5% of shots at 5 skaters; goalie_id null on all goals (infer from the preceding shot or goalie_change).
- 485 shift rows with NaN `end_s`; 44 box player-games with no shifts.
- Overtime/playoff periods: 57 blocked shots with period "OT1"/"OT2", `sec_from_start` reset in period 6; period-4 list-vs-shift match only 28.6% (154 events). Exclude OT or re-key.
- Collinear linemates: 25 teammate pairs share > 70% of their 5v5 time; offense/defense RAPM design is rank-deficient (371/372).
- Goalie share: 54-94% single goalie per team; goalie effect near-collinear with team in several teams.
- Traded players appear twice in war_2526.csv (player_id not unique).
- 13 unique players have war TOI 0.5 to 40 min below box.

### MINOR
- Team names mojibake in pbp (Montr�al) and in war names (Gigu�re, Kri�ova); join on ids, not names.
- `period_of_game` mixed str ('1','OT1', None); `game_date` not monotone with game_id (282, 250, 249).
- `skaters_home` junk in 1.3% rows (0 or up to 11).
- Box `position` includes 'D' (29) and 'F' (17) generic labels besides LD/RD/LW/RW/C.
- `time_on_ice` in box rounded to 0.1 min.

---

## Recommended Plan Tasks

1. Fix scope: pick regular season (game_id <= 329, 120 games) as the training set; keep playoffs (13 games) held out or tagged. Re-derive war_2526.csv scope, because it currently blends playoffs and `game_data_2526.csv` is only 61 games.
2. Rebuild on-ice membership from `shifts_2026.parquet` (clock-remaining convention: interval (1200-start_s, 1200-end_s]; goalies identified via goalie_id/shift ids not in box), with boundary rule validated against plus/minus columns on goals and shooter-in-own-shift on shots; use box TOI as a per-player-game check (tolerance 0.05 min).
3. Dedupe goals (remove `goal` rows or `shot` rows with goal=True) before counting; restrict to period 1-3 initially and handle OT and shootouts separately; drop events with null lists and strength_state_valid != True.
4. Fit an xG model from the pbp shot features (distance, angle, type, coordinates, strength, empty net); do not use shot_quality (outcome leakage on goals). Targets for RAPM: xG against per stint (5v5), with goals against as a validation only (too sparse: ~11 GA per skater at median TOI).
5. Run ridge (offense and defense coefficients per skater) on stints with >= 1 s at exactly 5v5; weights by stint duration; shrink with CV by game/time block (never random across games of the same date); require >= 100 5v5 min (155 skaters, 103 F + 52 D) for reporting and fold the rest into a replacement level. Add position (F/D) and zone-start/score-state covariates if available.
6. Handle collinearity explicitly: report ridge lambda, check top linemate pairs (Jenner-Leslie 90%, Flaherty-Tabin 86%, Poulin-Stacey/Roque 84%), consider pair-split diagnostics or hierarchical shrinkage towards position/team means.
7. Goalie adjustment: include defending-goalie fixed effect (or goalie GSAx) in the xGA model; 21 goalies, 2-4 per team; treat goalie_id as the defending goalie and infer goal-row goalie from the same-timestamp shot row.
8. Reconcile player keys: join on player_id, collapse traded players (10 duplicate ids) before RAPM-to-WAR merge, and keep playoff flag via game_id > 329.
9. Replace the all-zero EV_SA/EV_xGA/EV_GA columns in the model data pipeline with outputs from the rebuilt on-ice data and add a test asserting their sum is nonzero and that 5v5 SOG/60 is about 27 per team.
