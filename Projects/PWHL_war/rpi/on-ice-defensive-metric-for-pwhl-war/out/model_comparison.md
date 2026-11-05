# Model Comparison - on-ice defensive metric (PWHL WAR)

**Date**: 2026-11-05
**Verdict (FINAL, five candidates): KEEP BASELINE** (pre-set rule, spec s6 item 9, met by no candidate). Ceiling reached; the FINAL PASS section at the end supersedes the Phase 7 recommendation in section (5).
**Naive reference** (team-average rate x minutes): split-half r 0.564, season-to-season r 0.324.

Cohort: >=100 five-on-five minutes; 383 player-seasons, 200 consecutive-season pairs (118 distinct players). Same cohort in all four files (verified).

## Results (Pearson; Spearman recomputed from the CSVs in brackets)

| Candidate | Split-half r | Season-to-season r | Beats baseline on both? |
|---|---|---|---|
| baseline | 0.579 [0.440] | 0.201 [0.215] | - |
| toi_share_ownxg | 0.515 [0.453] | 0.340 [0.413] | no (split-half lower) |
| onice_xga | 0.579 [0.533] | 0.365 [0.437] | no (split-half difference 0.000) |
| naive | 0.564 [0.470] | 0.324 [0.408] | reference |

Bootstrap differences (table, 2000 resamples): onice_xga - baseline split-half -0.000 [-0.142, +0.171]; season-to-season +0.164 [+0.030, +0.307]. Rule requires both to exceed and exclude 0: split-half fails outright. toi_share_ownxg - baseline: -0.065 [-0.206, +0.109] / +0.139 [-0.004, +0.277], also fails.

## (2) Team-level versus individual
Naive uses no individual signal yet reaches 0.564 / 0.324, i.e. 97% of baseline's split-half and 90% of onice_xga's season-to-season r. Share of each candidate's r that naive reproduces: baseline 97% / 161% (baseline is below naive season-to-season), toi_share_ownxg 109% / 95%, onice_xga 97% / 89%. Individual increment over naive for onice_xga: +0.015 split-half [-0.048, +0.085], +0.041 season-to-season [-0.034, +0.115]. Both include 0 (Pearson). Spearman: split-half +0.063 [+0.004, +0.131] (barely excludes 0), season-to-season +0.030 [-0.051, +0.112]. Conclusion: repeatability is overwhelmingly team-level (team defence and minutes-share structure); a real individual signal is not demonstrated. Candidate correlations: toi_share_ownxg vs naive r = 0.97 (they are nearly the same measure); onice_xga vs naive 0.59.

## (3) xG source versus attribution
- xG source (toi_share_ownxg - baseline): split-half -0.065 (ns), season-to-season +0.139 (table CI just includes 0; my 1000-resample rerun gave [+0.015, +0.265]). Baseline's season-to-season r of 0.20 is the outlier: the API xG (own vs API game-total r = 0.68, own total 14% lower) is the likely weak link across seasons. The xG source accounts for most of the season-to-season gain (+0.139 of +0.164).
- Attribution (onice_xga - toi_share_ownxg): split-half +0.064 [+0.010, +0.124] (clear but small), season-to-season +0.025 [-0.047, +0.100] (none). On-ice attribution adds a little within-season signal and nothing detectable across seasons.

## (4) Validity checks
- Outliers: baseline split-half falls 0.579 -> 0.492 dropping its single minimum (player 310, 2025-26, d_full -0.644, 121 min), and 0.432 after trimming the outer 1% each tail; Spearman is 0.440. Baseline's split-half is thus partly outlier-driven; trimmed, onice_xga leads (0.526 vs 0.432) and the rule's split-half condition might then pass, but the rule was fixed on Pearson untrimmed, so it is not applied. Other candidates are stable (<=0.03 shift on dropping the minimum). Season-to-season trimmed: baseline 0.178, toi_share_ownxg 0.338, onice_xga 0.320, naive 0.345.
- Rank check: Spearman agrees on ordering for season-to-season (onice_xga - baseline +0.222 [+0.077, +0.365]) and shows split-half +0.093 [-0.008, +0.194], ie. still not significant for the rule.
- Non-independence: the two statistics use the same player-seasons; 87 of 230 players appear in all three seasons, so the 200 pairs share seasons across two transitions (118 distinct players). The bootstrap resamples rows, not players, so intervals are somewhat too narrow; the one significant rule-relevant interval (season-to-season) is therefore weaker than shown. Alternating-game halves also share team, schedule and opponent quality, which inflates split-half and explains why naive scores 0.56.
- Traded players: the CSVs have no team column, so not checked. Players changing teams between seasons would lower season-to-season r for team-dominated measures; this is untested.
- Cohort mismatches: none (383/200 in every file, identical keys).

## (5) Recommended single Phase 7 iteration
Apply empirical-Bayes shrinkage to onice_xga: shrink each player's on-ice xGA/60 toward his or her team's rate with weight minutes/(minutes + k), k estimated from the split-half variance (no tuning on the evaluation target beyond k). Rationale: onice_xga is the only candidate that beats naive on both statistics (point estimates), its noise at low minutes is the main remaining defect (minimum -1.75 at 1.5 min before the cohort cut), and shrinkage leaves attribution logic unchanged. Expected improvement: modest, +0.02 to +0.05 on season-to-season r (to ~0.39-0.41) and little change in split-half; do not expect the individual increment over naive to become significant. Success test: beat baseline per the unchanged rule AND exceed naive with interval excluding 0. If not met, ceiling reached.

## Honest bottom line
Keep the baseline in production for now; record that on-ice attribution with own xG is a defensible, not-worse alternative (equal split-half, +0.16 season-to-season) but the pre-set rule is not met because split-half does not improve. Defensive repeatability in the PWHL at these sample sizes is mostly team effect.


---

# FINAL PASS - fifth candidate (rapm) added 2026-11-05

Cohort unchanged: 383 player-seasons, 200 pairs, identical keys in all five files (verified).

## Rule applied literally to rapm
| Condition | rapm - baseline | Met? |
|---|---|---|
| split-half r exceeds baseline | 0.562 vs 0.579 (-0.017, [-0.158, +0.157]) | no (lower, interval spans 0) |
| season-to-season r exceeds baseline | 0.325 vs 0.201 (+0.124, [-0.011, +0.265]) | point estimate yes, interval includes 0 |

Rapm fails the rule on both measures. Overall: none of the five candidates replaces the baseline (onice_xga fails split-half, diff -0.000; toi_share_ownxg fails split-half and its season-to-season interval includes 0; naive is the reference; rapm fails both). **KEEP BASELINE.**

## Does rapm add anything?
- Over onice_xga: no, it is slightly worse. Split-half -0.017 [-0.025, -0.007]; season-to-season -0.040 [-0.076, -0.007]. Both intervals exclude 0 on the negative side. Pearson 0.988 (Spearman 0.981) with onice_xga on d_full, so rapm is onice_xga with a small loss. The intervals are narrow because the two series are nearly identical; the difference is real but tiny.
- Over naive: no. Split-half -0.002 [-0.069, +0.073]; season-to-season +0.001 [-0.097, +0.101]. Rapm correlates 0.844 with naive on d_full (onice_xga: 0.59), i.e. the shrinkage pulled it toward a team-level measure.
- Reactivation conditions (12+ teams, 3+ seasons, real on-ice data) were met and the earlier archive finding (no repeatability beyond team effect) is reproduced.

## Spearman (recomputed from the CSVs; Pearson in brackets)
| Candidate | Split-half | Season-to-season |
|---|---|---|
| baseline | 0.440 [0.579] | 0.215 [0.201] |
| toi_share_ownxg | 0.453 [0.515] | 0.413 [0.340] |
| onice_xga | 0.533 [0.579] | 0.437 [0.365] |
| naive | 0.470 [0.564] | 0.408 [0.324] |
| rapm | 0.517 [0.562] | 0.368 [0.325] |

Spearman bootstrap differences for rapm (over players, 1000 resamples, seed 42):
- rapm - onice_xga: split-half [-0.028, -0.004], season-to-season [-0.114, -0.029] (clearly worse, same direction as Pearson).
- rapm - naive: split-half [-0.025, +0.116], season-to-season [-0.158, +0.079] (no difference).
- rapm - baseline: split-half [-0.029, +0.181], season-to-season [-0.018, +0.319] (no clear difference; rule also fails on rank).

d_full correlations of rapm with others, Pearson / Spearman: onice_xga 0.988 / 0.981, toi_share_ownxg 0.860 / 0.830, naive 0.844 / 0.824, baseline 0.410 / 0.463.
Pearson and Spearman agree on ordering of season-to-season: onice_xga > naive ~ toi_share_ownxg > rapm (Spearman 0.368 vs 0.408-0.437) >> baseline. Rapm's rank season-to-season (0.368) is a little lower relative to naive than its Pearson, still within noise.

## Phase 7 shrinkage: not supported; ceiling reached
Rapm is in effect a regularised, team-shrunk onice_xga, and the extra shrinkage cost -0.017 split-half and -0.040 season-to-season (both clear). That is direct evidence that more shrinkage toward team level does not help here, and drives the individual increment over naive to about 0. The earlier forecast (+0.02 to +0.05 on season-to-season) is withdrawn. Realistic expectation for empirical-Bayes shrinkage of onice_xga: 0 to +0.02, possibly negative (full shrinkage = naive = 0.324 vs onice_xga 0.365, so the headroom is small). Verdict: ceiling reached for the on-ice approach with current data; no further single-change iteration is justified.

Worth trying instead (all low expected benefit):
1. Record onice_xga as a documented, not-worse alternative (equal split-half, +0.16 season-to-season) without replacing the baseline. Changing the rule is a deliberate recorded decision, not a retroactive application.
2. Add a team column and test (a) traded players and (b) within-team-season demeaned d_full. Diagnostic only; expected to show a weak individual signal (r about 0.1-0.2), not a fix.
3. Re-run when a fourth season is available.

## Caveats that matter
- Shared seasons: 87 of 230 players appear in all three seasons; the 200 pairs come from 118 players and both transitions use 2024-25. Pooled season-to-season r double-counts overlapping data and pools transitions with very different values (0.35-0.47 vs 0.12-0.30). The pairs are not independent, so even the player-level bootstrap is only approximately right.
- Team-level inflation of split-half: alternating-game halves share team, schedule and opponent quality. Naive, with no individual information, scores 0.564, so split-half is mostly team effect. Differences between candidates (at most 0.07) sit inside that noise, and a not-worse measure cannot realistically pass the rule's split-half condition.
- Baseline outlier sensitivity: one point (player 310, 2025-26) moves baseline split-half 0.579 -> 0.492; trimmed 1% per tail it is 0.432 and its Spearman is 0.440. Against a trimmed baseline onice_xga would lead (0.526 vs 0.432), so the rule's split-half verdict hinges on one outlier. The rule was fixed on untrimmed Pearson and is applied as written; this is a rule-design weakness to record, not a finding in favour of onice_xga.
- Rapm-vs-onice_xga "significance" reflects near-identical series (r 0.988): a small consistent loss, not a large one.
- Traded players untested (no team column). The API-vs-own xG gap (game-total r 0.68) remains a confound for the baseline's weak season-to-season r.
