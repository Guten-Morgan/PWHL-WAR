# Decision: on-ice defensive metric for PWHL WAR

Date: 2026-11-05. Numbers are copied from `comparison_table.md` and `model_comparison.md`. No new analysis.

## Rule (spec section 6.9, fixed before the results)

A candidate replaces the baseline only if BOTH its split-half r and its season-to-season r exceed the baseline's, AND both bootstrap intervals of the difference (players resampled, 2000 draws, seed 42) exclude zero.

## Applying it

| candidate | split-half diff vs baseline [95% interval] | season-to-season diff vs baseline [95% interval] | Replaces baseline? |
|---|---|---|---|
| toi_share_ownxg (control) | -0.065 [-0.206, +0.109] | +0.139 [-0.004, +0.277] | No (both intervals include 0; split-half lower) |
| onice_xga | -0.000 [-0.142, +0.171] | +0.164 [+0.030, +0.307] | No (split-half not higher, interval includes 0) |
| rapm | -0.017 [-0.158, +0.157] | +0.124 [-0.011, +0.265] | No (split-half lower, both intervals include 0) |
| naive (reference) | not a candidate | not a candidate | n/a |

## Decision

**KEEP BASELINE.** Production defence stays as is (team even-strength xGA split by TOI share, `defense_weight = -0.10`). No change to `regen_2526.py`, `export_js.py` or the site workflow. The "if adopted" tasks in Phase 8 are skipped.

## What the evidence does say (not used to override the rule)

- Most repeatability is team-level: the naive team-rate reference reaches about 97% of the baseline's split-half r.
- On-ice xGA's season-to-season gain over baseline (+0.164) is mostly from the xG source (+0.139 for the control that only swaps xG), not from true on-ice attribution.
- Attribution adds a small split-half gain over the control (+0.064 [+0.010, +0.124]) and nothing detectable season to season.
- RAPM is slightly worse than on-ice xGA and indistinguishable from naive.
- Spearman (outlier-robust): onice_xga is best on both measures (0.533, 0.437). The rule was fixed on Pearson, so it stands.

## Caveats on this decision

- The baseline's split-half depends on one outlier (player 310, 2025-26); the rule's Pearson basis is sensitive to it.
- Season-to-season pairs overlap (118 players supply 200 pairs; 2024-25 is in both transitions).
- Split-half halves share team and schedule effects.
- Traded players untested; own xG vs API xG game-total agreement is r = 0.68.

## Follow-ups (not part of this decision)

1. Published 2025-26 data includes 13 playoff games; regular-season scope fix (`pwhl_war/scope.py` exists).
2. Test swapping the xG source in the existing model, including the offensive side.
3. Fix `run_xga_war.py` and `run_combined_war.py`, which do not run as written.
4. Reconcile goals-per-win values (4.6667 in `regen_2526.py`, 4.508 in `constants.py`, 6.0 fallback).
