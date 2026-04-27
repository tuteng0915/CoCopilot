# Remask Locator/Rewriter Case-Study Analysis

> Historical note (2026-04-20): this HumanEval report uses pre-pkgv2 EvalPlus packaging and is superseded by `docs/analysis/seed_coder_instruct_locator_rewriter_case_study_pkgv2.md`. Do not cite the HumanEval transition counts here as current conclusions.

This report is generated mechanically. Category labels are candidate labels for case-study selection, not final semantic judgments.

## Inputs

- Dataset: `humaneval`
- Baseline JSONL: `outputs/base_tuteng/seed-coder-instruct_humaneval.jsonl`
- Baseline eval: `outputs/base_tuteng/seed-coder-instruct_humaneval_eval_results.json`
- Run `dream_t0.9`: `outputs/base_tuteng/seed-coder-instruct_dream_remask_humaneval_t0.9.jsonl` / `outputs/base_tuteng/seed-coder-instruct_dream_remask_humaneval_t0.9_eval_results.json`
- Run `dream_maskr0.01`: `outputs/base_tuteng/seed-coder-instruct_dream_remask_humaneval_maskr0.01.jsonl` / `outputs/base_tuteng/seed-coder-instruct_dream_remask_humaneval_maskr0.01_eval_results.json`
- Run `dream_gate0.012_offpolicy`: `outputs/base_tuteng/seed-coder-instruct_dream_remask_humaneval_maskr0.01_gate0.012.jsonl` / `outputs/base_tuteng/seed-coder-instruct_dream_remask_humaneval_maskr0.01_gate0.012_eval_results.json`
- Run `dream_gate0.012_fresh_fixed`: `outputs/base_tuteng/seed-coder-instruct_dream_remask_humaneval_maskr0.01_gate0.012_fresh_fixed.jsonl` / `outputs/base_tuteng/seed-coder-instruct_dream_remask_humaneval_maskr0.01_gate0.012_fresh_fixed_eval_results.json`

## Run-Level Outcomes

| run | n | plus% | delta plus pp | wrong->correct | correct->wrong | same pass | same fail | skipped | kept | mean mask frac | median edit ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dream_gate0.012_fresh_fixed | 164 | 70.1 | +0.0 | 0 | 0 | 115 | 49 | 130 | 34 | 0.0223 | 0.0000 |
| dream_gate0.012_offpolicy | 164 | 72.6 | +2.4 | 7 | 3 | 112 | 42 | 130 | 34 | 0.0223 | 0.0000 |
| dream_maskr0.01 | 164 | 65.2 | -4.9 | 10 | 18 | 97 | 39 | 0 | 0 | 0.0223 | 0.0000 |
| dream_t0.9 | 164 | 65.2 | -4.9 | 10 | 18 | 97 | 39 | 0 | 0 |  | 0.0000 |

## Candidate Category Counts

| category | n tasks | examples |
| --- | --- | --- |
| baseline_correct_no_observed_harm | 114 | HumanEval/1, HumanEval/20, HumanEval/6, HumanEval/0, HumanEval/115, HumanEval/12, HumanEval/14, HumanEval/17 |
| baseline_wrong_rewrite_ineffective | 48 | HumanEval/107, HumanEval/119, HumanEval/143, HumanEval/146, HumanEval/81, HumanEval/94, HumanEval/162, HumanEval/74 |
| packaging_or_eval_artifact_candidate | 28 | HumanEval/1, HumanEval/107, HumanEval/119, HumanEval/143, HumanEval/146, HumanEval/20, HumanEval/6, HumanEval/75 |
| gate_prevented_packaging_harm_candidate | 15 | HumanEval/0, HumanEval/115, HumanEval/12, HumanEval/14, HumanEval/17, HumanEval/21, HumanEval/25, HumanEval/28 |
| offpolicy_packaging_fix_not_reproduced_candidate | 7 | HumanEval/107, HumanEval/119, HumanEval/143, HumanEval/146, HumanEval/75, HumanEval/81, HumanEval/94 |
| gate_overconservative_packaging_candidate | 3 | HumanEval/162, HumanEval/74, HumanEval/82 |
| low_disagreement_packaging_hurt_candidate | 3 | HumanEval/1, HumanEval/20, HumanEval/6 |
| gate_prevented_harm_candidate | 1 | HumanEval/4 |
| rewrite_stochastic_candidate | 1 | HumanEval/75 |

## Mask-Fraction Bins

| run | mask_fraction bin | n | wrong->correct | correct->wrong | same pass | same fail | net |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dream_gate0.012_fresh_fixed | [0.000, 0.008) | 6 | 0 | 0 | 2 | 4 | +0 |
| dream_gate0.012_fresh_fixed | [0.008, 0.010) | 17 | 0 | 0 | 9 | 8 | +0 |
| dream_gate0.012_fresh_fixed | [0.010, 0.012) | 11 | 0 | 0 | 6 | 5 | +0 |
| dream_gate0.012_fresh_fixed | [0.012, 0.015) | 29 | 0 | 0 | 22 | 7 | +0 |
| dream_gate0.012_fresh_fixed | [0.015, 0.020) | 34 | 0 | 0 | 26 | 8 | +0 |
| dream_gate0.012_fresh_fixed | [0.020, 0.030) | 30 | 0 | 0 | 22 | 8 | +0 |
| dream_gate0.012_fresh_fixed | [0.030, 0.050) | 27 | 0 | 0 | 20 | 7 | +0 |
| dream_gate0.012_fresh_fixed | [0.050, inf) | 10 | 0 | 0 | 8 | 2 | +0 |
| dream_gate0.012_offpolicy | [0.000, 0.008) | 6 | 1 | 0 | 2 | 3 | +1 |
| dream_gate0.012_offpolicy | [0.008, 0.010) | 17 | 2 | 2 | 7 | 6 | +0 |
| dream_gate0.012_offpolicy | [0.010, 0.012) | 11 | 4 | 1 | 5 | 1 | +3 |
| dream_gate0.012_offpolicy | [0.012, 0.015) | 29 | 0 | 0 | 22 | 7 | +0 |
| dream_gate0.012_offpolicy | [0.015, 0.020) | 34 | 0 | 0 | 26 | 8 | +0 |
| dream_gate0.012_offpolicy | [0.020, 0.030) | 30 | 0 | 0 | 22 | 8 | +0 |
| dream_gate0.012_offpolicy | [0.030, 0.050) | 27 | 0 | 0 | 20 | 7 | +0 |
| dream_gate0.012_offpolicy | [0.050, inf) | 10 | 0 | 0 | 8 | 2 | +0 |
| dream_maskr0.01 | [0.000, 0.008) | 6 | 1 | 0 | 2 | 3 | +1 |
| dream_maskr0.01 | [0.008, 0.010) | 17 | 2 | 2 | 7 | 6 | +0 |
| dream_maskr0.01 | [0.010, 0.012) | 11 | 4 | 1 | 5 | 1 | +3 |
| dream_maskr0.01 | [0.012, 0.015) | 29 | 0 | 1 | 21 | 7 | -1 |
| dream_maskr0.01 | [0.015, 0.020) | 34 | 2 | 5 | 21 | 6 | -3 |
| dream_maskr0.01 | [0.020, 0.030) | 30 | 0 | 4 | 18 | 8 | -4 |
| dream_maskr0.01 | [0.030, 0.050) | 27 | 1 | 4 | 16 | 6 | -3 |
| dream_maskr0.01 | [0.050, inf) | 10 | 0 | 1 | 7 | 2 | -1 |

## Edit Metrics By Transition

| run | transition | n | median char edit ratio | p90 char edit ratio | median changed lines | changed signature | changed imports | refined parse fails |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dream_gate0.012_fresh_fixed | same_fail | 49 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | 0 |
| dream_gate0.012_fresh_fixed | same_pass | 115 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | 0 |
| dream_gate0.012_offpolicy | correct_to_wrong | 3 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | 0 |
| dream_gate0.012_offpolicy | same_fail | 42 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | 0 |
| dream_gate0.012_offpolicy | same_pass | 112 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | 0 |
| dream_gate0.012_offpolicy | wrong_to_correct | 7 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | 0 |
| dream_maskr0.01 | correct_to_wrong | 18 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | 0 |
| dream_maskr0.01 | same_fail | 39 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | 0 |
| dream_maskr0.01 | same_pass | 97 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | 0 |
| dream_maskr0.01 | wrong_to_correct | 10 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | 0 |
| dream_t0.9 | correct_to_wrong | 18 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | 0 |
| dream_t0.9 | same_fail | 39 | 0.0000 | 0.0022 | 0.00 | 0 | 0 | 0 |
| dream_t0.9 | same_pass | 97 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | 0 |
| dream_t0.9 | wrong_to_correct | 10 | 0.0000 | 0.0002 | 0.00 | 0 | 0 | 0 |

## Deterministic Case Candidates

| category | task | wrong->correct runs | correct->wrong runs | median mask frac | median edit ratio | evidence |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_correct_no_observed_harm | HumanEval/1 | 0 | 3 | 0.0093 | 0.0000 | dream_gate0.012_fresh_fixed:same_pass/kept; dream_gate0.012_offpolicy:correct_to_wrong/kept; dream_maskr0.01:correct_to_wrong/no_gate; dream_t0.9:correct_to_wrong/no_gate |
| baseline_correct_no_observed_harm | HumanEval/20 | 0 | 3 | 0.0098 | 0.0000 | dream_gate0.012_fresh_fixed:same_pass/kept; dream_gate0.012_offpolicy:correct_to_wrong/kept; dream_maskr0.01:correct_to_wrong/no_gate; dream_t0.9:correct_to_wrong/no_gate |
| baseline_correct_no_observed_harm | HumanEval/6 | 0 | 3 | 0.0108 | 0.0000 | dream_gate0.012_fresh_fixed:same_pass/kept; dream_gate0.012_offpolicy:correct_to_wrong/kept; dream_maskr0.01:correct_to_wrong/no_gate; dream_t0.9:correct_to_wrong/no_gate |
| baseline_correct_no_observed_harm | HumanEval/0 | 0 | 2 | 0.0175 | 0.0000 | dream_gate0.012_fresh_fixed:same_pass/skipped; dream_gate0.012_offpolicy:same_pass/skipped; dream_maskr0.01:correct_to_wrong/no_gate; dream_t0.9:correct_to_wrong/no_gate |
| baseline_correct_no_observed_harm | HumanEval/115 | 0 | 2 | 0.0227 | 0.0000 | dream_gate0.012_fresh_fixed:same_pass/skipped; dream_gate0.012_offpolicy:same_pass/skipped; dream_maskr0.01:correct_to_wrong/no_gate; dream_t0.9:correct_to_wrong/no_gate |
| baseline_correct_no_observed_harm | HumanEval/12 | 0 | 2 | 0.0357 | 0.0000 | dream_gate0.012_fresh_fixed:same_pass/skipped; dream_gate0.012_offpolicy:same_pass/skipped; dream_maskr0.01:correct_to_wrong/no_gate; dream_t0.9:correct_to_wrong/no_gate |
| baseline_correct_no_observed_harm | HumanEval/14 | 0 | 2 | 0.0303 | 0.0000 | dream_gate0.012_fresh_fixed:same_pass/skipped; dream_gate0.012_offpolicy:same_pass/skipped; dream_maskr0.01:correct_to_wrong/no_gate; dream_t0.9:correct_to_wrong/no_gate |
| baseline_correct_no_observed_harm | HumanEval/17 | 0 | 2 | 0.0179 | 0.0000 | dream_gate0.012_fresh_fixed:same_pass/skipped; dream_gate0.012_offpolicy:same_pass/skipped; dream_maskr0.01:correct_to_wrong/no_gate; dream_t0.9:correct_to_wrong/no_gate |
| baseline_wrong_rewrite_ineffective | HumanEval/107 | 3 | 0 | 0.0104 | 0.0000 | dream_gate0.012_fresh_fixed:same_fail/kept; dream_gate0.012_offpolicy:wrong_to_correct/kept; dream_maskr0.01:wrong_to_correct/no_gate; dream_t0.9:wrong_to_correct/no_gate |
| baseline_wrong_rewrite_ineffective | HumanEval/119 | 3 | 0 | 0.0097 | 0.0000 | dream_gate0.012_fresh_fixed:same_fail/kept; dream_gate0.012_offpolicy:wrong_to_correct/kept; dream_maskr0.01:wrong_to_correct/no_gate; dream_t0.9:wrong_to_correct/no_gate |
| baseline_wrong_rewrite_ineffective | HumanEval/143 | 3 | 0 | 0.0115 | 0.0000 | dream_gate0.012_fresh_fixed:same_fail/kept; dream_gate0.012_offpolicy:wrong_to_correct/kept; dream_maskr0.01:wrong_to_correct/no_gate; dream_t0.9:wrong_to_correct/no_gate |
| baseline_wrong_rewrite_ineffective | HumanEval/146 | 3 | 0 | 0.0103 | 0.0000 | dream_gate0.012_fresh_fixed:same_fail/kept; dream_gate0.012_offpolicy:wrong_to_correct/kept; dream_maskr0.01:wrong_to_correct/no_gate; dream_t0.9:wrong_to_correct/no_gate |
| baseline_wrong_rewrite_ineffective | HumanEval/81 | 3 | 0 | 0.0088 | 0.0000 | dream_gate0.012_fresh_fixed:same_fail/kept; dream_gate0.012_offpolicy:wrong_to_correct/kept; dream_maskr0.01:wrong_to_correct/no_gate; dream_t0.9:wrong_to_correct/no_gate |
| baseline_wrong_rewrite_ineffective | HumanEval/94 | 3 | 0 | 0.0104 | 0.0000 | dream_gate0.012_fresh_fixed:same_fail/kept; dream_gate0.012_offpolicy:wrong_to_correct/kept; dream_maskr0.01:wrong_to_correct/no_gate; dream_t0.9:wrong_to_correct/no_gate |
| baseline_wrong_rewrite_ineffective | HumanEval/162 | 2 | 0 | 0.0303 | 0.0000 | dream_gate0.012_fresh_fixed:same_fail/skipped; dream_gate0.012_offpolicy:same_fail/skipped; dream_maskr0.01:wrong_to_correct/no_gate; dream_t0.9:wrong_to_correct/no_gate |
| baseline_wrong_rewrite_ineffective | HumanEval/74 | 2 | 0 | 0.0152 | 0.0000 | dream_gate0.012_fresh_fixed:same_fail/skipped; dream_gate0.012_offpolicy:same_fail/skipped; dream_maskr0.01:wrong_to_correct/no_gate; dream_t0.9:wrong_to_correct/no_gate |
| gate_overconservative_packaging_candidate | HumanEval/162 | 2 | 0 | 0.0303 | 0.0000 | dream_gate0.012_fresh_fixed:same_fail/skipped; dream_gate0.012_offpolicy:same_fail/skipped; dream_maskr0.01:wrong_to_correct/no_gate; dream_t0.9:wrong_to_correct/no_gate |
| gate_overconservative_packaging_candidate | HumanEval/74 | 2 | 0 | 0.0152 | 0.0000 | dream_gate0.012_fresh_fixed:same_fail/skipped; dream_gate0.012_offpolicy:same_fail/skipped; dream_maskr0.01:wrong_to_correct/no_gate; dream_t0.9:wrong_to_correct/no_gate |
| gate_overconservative_packaging_candidate | HumanEval/82 | 2 | 0 | 0.0154 | 0.0000 | dream_gate0.012_fresh_fixed:same_fail/skipped; dream_gate0.012_offpolicy:same_fail/skipped; dream_maskr0.01:wrong_to_correct/no_gate; dream_t0.9:wrong_to_correct/no_gate |
| gate_prevented_harm_candidate | HumanEval/4 | 0 | 2 | 0.0154 | 0.0000 | dream_gate0.012_fresh_fixed:same_pass/skipped; dream_gate0.012_offpolicy:same_pass/skipped; dream_maskr0.01:correct_to_wrong/no_gate; dream_t0.9:correct_to_wrong/no_gate |
| gate_prevented_packaging_harm_candidate | HumanEval/0 | 0 | 2 | 0.0175 | 0.0000 | dream_gate0.012_fresh_fixed:same_pass/skipped; dream_gate0.012_offpolicy:same_pass/skipped; dream_maskr0.01:correct_to_wrong/no_gate; dream_t0.9:correct_to_wrong/no_gate |
| gate_prevented_packaging_harm_candidate | HumanEval/115 | 0 | 2 | 0.0227 | 0.0000 | dream_gate0.012_fresh_fixed:same_pass/skipped; dream_gate0.012_offpolicy:same_pass/skipped; dream_maskr0.01:correct_to_wrong/no_gate; dream_t0.9:correct_to_wrong/no_gate |
| gate_prevented_packaging_harm_candidate | HumanEval/12 | 0 | 2 | 0.0357 | 0.0000 | dream_gate0.012_fresh_fixed:same_pass/skipped; dream_gate0.012_offpolicy:same_pass/skipped; dream_maskr0.01:correct_to_wrong/no_gate; dream_t0.9:correct_to_wrong/no_gate |
| gate_prevented_packaging_harm_candidate | HumanEval/14 | 0 | 2 | 0.0303 | 0.0000 | dream_gate0.012_fresh_fixed:same_pass/skipped; dream_gate0.012_offpolicy:same_pass/skipped; dream_maskr0.01:correct_to_wrong/no_gate; dream_t0.9:correct_to_wrong/no_gate |
| gate_prevented_packaging_harm_candidate | HumanEval/17 | 0 | 2 | 0.0179 | 0.0000 | dream_gate0.012_fresh_fixed:same_pass/skipped; dream_gate0.012_offpolicy:same_pass/skipped; dream_maskr0.01:correct_to_wrong/no_gate; dream_t0.9:correct_to_wrong/no_gate |
| gate_prevented_packaging_harm_candidate | HumanEval/21 | 0 | 2 | 0.0185 | 0.0000 | dream_gate0.012_fresh_fixed:same_pass/skipped; dream_gate0.012_offpolicy:same_pass/skipped; dream_maskr0.01:correct_to_wrong/no_gate; dream_t0.9:correct_to_wrong/no_gate |
| gate_prevented_packaging_harm_candidate | HumanEval/25 | 0 | 2 | 0.0127 | 0.0000 | dream_gate0.012_fresh_fixed:same_pass/skipped; dream_gate0.012_offpolicy:same_pass/skipped; dream_maskr0.01:correct_to_wrong/no_gate; dream_t0.9:correct_to_wrong/no_gate |
| gate_prevented_packaging_harm_candidate | HumanEval/28 | 0 | 2 | 0.0625 | 0.0000 | dream_gate0.012_fresh_fixed:same_pass/skipped; dream_gate0.012_offpolicy:same_pass/skipped; dream_maskr0.01:correct_to_wrong/no_gate; dream_t0.9:correct_to_wrong/no_gate |
| low_disagreement_packaging_hurt_candidate | HumanEval/1 | 0 | 3 | 0.0093 | 0.0000 | dream_gate0.012_fresh_fixed:same_pass/kept; dream_gate0.012_offpolicy:correct_to_wrong/kept; dream_maskr0.01:correct_to_wrong/no_gate; dream_t0.9:correct_to_wrong/no_gate |
| low_disagreement_packaging_hurt_candidate | HumanEval/20 | 0 | 3 | 0.0098 | 0.0000 | dream_gate0.012_fresh_fixed:same_pass/kept; dream_gate0.012_offpolicy:correct_to_wrong/kept; dream_maskr0.01:correct_to_wrong/no_gate; dream_t0.9:correct_to_wrong/no_gate |
| low_disagreement_packaging_hurt_candidate | HumanEval/6 | 0 | 3 | 0.0108 | 0.0000 | dream_gate0.012_fresh_fixed:same_pass/kept; dream_gate0.012_offpolicy:correct_to_wrong/kept; dream_maskr0.01:correct_to_wrong/no_gate; dream_t0.9:correct_to_wrong/no_gate |
| offpolicy_packaging_fix_not_reproduced_candidate | HumanEval/107 | 3 | 0 | 0.0104 | 0.0000 | dream_gate0.012_fresh_fixed:same_fail/kept; dream_gate0.012_offpolicy:wrong_to_correct/kept; dream_maskr0.01:wrong_to_correct/no_gate; dream_t0.9:wrong_to_correct/no_gate |
| offpolicy_packaging_fix_not_reproduced_candidate | HumanEval/119 | 3 | 0 | 0.0097 | 0.0000 | dream_gate0.012_fresh_fixed:same_fail/kept; dream_gate0.012_offpolicy:wrong_to_correct/kept; dream_maskr0.01:wrong_to_correct/no_gate; dream_t0.9:wrong_to_correct/no_gate |
| offpolicy_packaging_fix_not_reproduced_candidate | HumanEval/143 | 3 | 0 | 0.0115 | 0.0000 | dream_gate0.012_fresh_fixed:same_fail/kept; dream_gate0.012_offpolicy:wrong_to_correct/kept; dream_maskr0.01:wrong_to_correct/no_gate; dream_t0.9:wrong_to_correct/no_gate |
| offpolicy_packaging_fix_not_reproduced_candidate | HumanEval/146 | 3 | 0 | 0.0103 | 0.0000 | dream_gate0.012_fresh_fixed:same_fail/kept; dream_gate0.012_offpolicy:wrong_to_correct/kept; dream_maskr0.01:wrong_to_correct/no_gate; dream_t0.9:wrong_to_correct/no_gate |
| offpolicy_packaging_fix_not_reproduced_candidate | HumanEval/75 | 3 | 0 | 0.0076 | 0.0000 | dream_gate0.012_fresh_fixed:same_fail/kept; dream_gate0.012_offpolicy:wrong_to_correct/kept; dream_maskr0.01:wrong_to_correct/no_gate; dream_t0.9:wrong_to_correct/no_gate |
| offpolicy_packaging_fix_not_reproduced_candidate | HumanEval/81 | 3 | 0 | 0.0088 | 0.0000 | dream_gate0.012_fresh_fixed:same_fail/kept; dream_gate0.012_offpolicy:wrong_to_correct/kept; dream_maskr0.01:wrong_to_correct/no_gate; dream_t0.9:wrong_to_correct/no_gate |
| offpolicy_packaging_fix_not_reproduced_candidate | HumanEval/94 | 3 | 0 | 0.0104 | 0.0000 | dream_gate0.012_fresh_fixed:same_fail/kept; dream_gate0.012_offpolicy:wrong_to_correct/kept; dream_maskr0.01:wrong_to_correct/no_gate; dream_t0.9:wrong_to_correct/no_gate |
| packaging_or_eval_artifact_candidate | HumanEval/1 | 0 | 3 | 0.0093 | 0.0000 | dream_gate0.012_fresh_fixed:same_pass/kept; dream_gate0.012_offpolicy:correct_to_wrong/kept; dream_maskr0.01:correct_to_wrong/no_gate; dream_t0.9:correct_to_wrong/no_gate |
| packaging_or_eval_artifact_candidate | HumanEval/107 | 3 | 0 | 0.0104 | 0.0000 | dream_gate0.012_fresh_fixed:same_fail/kept; dream_gate0.012_offpolicy:wrong_to_correct/kept; dream_maskr0.01:wrong_to_correct/no_gate; dream_t0.9:wrong_to_correct/no_gate |
| packaging_or_eval_artifact_candidate | HumanEval/119 | 3 | 0 | 0.0097 | 0.0000 | dream_gate0.012_fresh_fixed:same_fail/kept; dream_gate0.012_offpolicy:wrong_to_correct/kept; dream_maskr0.01:wrong_to_correct/no_gate; dream_t0.9:wrong_to_correct/no_gate |
| packaging_or_eval_artifact_candidate | HumanEval/143 | 3 | 0 | 0.0115 | 0.0000 | dream_gate0.012_fresh_fixed:same_fail/kept; dream_gate0.012_offpolicy:wrong_to_correct/kept; dream_maskr0.01:wrong_to_correct/no_gate; dream_t0.9:wrong_to_correct/no_gate |
| packaging_or_eval_artifact_candidate | HumanEval/146 | 3 | 0 | 0.0103 | 0.0000 | dream_gate0.012_fresh_fixed:same_fail/kept; dream_gate0.012_offpolicy:wrong_to_correct/kept; dream_maskr0.01:wrong_to_correct/no_gate; dream_t0.9:wrong_to_correct/no_gate |
| packaging_or_eval_artifact_candidate | HumanEval/20 | 0 | 3 | 0.0098 | 0.0000 | dream_gate0.012_fresh_fixed:same_pass/kept; dream_gate0.012_offpolicy:correct_to_wrong/kept; dream_maskr0.01:correct_to_wrong/no_gate; dream_t0.9:correct_to_wrong/no_gate |
| packaging_or_eval_artifact_candidate | HumanEval/6 | 0 | 3 | 0.0108 | 0.0000 | dream_gate0.012_fresh_fixed:same_pass/kept; dream_gate0.012_offpolicy:correct_to_wrong/kept; dream_maskr0.01:correct_to_wrong/no_gate; dream_t0.9:correct_to_wrong/no_gate |
| packaging_or_eval_artifact_candidate | HumanEval/75 | 3 | 0 | 0.0076 | 0.0000 | dream_gate0.012_fresh_fixed:same_fail/kept; dream_gate0.012_offpolicy:wrong_to_correct/kept; dream_maskr0.01:wrong_to_correct/no_gate; dream_t0.9:wrong_to_correct/no_gate |
| rewrite_stochastic_candidate | HumanEval/75 | 3 | 0 | 0.0076 | 0.0000 | dream_gate0.012_fresh_fixed:same_fail/kept; dream_gate0.012_offpolicy:wrong_to_correct/kept; dream_maskr0.01:wrong_to_correct/no_gate; dream_t0.9:wrong_to_correct/no_gate |

## Notes

- `gate_prevented_harm_candidate`: a no-gate run breaks a baseline-correct task and a gated run skips it while preserving pass.
- `gate_prevented_packaging_harm_candidate`: same pattern, but raw code is unchanged, so the observed harm is likely packaging/eval rather than rewriting.
- `offpolicy_fix_not_reproduced_candidate`: an off-policy/materialized gate run fixes a baseline-failing task but a fresh run does not.
- `offpolicy_packaging_fix_not_reproduced_candidate`: same pattern, but raw code is unchanged, so it should be audited as packaging/eval first.
- `low_disagreement_hurt_candidate`: a gated run keeps refinement and still causes `correct->wrong`.
- `low_disagreement_packaging_hurt_candidate`: same pattern, but raw code is unchanged.
- `gate_overconservative_candidate`: a no-gate run fixes a baseline-failing task but a gated run skips it.
- `gate_overconservative_packaging_candidate`: same pattern, but the apparent no-gate fix is likely packaging/eval.
- `packaging_or_eval_artifact_candidate`: raw output is unchanged from baseline but eval outcome changes.
- Locator span overlap cannot be judged from older artifacts unless mask spans were recorded during generation.
