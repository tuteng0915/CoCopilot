# Remask Locator/Rewriter Case-Study Analysis

This report is generated mechanically. Category labels are candidate labels for case-study selection, not final semantic judgments.

## Inputs

- Dataset: `humaneval`
- Baseline JSONL: `outputs/base_tuteng/packaging_v2/seed-coder-instruct_humaneval_pkgv2.jsonl`
- Baseline eval: `outputs/base_tuteng/packaging_v2/seed-coder-instruct_humaneval_pkgv2_eval_results.json`
- Run `dream_t0.9_pkgv2`: `outputs/base_tuteng/packaging_v2/seed-coder-instruct_dream_remask_humaneval_t0.9_pkgv2.jsonl` / `outputs/base_tuteng/packaging_v2/seed-coder-instruct_dream_remask_humaneval_t0.9_pkgv2_eval_results.json`
- Run `dream_maskr0.01_pkgv2`: `outputs/base_tuteng/packaging_v2/seed-coder-instruct_dream_remask_humaneval_maskr0.01_pkgv2.jsonl` / `outputs/base_tuteng/packaging_v2/seed-coder-instruct_dream_remask_humaneval_maskr0.01_pkgv2_eval_results.json`
- Run `dream_gate0.012_offpolicy_pkgv2`: `outputs/base_tuteng/packaging_v2/seed-coder-instruct_dream_remask_humaneval_maskr0.01_gate0.012_pkgv2.jsonl` / `outputs/base_tuteng/packaging_v2/seed-coder-instruct_dream_remask_humaneval_maskr0.01_gate0.012_pkgv2_eval_results.json`
- Run `dream_gate0.012_fresh_fixed_pkgv2`: `outputs/base_tuteng/packaging_v2/seed-coder-instruct_dream_remask_humaneval_maskr0.01_gate0.012_fresh_fixed_pkgv2.jsonl` / `outputs/base_tuteng/packaging_v2/seed-coder-instruct_dream_remask_humaneval_maskr0.01_gate0.012_fresh_fixed_pkgv2_eval_results.json`

## Run-Level Outcomes

| run | n | plus% | delta plus pp | wrong->correct | correct->wrong | same pass | same fail | skipped | kept | mean mask frac | median edit ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dream_gate0.012_fresh_fixed_pkgv2 | 164 | 75.6 | +0.0 | 0 | 0 | 124 | 40 | 130 | 34 | 0.0223 | 0.0000 |
| dream_gate0.012_offpolicy_pkgv2 | 164 | 75.6 | +0.0 | 0 | 0 | 124 | 40 | 130 | 34 | 0.0223 | 0.0000 |
| dream_maskr0.01_pkgv2 | 164 | 75.6 | +0.0 | 0 | 0 | 124 | 40 | 0 | 0 | 0.0223 | 0.0000 |
| dream_t0.9_pkgv2 | 164 | 75.6 | +0.0 | 0 | 0 | 124 | 40 | 0 | 0 |  | 0.0000 |

## Candidate Category Counts

| category | n tasks | examples |
| --- | --- | --- |
| baseline_correct_no_observed_harm | 124 | HumanEval/0, HumanEval/1, HumanEval/101, HumanEval/102, HumanEval/103, HumanEval/104, HumanEval/105, HumanEval/106 |
| baseline_wrong_rewrite_ineffective | 40 | HumanEval/10, HumanEval/100, HumanEval/108, HumanEval/122, HumanEval/124, HumanEval/125, HumanEval/126, HumanEval/127 |

## Mask-Fraction Bins

| run | mask_fraction bin | n | wrong->correct | correct->wrong | same pass | same fail | net |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dream_gate0.012_fresh_fixed_pkgv2 | [0.000, 0.008) | 6 | 0 | 0 | 3 | 3 | +0 |
| dream_gate0.012_fresh_fixed_pkgv2 | [0.008, 0.010) | 17 | 0 | 0 | 11 | 6 | +0 |
| dream_gate0.012_fresh_fixed_pkgv2 | [0.010, 0.012) | 11 | 0 | 0 | 10 | 1 | +0 |
| dream_gate0.012_fresh_fixed_pkgv2 | [0.012, 0.015) | 29 | 0 | 0 | 22 | 7 | +0 |
| dream_gate0.012_fresh_fixed_pkgv2 | [0.015, 0.020) | 34 | 0 | 0 | 28 | 6 | +0 |
| dream_gate0.012_fresh_fixed_pkgv2 | [0.020, 0.030) | 30 | 0 | 0 | 22 | 8 | +0 |
| dream_gate0.012_fresh_fixed_pkgv2 | [0.030, 0.050) | 27 | 0 | 0 | 20 | 7 | +0 |
| dream_gate0.012_fresh_fixed_pkgv2 | [0.050, inf) | 10 | 0 | 0 | 8 | 2 | +0 |
| dream_gate0.012_offpolicy_pkgv2 | [0.000, 0.008) | 6 | 0 | 0 | 3 | 3 | +0 |
| dream_gate0.012_offpolicy_pkgv2 | [0.008, 0.010) | 17 | 0 | 0 | 11 | 6 | +0 |
| dream_gate0.012_offpolicy_pkgv2 | [0.010, 0.012) | 11 | 0 | 0 | 10 | 1 | +0 |
| dream_gate0.012_offpolicy_pkgv2 | [0.012, 0.015) | 29 | 0 | 0 | 22 | 7 | +0 |
| dream_gate0.012_offpolicy_pkgv2 | [0.015, 0.020) | 34 | 0 | 0 | 28 | 6 | +0 |
| dream_gate0.012_offpolicy_pkgv2 | [0.020, 0.030) | 30 | 0 | 0 | 22 | 8 | +0 |
| dream_gate0.012_offpolicy_pkgv2 | [0.030, 0.050) | 27 | 0 | 0 | 20 | 7 | +0 |
| dream_gate0.012_offpolicy_pkgv2 | [0.050, inf) | 10 | 0 | 0 | 8 | 2 | +0 |
| dream_maskr0.01_pkgv2 | [0.000, 0.008) | 6 | 0 | 0 | 3 | 3 | +0 |
| dream_maskr0.01_pkgv2 | [0.008, 0.010) | 17 | 0 | 0 | 11 | 6 | +0 |
| dream_maskr0.01_pkgv2 | [0.010, 0.012) | 11 | 0 | 0 | 10 | 1 | +0 |
| dream_maskr0.01_pkgv2 | [0.012, 0.015) | 29 | 0 | 0 | 22 | 7 | +0 |
| dream_maskr0.01_pkgv2 | [0.015, 0.020) | 34 | 0 | 0 | 28 | 6 | +0 |
| dream_maskr0.01_pkgv2 | [0.020, 0.030) | 30 | 0 | 0 | 22 | 8 | +0 |
| dream_maskr0.01_pkgv2 | [0.030, 0.050) | 27 | 0 | 0 | 20 | 7 | +0 |
| dream_maskr0.01_pkgv2 | [0.050, inf) | 10 | 0 | 0 | 8 | 2 | +0 |

## Edit Metrics By Transition

| run | transition | n | median char edit ratio | p90 char edit ratio | median changed lines | changed signature | changed imports | refined parse fails |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dream_gate0.012_fresh_fixed_pkgv2 | same_fail | 40 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | 0 |
| dream_gate0.012_fresh_fixed_pkgv2 | same_pass | 124 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | 0 |
| dream_gate0.012_offpolicy_pkgv2 | same_fail | 40 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | 0 |
| dream_gate0.012_offpolicy_pkgv2 | same_pass | 124 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | 0 |
| dream_maskr0.01_pkgv2 | same_fail | 40 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | 0 |
| dream_maskr0.01_pkgv2 | same_pass | 124 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | 0 |
| dream_t0.9_pkgv2 | same_fail | 40 | 0.0000 | 0.0022 | 0.00 | 0 | 0 | 0 |
| dream_t0.9_pkgv2 | same_pass | 124 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | 0 |

## Deterministic Case Candidates

| category | task | wrong->correct runs | correct->wrong runs | median mask frac | median edit ratio | evidence |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_correct_no_observed_harm | HumanEval/0 | 0 | 0 | 0.0175 | 0.0000 | dream_gate0.012_fresh_fixed_pkgv2:same_pass/skipped; dream_gate0.012_offpolicy_pkgv2:same_pass/skipped; dream_maskr0.01_pkgv2:same_pass/no_gate; dream_t0.9_pkgv2:same_pass/no_gate |
| baseline_correct_no_observed_harm | HumanEval/1 | 0 | 0 | 0.0093 | 0.0000 | dream_gate0.012_fresh_fixed_pkgv2:same_pass/kept; dream_gate0.012_offpolicy_pkgv2:same_pass/kept; dream_maskr0.01_pkgv2:same_pass/no_gate; dream_t0.9_pkgv2:same_pass/no_gate |
| baseline_correct_no_observed_harm | HumanEval/101 | 0 | 0 | 0.0476 | 0.0000 | dream_gate0.012_fresh_fixed_pkgv2:same_pass/skipped; dream_gate0.012_offpolicy_pkgv2:same_pass/skipped; dream_maskr0.01_pkgv2:same_pass/no_gate; dream_t0.9_pkgv2:same_pass/no_gate |
| baseline_correct_no_observed_harm | HumanEval/102 | 0 | 0 | 0.0196 | 0.0000 | dream_gate0.012_fresh_fixed_pkgv2:same_pass/skipped; dream_gate0.012_offpolicy_pkgv2:same_pass/skipped; dream_maskr0.01_pkgv2:same_pass/no_gate; dream_t0.9_pkgv2:same_pass/no_gate |
| baseline_correct_no_observed_harm | HumanEval/103 | 0 | 0 | 0.0222 | 0.0000 | dream_gate0.012_fresh_fixed_pkgv2:same_pass/skipped; dream_gate0.012_offpolicy_pkgv2:same_pass/skipped; dream_maskr0.01_pkgv2:same_pass/no_gate; dream_t0.9_pkgv2:same_pass/no_gate |
| baseline_correct_no_observed_harm | HumanEval/104 | 0 | 0 | 0.0303 | 0.0000 | dream_gate0.012_fresh_fixed_pkgv2:same_pass/skipped; dream_gate0.012_offpolicy_pkgv2:same_pass/skipped; dream_maskr0.01_pkgv2:same_pass/no_gate; dream_t0.9_pkgv2:same_pass/no_gate |
| baseline_correct_no_observed_harm | HumanEval/105 | 0 | 0 | 0.0127 | 0.0000 | dream_gate0.012_fresh_fixed_pkgv2:same_pass/skipped; dream_gate0.012_offpolicy_pkgv2:same_pass/skipped; dream_maskr0.01_pkgv2:same_pass/no_gate; dream_t0.9_pkgv2:same_pass/no_gate |
| baseline_correct_no_observed_harm | HumanEval/106 | 0 | 0 | 0.0112 | 0.0000 | dream_gate0.012_fresh_fixed_pkgv2:same_pass/kept; dream_gate0.012_offpolicy_pkgv2:same_pass/kept; dream_maskr0.01_pkgv2:same_pass/no_gate; dream_t0.9_pkgv2:same_pass/no_gate |
| baseline_wrong_rewrite_ineffective | HumanEval/10 | 0 | 0 | 0.0147 | 0.0000 | dream_gate0.012_fresh_fixed_pkgv2:same_fail/skipped; dream_gate0.012_offpolicy_pkgv2:same_fail/skipped; dream_maskr0.01_pkgv2:same_fail/no_gate; dream_t0.9_pkgv2:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | HumanEval/100 | 0 | 0 | 0.0159 | 0.0000 | dream_gate0.012_fresh_fixed_pkgv2:same_fail/skipped; dream_gate0.012_offpolicy_pkgv2:same_fail/skipped; dream_maskr0.01_pkgv2:same_fail/no_gate; dream_t0.9_pkgv2:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | HumanEval/108 | 0 | 0 | 0.0233 | 0.0000 | dream_gate0.012_fresh_fixed_pkgv2:same_fail/skipped; dream_gate0.012_offpolicy_pkgv2:same_fail/skipped; dream_maskr0.01_pkgv2:same_fail/no_gate; dream_t0.9_pkgv2:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | HumanEval/122 | 0 | 0 | 0.0385 | 0.0000 | dream_gate0.012_fresh_fixed_pkgv2:same_fail/skipped; dream_gate0.012_offpolicy_pkgv2:same_fail/skipped; dream_maskr0.01_pkgv2:same_fail/no_gate; dream_t0.9_pkgv2:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | HumanEval/124 | 0 | 0 | 0.0094 | 0.0000 | dream_gate0.012_fresh_fixed_pkgv2:same_fail/kept; dream_gate0.012_offpolicy_pkgv2:same_fail/kept; dream_maskr0.01_pkgv2:same_fail/no_gate; dream_t0.9_pkgv2:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | HumanEval/125 | 0 | 0 | 0.0164 | 0.0000 | dream_gate0.012_fresh_fixed_pkgv2:same_fail/skipped; dream_gate0.012_offpolicy_pkgv2:same_fail/skipped; dream_maskr0.01_pkgv2:same_fail/no_gate; dream_t0.9_pkgv2:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | HumanEval/126 | 0 | 0 | 0.0149 | 0.0000 | dream_gate0.012_fresh_fixed_pkgv2:same_fail/skipped; dream_gate0.012_offpolicy_pkgv2:same_fail/skipped; dream_maskr0.01_pkgv2:same_fail/no_gate; dream_t0.9_pkgv2:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | HumanEval/127 | 0 | 0 | 0.0090 | 0.0000 | dream_gate0.012_fresh_fixed_pkgv2:same_fail/kept; dream_gate0.012_offpolicy_pkgv2:same_fail/kept; dream_maskr0.01_pkgv2:same_fail/no_gate; dream_t0.9_pkgv2:same_fail/no_gate |

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
