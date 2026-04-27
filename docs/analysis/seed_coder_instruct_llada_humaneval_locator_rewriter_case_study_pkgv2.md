# Remask Locator/Rewriter Case-Study Analysis

This report is generated mechanically. Category labels are candidate labels for case-study selection, not final semantic judgments.

## Inputs

- Dataset: `humaneval`
- Baseline JSONL: `outputs/base_tuteng/packaging_v2/seed-coder-instruct_humaneval_pkgv2.jsonl`
- Baseline eval: `outputs/base_tuteng/packaging_v2/seed-coder-instruct_humaneval_pkgv2_eval_results.json`
- Run `llada_t0.9_pkgv2`: `outputs/base_tuteng/packaging_v2/seed-coder-instruct_llada_remask_humaneval_t0.9_pkgv2.jsonl` / `outputs/base_tuteng/packaging_v2/seed-coder-instruct_llada_remask_humaneval_t0.9_pkgv2_eval_results.json`
- Run `llada_gate0.012_fixed_pkgv2`: `outputs/base_tuteng/packaging_v2/seed-coder-instruct_llada_remask_humaneval_maskr0.01_gate0.012_fixed_pkgv2.jsonl` / `outputs/base_tuteng/packaging_v2/seed-coder-instruct_llada_remask_humaneval_maskr0.01_gate0.012_fixed_pkgv2_eval_results.json`

## Run-Level Outcomes

| run | n | plus% | delta plus pp | wrong->correct | correct->wrong | same pass | same fail | skipped | kept | mean mask frac | median edit ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llada_gate0.012_fixed_pkgv2 | 164 | 75.6 | +0.0 | 0 | 0 | 124 | 40 | 121 | 43 | 0.0202 | 0.0000 |
| llada_t0.9_pkgv2 | 164 | 72.6 | -3.0 | 0 | 5 | 119 | 40 | 0 | 0 |  | 0.0000 |

## Candidate Category Counts

| category | n tasks | examples |
| --- | --- | --- |
| baseline_correct_no_observed_harm | 119 | HumanEval/0, HumanEval/1, HumanEval/101, HumanEval/102, HumanEval/104, HumanEval/105, HumanEval/106, HumanEval/107, HumanEval/109, HumanEval/11, HumanEval/110, HumanEval/111 |
| baseline_wrong_rewrite_ineffective | 40 | HumanEval/10, HumanEval/100, HumanEval/108, HumanEval/122, HumanEval/124, HumanEval/125, HumanEval/126, HumanEval/127, HumanEval/129, HumanEval/130, HumanEval/132, HumanEval/133 |
| gate_prevented_harm_candidate | 5 | HumanEval/103, HumanEval/2, HumanEval/28, HumanEval/79, HumanEval/84 |

## Mask-Fraction Bins

| run | mask_fraction bin | n | wrong->correct | correct->wrong | same pass | same fail | net |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llada_gate0.012_fixed_pkgv2 | [0.000, 0.008) | 11 | 0 | 0 | 6 | 5 | +0 |
| llada_gate0.012_fixed_pkgv2 | [0.008, 0.010) | 17 | 0 | 0 | 12 | 5 | +0 |
| llada_gate0.012_fixed_pkgv2 | [0.010, 0.012) | 15 | 0 | 0 | 14 | 1 | +0 |
| llada_gate0.012_fixed_pkgv2 | [0.012, 0.015) | 36 | 0 | 0 | 27 | 9 | +0 |
| llada_gate0.012_fixed_pkgv2 | [0.015, 0.020) | 30 | 0 | 0 | 25 | 5 | +0 |
| llada_gate0.012_fixed_pkgv2 | [0.020, 0.030) | 25 | 0 | 0 | 18 | 7 | +0 |
| llada_gate0.012_fixed_pkgv2 | [0.030, 0.050) | 24 | 0 | 0 | 17 | 7 | +0 |
| llada_gate0.012_fixed_pkgv2 | [0.050, inf) | 6 | 0 | 0 | 5 | 1 | +0 |

## Edit Metrics By Transition

| run | transition | n | median char edit ratio | p90 char edit ratio | median changed lines | changed signature | changed imports | refined parse fails |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llada_gate0.012_fixed_pkgv2 | same_fail | 40 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | 0 |
| llada_gate0.012_fixed_pkgv2 | same_pass | 124 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | 0 |
| llada_t0.9_pkgv2 | correct_to_wrong | 5 | 0.0137 | 0.0183 | 1.00 | 5 | 0 | 5 |
| llada_t0.9_pkgv2 | same_fail | 40 | 0.0000 | 0.0000 | 0.00 | 2 | 0 | 2 |
| llada_t0.9_pkgv2 | same_pass | 119 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | 0 |

## Deterministic Case Candidates

| category | task | wrong->correct runs | correct->wrong runs | median mask frac | median edit ratio | evidence |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_correct_no_observed_harm | HumanEval/0 | 0 | 0 | 0.0152 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_pass/skipped; llada_t0.9_pkgv2:same_pass/no_gate |
| baseline_correct_no_observed_harm | HumanEval/1 | 0 | 0 | 0.0079 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_pass/kept; llada_t0.9_pkgv2:same_pass/no_gate |
| baseline_correct_no_observed_harm | HumanEval/101 | 0 | 0 | 0.0417 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_pass/skipped; llada_t0.9_pkgv2:same_pass/no_gate |
| baseline_correct_no_observed_harm | HumanEval/102 | 0 | 0 | 0.0182 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_pass/skipped; llada_t0.9_pkgv2:same_pass/no_gate |
| baseline_correct_no_observed_harm | HumanEval/104 | 0 | 0 | 0.0286 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_pass/skipped; llada_t0.9_pkgv2:same_pass/no_gate |
| baseline_correct_no_observed_harm | HumanEval/105 | 0 | 0 | 0.0120 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_pass/skipped; llada_t0.9_pkgv2:same_pass/no_gate |
| baseline_correct_no_observed_harm | HumanEval/106 | 0 | 0 | 0.0101 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_pass/kept; llada_t0.9_pkgv2:same_pass/no_gate |
| baseline_correct_no_observed_harm | HumanEval/107 | 0 | 0 | 0.0088 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_pass/kept; llada_t0.9_pkgv2:same_pass/no_gate |
| baseline_correct_no_observed_harm | HumanEval/109 | 0 | 0 | 0.0145 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_pass/skipped; llada_t0.9_pkgv2:same_pass/no_gate |
| baseline_correct_no_observed_harm | HumanEval/11 | 0 | 0 | 0.0256 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_pass/skipped; llada_t0.9_pkgv2:same_pass/no_gate |
| baseline_correct_no_observed_harm | HumanEval/110 | 0 | 0 | 0.0081 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_pass/kept; llada_t0.9_pkgv2:same_pass/no_gate |
| baseline_correct_no_observed_harm | HumanEval/111 | 0 | 0 | 0.0118 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_pass/kept; llada_t0.9_pkgv2:same_pass/no_gate |
| baseline_wrong_rewrite_ineffective | HumanEval/10 | 0 | 0 | 0.0122 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_fail/skipped; llada_t0.9_pkgv2:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | HumanEval/100 | 0 | 0 | 0.0143 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_fail/skipped; llada_t0.9_pkgv2:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | HumanEval/108 | 0 | 0 | 0.0208 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_fail/skipped; llada_t0.9_pkgv2:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | HumanEval/122 | 0 | 0 | 0.0370 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_fail/skipped; llada_t0.9_pkgv2:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | HumanEval/124 | 0 | 0 | 0.0084 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_fail/kept; llada_t0.9_pkgv2:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | HumanEval/125 | 0 | 0 | 0.0147 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_fail/skipped; llada_t0.9_pkgv2:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | HumanEval/126 | 0 | 0 | 0.0130 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_fail/skipped; llada_t0.9_pkgv2:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | HumanEval/127 | 0 | 0 | 0.0077 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_fail/kept; llada_t0.9_pkgv2:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | HumanEval/129 | 0 | 0 | 0.0085 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_fail/kept; llada_t0.9_pkgv2:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | HumanEval/130 | 0 | 0 | 0.0092 | 0.0034 | llada_gate0.012_fixed_pkgv2:same_fail/kept; llada_t0.9_pkgv2:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | HumanEval/132 | 0 | 0 | 0.0161 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_fail/skipped; llada_t0.9_pkgv2:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | HumanEval/133 | 0 | 0 | 0.0455 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_fail/skipped; llada_t0.9_pkgv2:same_fail/no_gate |
| gate_prevented_harm_candidate | HumanEval/103 | 0 | 1 | 0.0204 | 0.0039 | llada_gate0.012_fixed_pkgv2:same_pass/skipped; llada_t0.9_pkgv2:correct_to_wrong/no_gate |
| gate_prevented_harm_candidate | HumanEval/2 | 0 | 1 | 0.0556 | 0.0066 | llada_gate0.012_fixed_pkgv2:same_pass/skipped; llada_t0.9_pkgv2:correct_to_wrong/no_gate |
| gate_prevented_harm_candidate | HumanEval/28 | 0 | 1 | 0.0500 | 0.0070 | llada_gate0.012_fixed_pkgv2:same_pass/skipped; llada_t0.9_pkgv2:correct_to_wrong/no_gate |
| gate_prevented_harm_candidate | HumanEval/79 | 0 | 1 | 0.0417 | 0.0068 | llada_gate0.012_fixed_pkgv2:same_pass/skipped; llada_t0.9_pkgv2:correct_to_wrong/no_gate |
| gate_prevented_harm_candidate | HumanEval/84 | 0 | 1 | 0.0333 | 0.0105 | llada_gate0.012_fixed_pkgv2:same_pass/skipped; llada_t0.9_pkgv2:correct_to_wrong/no_gate |

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
