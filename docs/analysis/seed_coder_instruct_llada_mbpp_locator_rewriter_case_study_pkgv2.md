# Remask Locator/Rewriter Case-Study Analysis

This report is generated mechanically. Category labels are candidate labels for case-study selection, not final semantic judgments.

## Inputs

- Dataset: `mbpp`
- Baseline JSONL: `outputs/base_tuteng/packaging_v2/seed-coder-instruct_mbpp_pkgv2.jsonl`
- Baseline eval: `outputs/base_tuteng/seed-coder-instruct_mbpp_eval_results.json`
- Run `llada_t0.9_pkgv2`: `outputs/base_tuteng/packaging_v2/seed-coder-instruct_llada_remask_mbpp_t0.9_pkgv2.jsonl` / `outputs/base_tuteng/seed-coder-instruct_llada_remask_mbpp_t0.9_eval_results.json`
- Run `llada_gate0.012_fixed_pkgv2`: `outputs/base_tuteng/packaging_v2/seed-coder-instruct_llada_remask_mbpp_maskr0.01_gate0.012_fixed_pkgv2.jsonl` / `outputs/base_tuteng/seed-coder-instruct_llada_remask_mbpp_maskr0.01_gate0.012_fixed_eval_results.json`

## Run-Level Outcomes

| run | n | plus% | delta plus pp | wrong->correct | correct->wrong | same pass | same fail | skipped | kept | mean mask frac | median edit ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llada_gate0.012_fixed_pkgv2 | 378 | 72.2 | +0.0 | 0 | 0 | 273 | 105 | 333 | 45 | 0.0350 | 0.0000 |
| llada_t0.9_pkgv2 | 378 | 64.6 | -7.7 | 1 | 30 | 243 | 104 | 0 | 0 |  | 0.0000 |

## Candidate Category Counts

| category | n tasks | examples |
| --- | --- | --- |
| baseline_correct_no_observed_harm | 243 | Mbpp/100, Mbpp/104, Mbpp/108, Mbpp/109, Mbpp/111, Mbpp/12, Mbpp/120, Mbpp/127, Mbpp/128, Mbpp/129, Mbpp/131, Mbpp/133 |
| baseline_wrong_rewrite_ineffective | 104 | Mbpp/101, Mbpp/102, Mbpp/103, Mbpp/11, Mbpp/113, Mbpp/118, Mbpp/119, Mbpp/123, Mbpp/124, Mbpp/125, Mbpp/126, Mbpp/130 |
| gate_prevented_harm_candidate | 29 | Mbpp/105, Mbpp/106, Mbpp/116, Mbpp/132, Mbpp/135, Mbpp/161, Mbpp/168, Mbpp/227, Mbpp/242, Mbpp/250, Mbpp/272, Mbpp/290 |
| gate_overconservative_candidate | 1 | Mbpp/406 |
| rewrite_stochastic_candidate | 1 | Mbpp/406 |
| uncategorized | 1 | Mbpp/721 |

## Mask-Fraction Bins

| run | mask_fraction bin | n | wrong->correct | correct->wrong | same pass | same fail | net |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llada_gate0.012_fixed_pkgv2 | [0.000, 0.008) | 14 | 0 | 0 | 7 | 7 | +0 |
| llada_gate0.012_fixed_pkgv2 | [0.008, 0.010) | 15 | 0 | 0 | 8 | 7 | +0 |
| llada_gate0.012_fixed_pkgv2 | [0.010, 0.012) | 16 | 0 | 0 | 8 | 8 | +0 |
| llada_gate0.012_fixed_pkgv2 | [0.012, 0.015) | 24 | 0 | 0 | 15 | 9 | +0 |
| llada_gate0.012_fixed_pkgv2 | [0.015, 0.020) | 39 | 0 | 0 | 27 | 12 | +0 |
| llada_gate0.012_fixed_pkgv2 | [0.020, 0.030) | 57 | 0 | 0 | 41 | 16 | +0 |
| llada_gate0.012_fixed_pkgv2 | [0.030, 0.050) | 132 | 0 | 0 | 98 | 34 | +0 |
| llada_gate0.012_fixed_pkgv2 | [0.050, inf) | 81 | 0 | 0 | 69 | 12 | +0 |

## Edit Metrics By Transition

| run | transition | n | median char edit ratio | p90 char edit ratio | median changed lines | changed signature | changed imports | refined parse fails |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llada_gate0.012_fixed_pkgv2 | same_fail | 105 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | 0 |
| llada_gate0.012_fixed_pkgv2 | same_pass | 273 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | 0 |
| llada_t0.9_pkgv2 | correct_to_wrong | 30 | 0.0221 | 0.0601 | 1.00 | 27 | 0 | 27 |
| llada_t0.9_pkgv2 | same_fail | 104 | 0.0000 | 0.0085 | 0.00 | 7 | 0 | 6 |
| llada_t0.9_pkgv2 | same_pass | 243 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | 0 |
| llada_t0.9_pkgv2 | wrong_to_correct | 1 | 0.0244 | 0.0244 | 1.00 | 0 | 0 | 0 |

## Deterministic Case Candidates

| category | task | wrong->correct runs | correct->wrong runs | median mask frac | median edit ratio | evidence |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_correct_no_observed_harm | Mbpp/100 | 0 | 0 | 0.0256 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_pass/skipped; llada_t0.9_pkgv2:same_pass/no_gate |
| baseline_correct_no_observed_harm | Mbpp/104 | 0 | 0 | 0.0417 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_pass/skipped; llada_t0.9_pkgv2:same_pass/no_gate |
| baseline_correct_no_observed_harm | Mbpp/108 | 0 | 0 | 0.0323 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_pass/skipped; llada_t0.9_pkgv2:same_pass/no_gate |
| baseline_correct_no_observed_harm | Mbpp/109 | 0 | 0 | 0.0147 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_pass/skipped; llada_t0.9_pkgv2:same_pass/no_gate |
| baseline_correct_no_observed_harm | Mbpp/111 | 0 | 0 | 0.0164 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_pass/skipped; llada_t0.9_pkgv2:same_pass/no_gate |
| baseline_correct_no_observed_harm | Mbpp/12 | 0 | 0 | 0.0667 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_pass/skipped; llada_t0.9_pkgv2:same_pass/no_gate |
| baseline_correct_no_observed_harm | Mbpp/120 | 0 | 0 | 0.0417 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_pass/skipped; llada_t0.9_pkgv2:same_pass/no_gate |
| baseline_correct_no_observed_harm | Mbpp/127 | 0 | 0 | 0.0769 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_pass/skipped; llada_t0.9_pkgv2:same_pass/no_gate |
| baseline_correct_no_observed_harm | Mbpp/128 | 0 | 0 | 0.0333 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_pass/skipped; llada_t0.9_pkgv2:same_pass/no_gate |
| baseline_correct_no_observed_harm | Mbpp/129 | 0 | 0 | 0.0055 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_pass/kept; llada_t0.9_pkgv2:same_pass/no_gate |
| baseline_correct_no_observed_harm | Mbpp/131 | 0 | 0 | 0.0070 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_pass/kept; llada_t0.9_pkgv2:same_pass/no_gate |
| baseline_correct_no_observed_harm | Mbpp/133 | 0 | 0 | 0.0417 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_pass/skipped; llada_t0.9_pkgv2:same_pass/no_gate |
| baseline_wrong_rewrite_ineffective | Mbpp/101 | 0 | 0 | 0.0286 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_fail/skipped; llada_t0.9_pkgv2:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | Mbpp/102 | 0 | 0 | 0.0244 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_fail/skipped; llada_t0.9_pkgv2:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | Mbpp/103 | 0 | 0 | 0.0108 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_fail/kept; llada_t0.9_pkgv2:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | Mbpp/11 | 0 | 0 | 0.0213 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_fail/skipped; llada_t0.9_pkgv2:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | Mbpp/113 | 0 | 0 | 0.0370 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_fail/skipped; llada_t0.9_pkgv2:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | Mbpp/118 | 0 | 0 | 0.0833 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_fail/skipped; llada_t0.9_pkgv2:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | Mbpp/119 | 0 | 0 | 0.0106 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_fail/kept; llada_t0.9_pkgv2:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | Mbpp/123 | 0 | 0 | 0.0087 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_fail/kept; llada_t0.9_pkgv2:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | Mbpp/124 | 0 | 0 | 0.0435 | 0.0390 | llada_gate0.012_fixed_pkgv2:same_fail/skipped; llada_t0.9_pkgv2:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | Mbpp/125 | 0 | 0 | 0.0110 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_fail/kept; llada_t0.9_pkgv2:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | Mbpp/126 | 0 | 0 | 0.0083 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_fail/kept; llada_t0.9_pkgv2:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | Mbpp/130 | 0 | 0 | 0.0476 | 0.0000 | llada_gate0.012_fixed_pkgv2:same_fail/skipped; llada_t0.9_pkgv2:same_fail/no_gate |
| gate_overconservative_candidate | Mbpp/406 | 1 | 0 | 0.0625 | 0.0122 | llada_gate0.012_fixed_pkgv2:same_fail/skipped; llada_t0.9_pkgv2:wrong_to_correct/no_gate |
| gate_prevented_harm_candidate | Mbpp/105 | 0 | 1 | 0.0833 | 0.0143 | llada_gate0.012_fixed_pkgv2:same_pass/skipped; llada_t0.9_pkgv2:correct_to_wrong/no_gate |
| gate_prevented_harm_candidate | Mbpp/106 | 0 | 1 | 0.0526 | 0.0096 | llada_gate0.012_fixed_pkgv2:same_pass/skipped; llada_t0.9_pkgv2:correct_to_wrong/no_gate |
| gate_prevented_harm_candidate | Mbpp/116 | 0 | 1 | 0.0500 | 0.0246 | llada_gate0.012_fixed_pkgv2:same_pass/skipped; llada_t0.9_pkgv2:correct_to_wrong/no_gate |
| gate_prevented_harm_candidate | Mbpp/132 | 0 | 1 | 0.0625 | 0.0114 | llada_gate0.012_fixed_pkgv2:same_pass/skipped; llada_t0.9_pkgv2:correct_to_wrong/no_gate |
| gate_prevented_harm_candidate | Mbpp/135 | 0 | 1 | 0.0556 | 0.0104 | llada_gate0.012_fixed_pkgv2:same_pass/skipped; llada_t0.9_pkgv2:correct_to_wrong/no_gate |
| gate_prevented_harm_candidate | Mbpp/161 | 0 | 1 | 0.0385 | 0.0122 | llada_gate0.012_fixed_pkgv2:same_pass/skipped; llada_t0.9_pkgv2:correct_to_wrong/no_gate |
| gate_prevented_harm_candidate | Mbpp/168 | 0 | 1 | 0.0714 | 0.0100 | llada_gate0.012_fixed_pkgv2:same_pass/skipped; llada_t0.9_pkgv2:correct_to_wrong/no_gate |
| gate_prevented_harm_candidate | Mbpp/227 | 0 | 1 | 0.0476 | 0.0300 | llada_gate0.012_fixed_pkgv2:same_pass/skipped; llada_t0.9_pkgv2:correct_to_wrong/no_gate |
| gate_prevented_harm_candidate | Mbpp/242 | 0 | 1 | 0.0833 | 0.0132 | llada_gate0.012_fixed_pkgv2:same_pass/skipped; llada_t0.9_pkgv2:correct_to_wrong/no_gate |
| gate_prevented_harm_candidate | Mbpp/250 | 0 | 1 | 0.0625 | 0.0089 | llada_gate0.012_fixed_pkgv2:same_pass/skipped; llada_t0.9_pkgv2:correct_to_wrong/no_gate |
| gate_prevented_harm_candidate | Mbpp/272 | 0 | 1 | 0.0476 | 0.0074 | llada_gate0.012_fixed_pkgv2:same_pass/skipped; llada_t0.9_pkgv2:correct_to_wrong/no_gate |
| gate_prevented_harm_candidate | Mbpp/290 | 0 | 1 | 0.0156 | 0.0023 | llada_gate0.012_fixed_pkgv2:same_pass/skipped; llada_t0.9_pkgv2:correct_to_wrong/no_gate |
| rewrite_stochastic_candidate | Mbpp/406 | 1 | 0 | 0.0625 | 0.0122 | llada_gate0.012_fixed_pkgv2:same_fail/skipped; llada_t0.9_pkgv2:wrong_to_correct/no_gate |

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
