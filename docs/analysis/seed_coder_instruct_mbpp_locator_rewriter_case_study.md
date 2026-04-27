# Remask Locator/Rewriter Case-Study Analysis

> Historical note (2026-04-20): MBPP packaging normalization changed `0/378` records, so these numbers are still consistent with pkgv2. Prefer the pkgv2-named report (`docs/analysis/seed_coder_instruct_mbpp_locator_rewriter_case_study_pkgv2.md`) for current citations.

This report is generated mechanically. Category labels are candidate labels for case-study selection, not final semantic judgments.

## Inputs

- Dataset: `mbpp`
- Baseline JSONL: `outputs/base_tuteng/seed-coder-instruct_mbpp.jsonl`
- Baseline eval: `outputs/base_tuteng/seed-coder-instruct_mbpp_eval_results.json`
- Run `dream_t0.9`: `outputs/base_tuteng/seed-coder-instruct_dream_remask_mbpp_t0.9.jsonl` / `outputs/base_tuteng/seed-coder-instruct_dream_remask_mbpp_t0.9_eval_results.json`
- Run `dream_maskr0.01`: `outputs/base_tuteng/seed-coder-instruct_dream_remask_mbpp_maskr0.01.jsonl` / `outputs/base_tuteng/seed-coder-instruct_dream_remask_mbpp_maskr0.01_eval_results.json`
- Run `dream_gate0.012_offpolicy`: `outputs/base_tuteng/seed-coder-instruct_dream_remask_mbpp_maskr0.01_gate0.012.jsonl` / `outputs/base_tuteng/seed-coder-instruct_dream_remask_mbpp_maskr0.01_gate0.012_eval_results.json`
- Run `dream_gate0.012_fresh_fixed`: `outputs/base_tuteng/seed-coder-instruct_dream_remask_mbpp_maskr0.01_gate0.012_fresh_fixed.jsonl` / `outputs/base_tuteng/seed-coder-instruct_dream_remask_mbpp_maskr0.01_gate0.012_fresh_fixed_eval_results.json`

## Run-Level Outcomes

| run | n | plus% | delta plus pp | wrong->correct | correct->wrong | same pass | same fail | skipped | kept | mean mask frac | median edit ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dream_gate0.012_fresh_fixed | 378 | 72.2 | +0.0 | 0 | 0 | 273 | 105 | 338 | 40 | 0.0391 | 0.0000 |
| dream_gate0.012_offpolicy | 378 | 72.2 | +0.0 | 0 | 0 | 273 | 105 | 338 | 40 | 0.0391 | 0.0000 |
| dream_maskr0.01 | 378 | 72.5 | +0.3 | 2 | 1 | 272 | 103 | 0 | 0 | 0.0391 | 0.0000 |
| dream_t0.9 | 378 | 72.2 | +0.0 | 2 | 2 | 271 | 103 | 0 | 0 |  | 0.0000 |

## Candidate Category Counts

| category | n tasks | examples |
| --- | --- | --- |
| baseline_correct_no_observed_harm | 271 | Mbpp/100, Mbpp/104, Mbpp/105, Mbpp/106, Mbpp/108, Mbpp/109, Mbpp/111, Mbpp/116 |
| baseline_wrong_rewrite_ineffective | 103 | Mbpp/101, Mbpp/102, Mbpp/103, Mbpp/11, Mbpp/113, Mbpp/118, Mbpp/119, Mbpp/123 |
| gate_overconservative_candidate | 2 | Mbpp/406, Mbpp/809 |
| rewrite_stochastic_candidate | 2 | Mbpp/406, Mbpp/809 |
| gate_prevented_harm_candidate | 1 | Mbpp/432 |
| uncategorized | 1 | Mbpp/256 |

## Mask-Fraction Bins

| run | mask_fraction bin | n | wrong->correct | correct->wrong | same pass | same fail | net |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dream_gate0.012_fresh_fixed | [0.000, 0.008) | 10 | 0 | 0 | 4 | 6 | +0 |
| dream_gate0.012_fresh_fixed | [0.008, 0.010) | 13 | 0 | 0 | 8 | 5 | +0 |
| dream_gate0.012_fresh_fixed | [0.010, 0.012) | 17 | 0 | 0 | 8 | 9 | +0 |
| dream_gate0.012_fresh_fixed | [0.012, 0.015) | 17 | 0 | 0 | 10 | 7 | +0 |
| dream_gate0.012_fresh_fixed | [0.015, 0.020) | 34 | 0 | 0 | 23 | 11 | +0 |
| dream_gate0.012_fresh_fixed | [0.020, 0.030) | 60 | 0 | 0 | 43 | 17 | +0 |
| dream_gate0.012_fresh_fixed | [0.030, 0.050) | 116 | 0 | 0 | 82 | 34 | +0 |
| dream_gate0.012_fresh_fixed | [0.050, inf) | 111 | 0 | 0 | 95 | 16 | +0 |
| dream_gate0.012_offpolicy | [0.000, 0.008) | 10 | 0 | 0 | 4 | 6 | +0 |
| dream_gate0.012_offpolicy | [0.008, 0.010) | 13 | 0 | 0 | 8 | 5 | +0 |
| dream_gate0.012_offpolicy | [0.010, 0.012) | 17 | 0 | 0 | 8 | 9 | +0 |
| dream_gate0.012_offpolicy | [0.012, 0.015) | 17 | 0 | 0 | 10 | 7 | +0 |
| dream_gate0.012_offpolicy | [0.015, 0.020) | 34 | 0 | 0 | 23 | 11 | +0 |
| dream_gate0.012_offpolicy | [0.020, 0.030) | 60 | 0 | 0 | 43 | 17 | +0 |
| dream_gate0.012_offpolicy | [0.030, 0.050) | 116 | 0 | 0 | 82 | 34 | +0 |
| dream_gate0.012_offpolicy | [0.050, inf) | 111 | 0 | 0 | 95 | 16 | +0 |
| dream_maskr0.01 | [0.000, 0.008) | 10 | 0 | 0 | 4 | 6 | +0 |
| dream_maskr0.01 | [0.008, 0.010) | 13 | 0 | 0 | 8 | 5 | +0 |
| dream_maskr0.01 | [0.010, 0.012) | 17 | 0 | 0 | 8 | 9 | +0 |
| dream_maskr0.01 | [0.012, 0.015) | 17 | 0 | 0 | 10 | 7 | +0 |
| dream_maskr0.01 | [0.015, 0.020) | 34 | 0 | 0 | 23 | 11 | +0 |
| dream_maskr0.01 | [0.020, 0.030) | 60 | 0 | 0 | 43 | 17 | +0 |
| dream_maskr0.01 | [0.030, 0.050) | 116 | 1 | 1 | 81 | 33 | +0 |
| dream_maskr0.01 | [0.050, inf) | 111 | 1 | 0 | 95 | 15 | +1 |

## Edit Metrics By Transition

| run | transition | n | median char edit ratio | p90 char edit ratio | median changed lines | changed signature | changed imports | refined parse fails |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dream_gate0.012_fresh_fixed | same_fail | 105 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | 0 |
| dream_gate0.012_fresh_fixed | same_pass | 273 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | 0 |
| dream_gate0.012_offpolicy | same_fail | 105 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | 0 |
| dream_gate0.012_offpolicy | same_pass | 273 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | 0 |
| dream_maskr0.01 | correct_to_wrong | 1 | 0.0189 | 0.0189 | 1.00 | 0 | 0 | 0 |
| dream_maskr0.01 | same_fail | 103 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | 0 |
| dream_maskr0.01 | same_pass | 272 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | 0 |
| dream_maskr0.01 | wrong_to_correct | 2 | 0.0179 | 0.0231 | 1.00 | 0 | 0 | 0 |
| dream_t0.9 | correct_to_wrong | 2 | 0.0113 | 0.0173 | 1.00 | 1 | 0 | 0 |
| dream_t0.9 | same_fail | 103 | 0.0000 | 0.0017 | 0.00 | 1 | 0 | 0 |
| dream_t0.9 | same_pass | 271 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | 0 |
| dream_t0.9 | wrong_to_correct | 2 | 0.0179 | 0.0231 | 1.00 | 0 | 0 | 0 |

## Deterministic Case Candidates

| category | task | wrong->correct runs | correct->wrong runs | median mask frac | median edit ratio | evidence |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_correct_no_observed_harm | Mbpp/100 | 0 | 0 | 0.0286 | 0.0000 | dream_gate0.012_fresh_fixed:same_pass/skipped; dream_gate0.012_offpolicy:same_pass/skipped; dream_maskr0.01:same_pass/no_gate; dream_t0.9:same_pass/no_gate |
| baseline_correct_no_observed_harm | Mbpp/104 | 0 | 0 | 0.0455 | 0.0000 | dream_gate0.012_fresh_fixed:same_pass/skipped; dream_gate0.012_offpolicy:same_pass/skipped; dream_maskr0.01:same_pass/no_gate; dream_t0.9:same_pass/no_gate |
| baseline_correct_no_observed_harm | Mbpp/105 | 0 | 0 | 0.1111 | 0.0000 | dream_gate0.012_fresh_fixed:same_pass/skipped; dream_gate0.012_offpolicy:same_pass/skipped; dream_maskr0.01:same_pass/no_gate; dream_t0.9:same_pass/no_gate |
| baseline_correct_no_observed_harm | Mbpp/106 | 0 | 0 | 0.0714 | 0.0000 | dream_gate0.012_fresh_fixed:same_pass/skipped; dream_gate0.012_offpolicy:same_pass/skipped; dream_maskr0.01:same_pass/no_gate; dream_t0.9:same_pass/no_gate |
| baseline_correct_no_observed_harm | Mbpp/108 | 0 | 0 | 0.0345 | 0.0000 | dream_gate0.012_fresh_fixed:same_pass/skipped; dream_gate0.012_offpolicy:same_pass/skipped; dream_maskr0.01:same_pass/no_gate; dream_t0.9:same_pass/no_gate |
| baseline_correct_no_observed_harm | Mbpp/109 | 0 | 0 | 0.0156 | 0.0000 | dream_gate0.012_fresh_fixed:same_pass/skipped; dream_gate0.012_offpolicy:same_pass/skipped; dream_maskr0.01:same_pass/no_gate; dream_t0.9:same_pass/no_gate |
| baseline_correct_no_observed_harm | Mbpp/111 | 0 | 0 | 0.0200 | 0.0000 | dream_gate0.012_fresh_fixed:same_pass/skipped; dream_gate0.012_offpolicy:same_pass/skipped; dream_maskr0.01:same_pass/no_gate; dream_t0.9:same_pass/no_gate |
| baseline_correct_no_observed_harm | Mbpp/116 | 0 | 0 | 0.0556 | 0.0000 | dream_gate0.012_fresh_fixed:same_pass/skipped; dream_gate0.012_offpolicy:same_pass/skipped; dream_maskr0.01:same_pass/no_gate; dream_t0.9:same_pass/no_gate |
| baseline_wrong_rewrite_ineffective | Mbpp/101 | 0 | 0 | 0.0303 | 0.0000 | dream_gate0.012_fresh_fixed:same_fail/skipped; dream_gate0.012_offpolicy:same_fail/skipped; dream_maskr0.01:same_fail/no_gate; dream_t0.9:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | Mbpp/102 | 0 | 0 | 0.0278 | 0.0000 | dream_gate0.012_fresh_fixed:same_fail/skipped; dream_gate0.012_offpolicy:same_fail/skipped; dream_maskr0.01:same_fail/no_gate; dream_t0.9:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | Mbpp/103 | 0 | 0 | 0.0116 | 0.0000 | dream_gate0.012_fresh_fixed:same_fail/kept; dream_gate0.012_offpolicy:same_fail/kept; dream_maskr0.01:same_fail/no_gate; dream_t0.9:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | Mbpp/11 | 0 | 0 | 0.0238 | 0.0000 | dream_gate0.012_fresh_fixed:same_fail/skipped; dream_gate0.012_offpolicy:same_fail/skipped; dream_maskr0.01:same_fail/no_gate; dream_t0.9:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | Mbpp/113 | 0 | 0 | 0.0435 | 0.0000 | dream_gate0.012_fresh_fixed:same_fail/skipped; dream_gate0.012_offpolicy:same_fail/skipped; dream_maskr0.01:same_fail/no_gate; dream_t0.9:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | Mbpp/118 | 0 | 0 | 0.0909 | 0.0000 | dream_gate0.012_fresh_fixed:same_fail/skipped; dream_gate0.012_offpolicy:same_fail/skipped; dream_maskr0.01:same_fail/no_gate; dream_t0.9:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | Mbpp/119 | 0 | 0 | 0.0116 | 0.0000 | dream_gate0.012_fresh_fixed:same_fail/kept; dream_gate0.012_offpolicy:same_fail/kept; dream_maskr0.01:same_fail/no_gate; dream_t0.9:same_fail/no_gate |
| baseline_wrong_rewrite_ineffective | Mbpp/123 | 0 | 0 | 0.0097 | 0.0000 | dream_gate0.012_fresh_fixed:same_fail/kept; dream_gate0.012_offpolicy:same_fail/kept; dream_maskr0.01:same_fail/no_gate; dream_t0.9:same_fail/no_gate |
| gate_overconservative_candidate | Mbpp/406 | 2 | 0 | 0.0667 | 0.0122 | dream_gate0.012_fresh_fixed:same_fail/skipped; dream_gate0.012_offpolicy:same_fail/skipped; dream_maskr0.01:wrong_to_correct/no_gate; dream_t0.9:wrong_to_correct/no_gate |
| gate_overconservative_candidate | Mbpp/809 | 2 | 0 | 0.0357 | 0.0057 | dream_gate0.012_fresh_fixed:same_fail/skipped; dream_gate0.012_offpolicy:same_fail/skipped; dream_maskr0.01:wrong_to_correct/no_gate; dream_t0.9:wrong_to_correct/no_gate |
| gate_prevented_harm_candidate | Mbpp/432 | 0 | 2 | 0.0455 | 0.0094 | dream_gate0.012_fresh_fixed:same_pass/skipped; dream_gate0.012_offpolicy:same_pass/skipped; dream_maskr0.01:correct_to_wrong/no_gate; dream_t0.9:correct_to_wrong/no_gate |
| rewrite_stochastic_candidate | Mbpp/406 | 2 | 0 | 0.0667 | 0.0122 | dream_gate0.012_fresh_fixed:same_fail/skipped; dream_gate0.012_offpolicy:same_fail/skipped; dream_maskr0.01:wrong_to_correct/no_gate; dream_t0.9:wrong_to_correct/no_gate |
| rewrite_stochastic_candidate | Mbpp/809 | 2 | 0 | 0.0357 | 0.0057 | dream_gate0.012_fresh_fixed:same_fail/skipped; dream_gate0.012_offpolicy:same_fail/skipped; dream_maskr0.01:wrong_to_correct/no_gate; dream_t0.9:wrong_to_correct/no_gate |

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
