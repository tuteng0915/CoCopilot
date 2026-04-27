# Spec: Seed-Coder-Instruct Conservative Remask

## Goal

Make CoCoder less harmful for `ByteDance-Seed/Seed-Coder-8B-Instruct` while keeping a clean pass@1 setup:

- no pass@2 / oracle selection
- no post-hoc original-vs-refined accept/reject using test execution
- no runtime feedback loop
- no syntax-token protection heuristics in this round

The target failure mode is high `correct -> wrong` churn where that churn is true raw-code damage. After `pkgv2` normalization, the strongest evidence is LLaDA MBPP and the smaller LLaDA HumanEval degradation; Dream HumanEval is no-op under current packaging.

## Status Snapshot

Updated: 2026-04-20.

Current state:

- Seed-Coder-Instruct baseline and `tau=0.9` Dream/LLaDA pair results are complete and registered in the result tooling.
- `docs/results.md` now uses EvalPlus packaging-normalized (`pkgv2`) HumanEval results for Seed-Coder-Instruct; MBPP packaging normalization changed `0/378` records, so the original MBPP evaluations remain valid.
- Conservative remask support is implemented in `src/coder/scripts/gen_remask.py`.
- Low-disagreement gating prevents the true `tau=0.9` regressions that remain after packaging normalization, most clearly for LLaDA HumanEval and LLaDA MBPP.
- The previously observed off-policy Dream HumanEval lift (`+2.4pp`) and the apparent Dream HumanEval degradation were packaging/eval artifacts, not stable raw-code rewriter effects.
- Current Dream HumanEval artifacts are effectively no-op under `pkgv2`: baseline, `tau=0.9`, `maskr0.01`, materialized `gate0.012`, and fresh fixed `gate0.012` all evaluate to `124/164` plus passes (`75.6%`).

Current claim:

> For a strong AR drafter, the main validated role of the current remask gate is harm prevention, not pass@1 improvement. Under consistent packaging, Dream HumanEval provides no stable improvement/degradation signal; LLaDA HumanEval and LLaDA MBPP still show true `tau=0.9` damage, and the low-disagreement gate restores baseline behavior.

Packaging-normalized claim:

> Under a consistent EvalPlus packaging policy, Seed-Coder-Instruct Dream HumanEval baseline, `tau=0.9`, `maskr0.01`, materialized `gate0.012`, and fresh fixed `gate0.012` all have the same plus pass count (`124/164`, 75.6%). The previous HumanEval transition counts should be treated as packaging/eval artifacts until rerun from freshly normalized samples.
> LLaDA HumanEval still has true raw-code degradation at `tau=0.9` (`119/164`, 72.6%), and the same low-disagreement gate restores the normalized baseline (`124/164`, 75.6%).

Implementation changes so far:

- Added `--record_mask_stats`, `--record_mask_spans`, `--gate_min_mask_fraction`, and `--gate_max_mask_fraction`.
- Added shared pre-generation mask planning so gate decisions use only locator/refiner confidence and planned edit size.
- Recorded `draft_tokens`, `mask_tokens`, `mask_fraction`, confidence stats, `skip_refine`, and `skip_reason` in `gen` metadata.
- Centralized EvalPlus packaging in `build_evalplus_solution()`: full-function completions keep prompt imports, target top-level functions are extracted with AST when possible, and nested helper functions are preserved.
- Fixed EvalPlus packaging for skipped samples: skipped EvalPlus records now preserve the input `solution` when available.
- Clarified locator logging so LLaDA/Dream self-locator runs do not print a misleading Dream-only message.

Important artifacts:

| Artifact | Path |
|---|---|
| Gate sweep CSV | `outputs/base_tuteng/seed_coder_instruct_dream_gate_sweep.csv` |
| Dream fresh gated HE | `outputs/base_tuteng/seed-coder-instruct_dream_remask_humaneval_maskr0.01_gate0.012_fresh_fixed_summary.json` |
| Dream fresh gated MBPP | `outputs/base_tuteng/seed-coder-instruct_dream_remask_mbpp_maskr0.01_gate0.012_fresh_fixed_summary.json` |
| LLaDA gated HE | `outputs/base_tuteng/seed-coder-instruct_llada_remask_humaneval_maskr0.01_gate0.012_fixed_summary.json` |
| LLaDA gated MBPP | `outputs/base_tuteng/seed-coder-instruct_llada_remask_mbpp_maskr0.01_gate0.012_fixed_summary.json` |
| HE packaging-normalized analysis | `docs/analysis/seed_coder_instruct_locator_rewriter_case_study_pkgv2.md` |
| MBPP packaging-normalized analysis | `docs/analysis/seed_coder_instruct_mbpp_locator_rewriter_case_study_pkgv2.md` |
| LLaDA HE packaging-normalized analysis | `docs/analysis/seed_coder_instruct_llada_humaneval_locator_rewriter_case_study_pkgv2.md` |
| LLaDA MBPP packaging-normalized analysis | `docs/analysis/seed_coder_instruct_llada_mbpp_locator_rewriter_case_study_pkgv2.md` |
| Packaging-normalized samples/evals | `outputs/base_tuteng/packaging_v2/` |

## Current Baseline And Tau=0.9 Results

These are the current numbers to cite. They match `docs/results.md`: HumanEval uses EvalPlus packaging-normalized `pkgv2`; MBPP is unchanged because normalization changed no records.

Current Seed-Coder-Instruct standalone:

| Dataset | plus% | base% |
|---|---:|---:|
| HumanEval+ | 75.6 | 81.1 |
| MBPP+ | 72.2 | 84.9 |

Current `tau=0.9` remask:

| Dataset | Refiner | plus% | Delta vs current baseline | wrong->correct | correct->wrong |
|---|---|---:|---:|---:|---:|
| HumanEval+ | Dream | 75.6 | +0.0pp | 0 | 0 |
| HumanEval+ | LLaDA | 72.6 | -3.0pp | 0 | 5 |
| MBPP+ | Dream | 72.2 | +0.0pp | 2 | 2 |
| MBPP+ | LLaDA | 64.6 | -7.6pp | 1 | 30 |

Current interpretation:

- Dream HumanEval no longer supports either a benefit or a harm claim after `pkgv2`; old `65.2%`/`70.1%` HumanEval numbers are historical and superseded.
- LLaDA HumanEval remains a true harm case (`5` normalized `correct -> wrong` transitions), and the gate restores baseline.
- MBPP is the cleaner raw-code case-study source because packaging normalization had zero effect there; LLaDA MBPP is the strongest gate-safety example.
- There is still no stable pass@1 improvement claim for Seed-Coder-Instruct conservative remask.

## Phase A: Conservative Mask-Strength Sweep

Use `mask_ratio` instead of a fixed confidence threshold. This avoids assuming that the dLLM confidence scale is calibrated across different AR drafters.

The original sweep started with Dream because it looked more stable in the pre-pkgv2 `tau=0.9` run. For new analysis under the current conclusions, prioritize LLaDA MBPP and LLaDA HumanEval because those are the runs with true raw-code harm after normalization.

Recommended sweep:

```bash
for ratio in 0.01 0.02 0.03 0.05; do
  CUDA_VISIBLE_DEVICES=<gpu> python -m coder.scripts.gen_remask \
    --refiner dream \
    --input outputs/base_tuteng/seed-coder-instruct_humaneval.jsonl \
    --out outputs/base_tuteng/seed-coder-instruct_dream_remask_humaneval_maskr${ratio}.jsonl \
    --mask_ratio "$ratio" \
    --record_mask_stats \
    --device cuda:0

  python -m coder.scripts.eval_evalplus \
    --backend local \
    --samples outputs/base_tuteng/seed-coder-instruct_dream_remask_humaneval_maskr${ratio}.jsonl \
    --dataset humaneval \
    --summary_model seed_coder_instruct_dream_humaneval_maskr${ratio} \
    --summary_out outputs/base_tuteng/seed-coder-instruct_dream_remask_humaneval_maskr${ratio}_summary.json
done
```

For any rerun intended to support current claims, evaluate HumanEval through `pkgv2` and include MBPP because it is packaging-clean.

Acceptance signal:

- primary: plus% not below baseline
- diagnostic: `correct -> wrong` must drop sharply relative to `tau=0.9`
- secondary: non-compilable count should not increase

## Phase B: Pre-Refinement Gate

Add a pre-refinement gate using only locator-side information before rewriting. This is not a verifier and does not inspect test outcomes.

Policy:

- compute the actual planned mask fraction after granularity expansion
- skip refinement if `mask_fraction < min_mask_fraction`
- skip refinement if `mask_fraction > max_mask_fraction`

Example:

```bash
CUDA_VISIBLE_DEVICES=<gpu> python -m coder.scripts.gen_remask \
  --refiner dream \
  --input outputs/base_tuteng/seed-coder-instruct_humaneval.jsonl \
  --out outputs/base_tuteng/seed-coder-instruct_dream_remask_humaneval_maskr0.03_gate1-6.jsonl \
  --mask_ratio 0.03 \
  --gate_min_mask_fraction 0.01 \
  --gate_max_mask_fraction 0.06 \
  --record_mask_stats \
  --device cuda:0
```

For fixed token-level `mask_ratio`, the gate is more useful once `span` or `line` granularity expands masks. It is still useful as instrumentation and as a common interface for threshold-based variants.

## Phase C: Granularity Sweep

Using the best conservative strength from Phase A, compare:

- `--mask_granularity token`
- `--mask_granularity span --span_merge_gap 1`
- `--mask_granularity line`

Suggested commands:

```bash
CUDA_VISIBLE_DEVICES=<gpu> python -m coder.scripts.gen_remask \
  --refiner dream \
  --input outputs/base_tuteng/seed-coder-instruct_humaneval.jsonl \
  --out outputs/base_tuteng/seed-coder-instruct_dream_remask_humaneval_maskr0.03_span_g1.jsonl \
  --mask_ratio 0.03 \
  --mask_granularity span \
  --span_merge_gap 1 \
  --gate_max_mask_fraction 0.08 \
  --record_mask_stats \
  --device cuda:0
```

Line-level should be run last; it can easily expand a tiny token mask into a large rewrite.

## Required Diagnostics

For every completed run, report:

- plus/base pass@1
- delta vs Seed-Coder-Instruct baseline
- `wrong -> correct`
- `correct -> wrong`
- non-compilable count before/after sanitize if available
- mean / median / max `gen.mask_fraction`
- number of skipped samples when gate is enabled

Transition counter:

```bash
python3 - <<'PY'
import json

base_path = "outputs/base_tuteng/seed-coder-instruct_humaneval_eval_results.json"
run_path = "outputs/base_tuteng/<run>_eval_results.json"

base = json.load(open(base_path))["eval"]
run = json.load(open(run_path))["eval"]

def plus_pass(eval_map, task_id):
    return any((r.get("plus_status") or "").lower() == "pass"
               for r in eval_map.get(task_id, []))

ids = sorted(set(base) | set(run))
base_pass = {tid for tid in ids if plus_pass(base, tid)}
run_pass = {tid for tid in ids if plus_pass(run, tid)}
print("same_pass", len(base_pass & run_pass))
print("wrong_to_correct", len(run_pass - base_pass))
print("correct_to_wrong", len(base_pass - run_pass))
print("same_fail", len(set(ids) - base_pass - run_pass))
PY
```

Mask-stat counter:

```bash
python3 - <<'PY'
import json, statistics

path = "outputs/base_tuteng/<run>.jsonl"
vals = []
skips = 0
with open(path) as f:
    for line in f:
        rec = json.loads(line)
        gen = rec.get("gen") or {}
        if gen.get("skip_refine"):
            skips += 1
        if gen.get("mask_fraction") is not None:
            vals.append(float(gen["mask_fraction"]))

print("n", len(vals), "skips", skips)
print("mean", statistics.mean(vals) if vals else None)
print("median", statistics.median(vals) if vals else None)
print("max", max(vals) if vals else None)
PY
```

## Method Interpretation

The expected win is not a large increase in fixed cases. The main validated target is to reduce broken cases where raw-code damage exists. A useful conservative setting should:

- preserve most baseline-correct samples
- keep the small number of fixed cases
- avoid the `tau=0.9` pattern where fixes are outweighed by regressions, as seen most clearly in LLaDA HumanEval and LLaDA MBPP

## Legacy Progress Log (pre-pkgv2 unless noted)

The rows in this section are kept as experiment history. HumanEval numbers here were produced before the full `pkgv2` packaging normalization and must not be cited as current conclusions. MBPP rows remain valid because MBPP normalization changed `0/378` records.

Implemented in `src/coder/scripts/gen_remask.py`:

- `--record_mask_stats`
- `--gate_min_mask_fraction`
- `--gate_max_mask_fraction`
- shared pre-generation mask planning so gate decisions use only locator/refiner confidence, not test feedback

Completed Dream `mask_ratio` sweep for Seed-Coder-Instruct under the pre-pkgv2 HumanEval packaging:

| Run | HE+ | HE base | HE fixed | HE broken | MBPP+ | MBPP base | MBPP fixed | MBPP broken |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 70.1 | 75.0 | - | - | 72.2 | 84.9 | - | - |
| maskr0.01 | 65.2 | 70.7 | 10 | 18 | 72.5 | 85.2 | 2 | 1 |
| maskr0.03 | 65.2 | 70.7 | 10 | 18 | 72.5 | 85.2 | 2 | 1 |
| maskr0.05 | 65.2 | 70.7 | 10 | 18 | 72.5 | 85.2 | 2 | 1 |

Historical conclusion from the pre-pkgv2 table: lower token-level `mask_ratio` did not control apparent HumanEval churn. Current `pkgv2` conclusion supersedes this for HumanEval: Dream `maskr0.01` is a no-op relative to the normalized baseline (`75.6%`, `0/0` transitions).

Pre-refinement gate check, before full `pkgv2` normalization:

| Run | HE+ | HE base | HE fixed | HE broken | HE skipped | MBPP+ | MBPP base | MBPP fixed | MBPP broken | MBPP skipped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| maskr0.01 + gate_max0.015 | 72.0 | 77.4 | 7 | 4 | 101 | 72.2 | 84.9 | 0 | 0 | 321 |

The gated artifact was materialized from the completed `maskr0.01` output plus the original baseline drafts because GPU permission for a fresh Dream run was unavailable. This is behavior-equivalent for this gate: samples with `mask_fraction <= 0.015` keep the prior refinement, and samples above the gate use the original draft.

Current recommendation after `pkgv2` reanalysis:

- Treat `mask_ratio=0.01, gate_max_mask_fraction=0.012` as a safety setting for LLaDA HumanEval/MBPP and as a conservative no-op baseline for Dream HumanEval, not as an improvement setting.
- Use the fixed EvalPlus packaging path for every HumanEval comparison; old HumanEval off-policy sweep numbers are packaging-confounded.
- Use MBPP, especially LLaDA MBPP, as the cleaner raw-code-edit case-study source.
- Record mask spans (`--record_mask_spans`) for any new locator/rewriter analysis, then test stability across seeds or deterministic decoding before claiming a pass@1 gain.

## Legacy Expanded Gate Sweep (pre-pkgv2 HumanEval)

This sweep is useful for understanding why low-disagreement gating looked promising, but the HumanEval lift in this section is superseded by the `pkgv2` reanalysis. Treat HumanEval rows below as historical diagnostics; MBPP rows remain valid.

Completed an off-policy gate sweep over:

- datasets: HumanEval, MBPP
- Dream token-level `mask_ratio`: `0.01, 0.03, 0.05`
- `gate_max_mask_fraction`: `0.0, 0.008, 0.010, 0.012, 0.015, 0.018, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.060, 0.080, 0.100, 1.0`

Full CSV:

- `outputs/base_tuteng/seed_coder_instruct_dream_gate_sweep.csv`

Historical best points from the sweep:

| Dataset | mask_ratio | gate_max | plus% | base% | Delta plus | fixed | broken | kept/skipped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HumanEval | 0.01 | 0.012 | 72.6 | 78.0 | +2.4pp | 7 | 3 | 34/130 |
| HumanEval | 0.01 | 0.015 | 72.0 | 77.4 | +1.8pp | 7 | 4 | 63/101 |
| HumanEval | 0.01 | 0.018 | 71.3 | 77.4 | +1.2pp | 9 | 7 | 88/76 |
| MBPP | 0.01 | 0.012 | 72.2 | 84.9 | +0.0pp | 0 | 0 | 40/338 |
| MBPP | 0.01 | 0.045 | 72.5 | 85.2 | +0.3pp | 1 | 0 | 234/144 |

Representative materialized artifacts were evaluated with the then-current EvalPlus packaging and matched the sweep:

- `seed-coder-instruct_dream_remask_humaneval_maskr0.01_gate0.012`: HE+ 72.6, base 78.0
- `seed-coder-instruct_dream_remask_humaneval_maskr0.01_gate0.018`: HE+ 71.3, base 77.4
- `seed-coder-instruct_dream_remask_mbpp_maskr0.01_gate0.012`: MBPP+ 72.2, base 84.9
- `seed-coder-instruct_dream_remask_mbpp_maskr0.01_gate0.045`: MBPP+ 72.5, base 85.2

HumanEval `mask_ratio=0.01` mask-fraction bins:

| mask_fraction bin | n | fixed | broken | net |
|---|---:|---:|---:|---:|
| `[0.000, 0.012)` | 34 | 7 | 3 | +4 |
| `[0.012, 0.015)` | 29 | 0 | 1 | -1 |
| `[0.015, 0.020)` | 34 | 2 | 5 | -3 |
| `[0.020, 0.030)` | 30 | 0 | 4 | -4 |
| `[0.030, 0.050)` | 27 | 1 | 4 | -3 |
| `[0.050, 1.000]` | 10 | 0 | 1 | -1 |

Revised interpretation of this legacy sweep:

- The apparent HumanEval repair concentration below about `1.2%` planned mask fraction was not stable under fresh generation and then disappeared under full `pkgv2` normalization.
- The MBPP part of the sweep remains packaging-clean. It still does not show a strong Dream repair signal: strict gates preserve baseline, and the looser `0.045` MBPP-only point is only a small materialized-sweep observation.
- Current conclusion: disagreement gating is a reliable harm-reduction mechanism where true raw-code harm exists, especially LLaDA MBPP, but it is not a reliable improvement mechanism.

## Fresh Validation Before Full pkgv2 Normalization (historical)

Fresh GPU reruns exposed an EvalPlus packaging issue in `gen_remask.py`: when a skipped sample kept the original draft, the script still rebuilt `solution` from `raw_completion`. If the completion started with `def`, prompt imports such as `from typing import List` could be dropped. This made some skipped samples fail even though refinement was skipped.

Fix applied:

- `build_evalplus_solution()` now preserves prompt imports for full-function completions.
- skipped EvalPlus samples preserve the input `solution` when available.

Fresh gated results after the first packaging fix, before the later full `pkgv2` normalization:

| Run | Dataset | plus% | base% | Delta plus | fixed | broken | skipped |
|---|---|---:|---:|---:|---:|---:|---:|
| Dream `maskr0.01 + gate0.012` fresh fixed | HumanEval | 70.1 | 75.0 | +0.0pp | 0 | 0 | 130 |
| Dream `maskr0.01 + gate0.012` fresh fixed | MBPP | 72.2 | 84.9 | +0.0pp | 0 | 0 | 338 |
| LLaDA `maskr0.01 + gate0.012` fixed | HumanEval | 70.1 | 75.0 | +0.0pp | 0 | 0 | 121 |
| LLaDA `maskr0.01 + gate0.012` fixed | MBPP | 72.2 | 84.9 | +0.0pp | 0 | 0 | 333 |

At this historical stage, compared with the harmful pre-pkgv2 `tau=0.9` runs, the gate looked clearly safer:

| Run | Dataset | plus% | Delta plus | fixed | broken |
|---|---|---:|---:|---:|---:|
| Dream `tau=0.9` | HumanEval | 65.2 | -4.9pp | 10 | 18 |
| Dream gated fresh fixed | HumanEval | 70.1 | +0.0pp | 0 | 0 |
| LLaDA `tau=0.9` | HumanEval | 62.8 | -7.3pp | 10 | 22 |
| LLaDA gated fixed | HumanEval | 70.1 | +0.0pp | 0 | 0 |
| LLaDA `tau=0.9` | MBPP | 64.6 | -7.6pp | 1 | 30 |
| LLaDA gated fixed | MBPP | 72.2 | +0.0pp | 0 | 0 |

Historical conclusion at this stage: low-disagreement gating prevented degradation on fresh runs, but the observed `+2.4pp` HumanEval gain was tied to one previous stochastic Dream sample and should not be reported as stable. The next `pkgv2` section supersedes the HumanEval numbers here: Dream HumanEval is no-op, while LLaDA HumanEval keeps a smaller true harm signal (`0/5` transitions at `tau=0.9`) that the gate prevents.

## Packaging-Normalized Reanalysis

This is the authoritative current analysis for HumanEval and supersedes the HumanEval numbers in the legacy sections above.

A follow-up case-study analyzer separated raw-code changes from EvalPlus `solution` packaging changes:

- Analyzer: `src/coder/scripts/analyze_remask_case_study.py`
- Packaging normalizer: `src/coder/scripts/normalize_evalplus_packaging.py`
- HumanEval report: `docs/analysis/seed_coder_instruct_locator_rewriter_case_study_pkgv2.md`
- MBPP report: `docs/analysis/seed_coder_instruct_mbpp_locator_rewriter_case_study_pkgv2.md`
- LLaDA HumanEval report: `docs/analysis/seed_coder_instruct_llada_humaneval_locator_rewriter_case_study_pkgv2.md`
- LLaDA MBPP report: `docs/analysis/seed_coder_instruct_llada_mbpp_locator_rewriter_case_study_pkgv2.md`
- Normalized outputs: `outputs/base_tuteng/packaging_v2/`

The normalizer rebuilds every EvalPlus `solution` from `prompt` and `raw_completion` with one policy:

- keep prompt imports for full-function completions
- extract the target top-level function with AST when possible
- preserve nested helper functions instead of truncating at an indented `def`

Packaging changes affected HumanEval but not MBPP:

| Dataset / Run family | Packaging-changed records |
|---|---:|
| HumanEval baseline | 14 / 164 |
| HumanEval Dream `tau=0.9` | 24 / 164 |
| HumanEval Dream `maskr0.01` | 24 / 164 |
| HumanEval Dream `gate0.012` materialized | 10 / 164 |
| HumanEval Dream `gate0.012` fresh fixed | 14 / 164 |
| HumanEval LLaDA `tau=0.9` | 24 / 164 |
| HumanEval LLaDA `gate0.012` fixed | 14 / 164 |
| MBPP normalized runs | 0 / 378 |

After re-evaluating the normalized HumanEval samples:

| Run | plus pass | plus% | base pass | base% | wrong->correct | correct->wrong |
|---|---:|---:|---:|---:|---:|---:|
| Seed-Coder baseline pkgv2 | 124 / 164 | 75.6 | 133 / 164 | 81.1 | - | - |
| Dream `tau=0.9` pkgv2 | 124 / 164 | 75.6 | 134 / 164 | 81.7 | 0 | 0 |
| Dream `maskr0.01` pkgv2 | 124 / 164 | 75.6 | 134 / 164 | 81.7 | 0 | 0 |
| Dream `gate0.012` materialized pkgv2 | 124 / 164 | 75.6 | 133 / 164 | 81.1 | 0 | 0 |
| Dream `gate0.012` fresh fixed pkgv2 | 124 / 164 | 75.6 | 133 / 164 | 81.1 | 0 | 0 |
| LLaDA `tau=0.9` pkgv2 | 119 / 164 | 72.6 | 127 / 164 | 77.4 | 0 | 5 |
| LLaDA `gate0.012` fixed pkgv2 | 124 / 164 | 75.6 | 133 / 164 | 81.1 | 0 | 0 |

For MBPP, packaging normalization changed no records, so the original MBPP eval files remain valid under pkgv2 JSONL. LLaDA MBPP remains the clearest gate-safety case:

| Run | plus% | wrong->correct | correct->wrong | Gate-prevented harm candidates |
|---|---:|---:|---:|---:|
| LLaDA `tau=0.9` MBPP pkgv2 | 64.6 | 1 | 30 | - |
| LLaDA `gate0.012` MBPP pkgv2 | 72.2 | 0 | 0 | 29 |

Interpretation:

- The old HumanEval `+2.4pp` off-policy lift was not a reliable rewriter signal.
- The old HumanEval `correct -> wrong` churn was also largely packaging-induced, not raw-code damage.
- Under consistent packaging, current Dream remask artifacts are mostly no-op on HumanEval.
- LLaDA still shows real HumanEval raw-code harm at `tau=0.9` (`5` true `correct -> wrong`), and the low-disagreement gate prevents those cases under the normalized policy.
- LLaDA MBPP shows the same pattern more strongly: `30` true `correct -> wrong` cases at `tau=0.9`, with `29` gate-prevented harm candidates after `gate0.012`.
- MBPP remains the cleaner raw-code-edit case-study source because packaging normalization changed no records there.
- Future runs should use the fixed packaging path and `--record_mask_spans` if they are intended for locator/rewriter case studies.

## Next Steps

1. Keep `pkgv2` as the evaluation contract.

   Any new HumanEval run should be normalized and evaluated through the shared `build_evalplus_solution()` path before conclusions are written. Do not compare old pre-pkgv2 HumanEval summaries with new outputs.

2. Build the locator/rewriter case study from statistical categories.

   Use `analyze_remask_case_study.py` to stratify examples into true harm, gate-prevented harm, overconservative skip, no-op rewrite, packaging artifact, and off-policy artifact categories. Prefer MBPP and LLaDA MBPP for raw-code evidence because MBPP had zero packaging changes and LLaDA MBPP has the strongest true harm signal (`30` `correct -> wrong` at `tau=0.9`).

3. Record spans before selecting examples.

   New runs intended for analysis should use `--record_mask_spans`, `--max_recorded_mask_spans`, and `--mask_span_context_chars`. Manual examples should be sampled from the analyzer's category CSVs, not hand-picked from successful anecdotes.

4. Test a stricter edit policy.

   Current diffusion refinement can still rewrite the masked position in a way that collapses to no effective repair or introduces noise. Try a one-token or minimal-span policy and record edit distance, so the method is closer to "surgical repair" than "small-region stochastic rewrite".

5. Separate safety gate from improvement mechanism.

   Keep `mask_ratio=0.01, gate_max_mask_fraction=0.012` as the safety baseline. Any improvement method should be compared against both Seed-Coder standalone and this safety-gated baseline, and must show gains under `pkgv2` on HumanEval or on packaging-clean MBPP.

6. Sweep gate threshold only after stability is measured.

   The legacy off-policy sweep suggests `0.012` is the safest shared threshold and `0.045` can recover a small MBPP-only materialized gain. Do not promote either as an improvement setting until fresh multi-seed runs reproduce it under the current packaging/evaluation path.

7. Run granularity ablations last.

   Span/line granularity will expand planned edit size, so it should use the same `mask_fraction` gate and report actual skipped/kept counts. Start with `span --span_merge_gap 1`; line-level is likely too aggressive for Seed-Coder-Instruct.

8. Optional efficiency follow-up.

   The current gate still scores every draft with the refiner before skipping. If the gate becomes a standard safety mechanism, consider a cheaper locator-only path or cached mask plans to avoid paying full refiner setup cost for mostly skipped runs.
