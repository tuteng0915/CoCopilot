# Spec: Seed-Coder-Instruct Conservative Remask

## Goal

Make CoCoder less harmful for `ByteDance-Seed/Seed-Coder-8B-Instruct` while keeping a clean pass@1 setup:

- no pass@2 / oracle selection
- no post-hoc original-vs-refined accept/reject using test execution
- no runtime feedback loop
- no syntax-token protection heuristics in this round

The target failure mode is high `correct -> wrong` churn. Existing `tau=0.9` results show the refiner can fix some tasks, but breaks too many already-correct drafts.

## Status Snapshot

Updated: 2026-04-17.

Current state:

- Seed-Coder-Instruct baseline and `tau=0.9` Dream/LLaDA pair results are complete and registered in the result tooling.
- Conservative remask support is implemented in `src/coder/scripts/gen_remask.py`.
- Low-disagreement gating reliably prevents the large Seed-Coder-Instruct regressions seen with `tau=0.9`.
- The previously observed off-policy Dream HumanEval lift (`+2.4pp`) did not reproduce in a fresh run, so it should not be reported as a stable pass@1 improvement.

Current claim:

> For a strong AR drafter, large dLLM-proposed edit regions are more likely to be harmful disagreement than reliable repair opportunities. A pre-refinement disagreement gate can reduce `correct -> wrong` churn and preserve baseline accuracy, but further stability work is needed before claiming an improvement.

Implementation changes so far:

- Added `--record_mask_stats`, `--gate_min_mask_fraction`, and `--gate_max_mask_fraction`.
- Added shared pre-generation mask planning so gate decisions use only locator/refiner confidence and planned edit size.
- Recorded `draft_tokens`, `mask_tokens`, `mask_fraction`, confidence stats, `skip_refine`, and `skip_reason` in `gen` metadata.
- Fixed EvalPlus packaging for skipped samples: skipped EvalPlus records now preserve the input `solution` when available, and full-function completions keep prompt imports.
- Clarified locator logging so LLaDA/Dream self-locator runs do not print a misleading Dream-only message.

Important artifacts:

| Artifact | Path |
|---|---|
| Gate sweep CSV | `outputs/base_tuteng/seed_coder_instruct_dream_gate_sweep.csv` |
| Dream fresh gated HE | `outputs/base_tuteng/seed-coder-instruct_dream_remask_humaneval_maskr0.01_gate0.012_fresh_fixed_summary.json` |
| Dream fresh gated MBPP | `outputs/base_tuteng/seed-coder-instruct_dream_remask_mbpp_maskr0.01_gate0.012_fresh_fixed_summary.json` |
| LLaDA gated HE | `outputs/base_tuteng/seed-coder-instruct_llada_remask_humaneval_maskr0.01_gate0.012_fixed_summary.json` |
| LLaDA gated MBPP | `outputs/base_tuteng/seed-coder-instruct_llada_remask_mbpp_maskr0.01_gate0.012_fixed_summary.json` |

## Baseline To Beat

Current Seed-Coder-Instruct standalone:

| Dataset | plus% | base% |
|---|---:|---:|
| HumanEval+ | 70.1 | 75.0 |
| MBPP+ | 72.2 | 84.9 |

Current `tau=0.9` remask:

| Dataset | Refiner | plus% | Delta |
|---|---|---:|---:|
| HumanEval+ | Dream | 65.2 | -4.9pp |
| HumanEval+ | LLaDA | 62.8 | -7.3pp |
| MBPP+ | Dream | 72.2 | +0.0pp |
| MBPP+ | LLaDA | 64.6 | -7.6pp |

Initial transition counts:

| Pair | wrong->correct | correct->wrong | Net |
|---|---:|---:|---:|
| Dream HE | 10 | 18 | -8 |
| LLaDA HE | 10 | 22 | -12 |
| Dream MBPP | 2 | 2 | 0 |
| LLaDA MBPP | 1 | 30 | -29 |

## Phase A: Conservative Mask-Strength Sweep

Use `mask_ratio` instead of a fixed confidence threshold. This avoids assuming that the dLLM confidence scale is calibrated across different AR drafters.

Start with Dream only because it is more stable than LLaDA in the `tau=0.9` run.

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

Repeat for MBPP after the HumanEval signal is clear.

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

## Interpretation

The expected win is not a large increase in fixed cases. The main target is to reduce broken cases. A useful conservative setting should:

- preserve most baseline-correct samples
- keep the small number of fixed cases
- avoid the `tau=0.9` pattern where fixes are outweighed by regressions

## Progress Log

Implemented in `src/coder/scripts/gen_remask.py`:

- `--record_mask_stats`
- `--gate_min_mask_fraction`
- `--gate_max_mask_fraction`
- shared pre-generation mask planning so gate decisions use only locator/refiner confidence, not test feedback

Completed Dream `mask_ratio` sweep for Seed-Coder-Instruct:

| Run | HE+ | HE base | HE fixed | HE broken | MBPP+ | MBPP base | MBPP fixed | MBPP broken |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 70.1 | 75.0 | - | - | 72.2 | 84.9 | - | - |
| maskr0.01 | 65.2 | 70.7 | 10 | 18 | 72.5 | 85.2 | 2 | 1 |
| maskr0.03 | 65.2 | 70.7 | 10 | 18 | 72.5 | 85.2 | 2 | 1 |
| maskr0.05 | 65.2 | 70.7 | 10 | 18 | 72.5 | 85.2 | 2 | 1 |

Conclusion: lower token-level `mask_ratio` alone does not fix HumanEval. It still has too much `correct -> wrong` churn.

Pre-refinement gate check:

| Run | HE+ | HE base | HE fixed | HE broken | HE skipped | MBPP+ | MBPP base | MBPP fixed | MBPP broken | MBPP skipped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| maskr0.01 + gate_max0.015 | 72.0 | 77.4 | 7 | 4 | 101 | 72.2 | 84.9 | 0 | 0 | 321 |

The gated artifact was materialized from the completed `maskr0.01` output plus the original baseline drafts because GPU permission for a fresh Dream run was unavailable. This is behavior-equivalent for this gate: samples with `mask_fraction <= 0.015` keep the prior refinement, and samples above the gate use the original draft.

Current recommendation:

- Treat `mask_ratio=0.01, gate_max_mask_fraction=0.012` as a safety setting, not a confirmed improvement setting.
- The useful sweep is around the gate, not the ratio. The off-policy sweep below found a promising HumanEval lift, but a fresh rerun did not reproduce the lift.
- Fix and keep the EvalPlus packaging change before any further gated runs; skipped samples must preserve the original EvalPlus `solution`.
- Next experiments should test stability across seeds / deterministic decoding before claiming a pass@1 gain.

## Expanded Gate Sweep

Completed an off-policy gate sweep over:

- datasets: HumanEval, MBPP
- Dream token-level `mask_ratio`: `0.01, 0.03, 0.05`
- `gate_max_mask_fraction`: `0.0, 0.008, 0.010, 0.012, 0.015, 0.018, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.060, 0.080, 0.100, 1.0`

Full CSV:

- `outputs/base_tuteng/seed_coder_instruct_dream_gate_sweep.csv`

Best points from the sweep:

| Dataset | mask_ratio | gate_max | plus% | base% | Delta plus | fixed | broken | kept/skipped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HumanEval | 0.01 | 0.012 | 72.6 | 78.0 | +2.4pp | 7 | 3 | 34/130 |
| HumanEval | 0.01 | 0.015 | 72.0 | 77.4 | +1.8pp | 7 | 4 | 63/101 |
| HumanEval | 0.01 | 0.018 | 71.3 | 77.4 | +1.2pp | 9 | 7 | 88/76 |
| MBPP | 0.01 | 0.012 | 72.2 | 84.9 | +0.0pp | 0 | 0 | 40/338 |
| MBPP | 0.01 | 0.045 | 72.5 | 85.2 | +0.3pp | 1 | 0 | 234/144 |

Representative materialized artifacts were evaluated with EvalPlus and matched the sweep:

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

Interpretation update:

- The off-policy sweep supports the low-disagreement hypothesis on one completed Dream sample: useful repairs concentrate below about `1.2%` planned mask fraction; every larger bin is net harmful.
- A fresh Dream rerun of the same setting did not reproduce the HumanEval lift. After fixing EvalPlus packaging, it returned exactly to the Seed-Coder baseline on HumanEval and MBPP.
- MBPP does not show a strong HE-style repair signal. Strict gates preserve baseline; a looser MBPP-only gate around `0.045` gives a small +0.3pp in the materialized sweep only.
- Current conclusion: disagreement gating is a reliable harm-reduction mechanism for Seed-Coder-Instruct, but it is not yet a reliable improvement mechanism.

## Fresh Validation And Packaging Fix

Fresh GPU reruns exposed an EvalPlus packaging issue in `gen_remask.py`: when a skipped sample kept the original draft, the script still rebuilt `solution` from `raw_completion`. If the completion started with `def`, prompt imports such as `from typing import List` could be dropped. This made some skipped samples fail even though refinement was skipped.

Fix applied:

- `build_evalplus_solution()` now preserves prompt imports for full-function completions.
- skipped EvalPlus samples preserve the input `solution` when available.

Fresh gated results after the packaging fix:

| Run | Dataset | plus% | base% | Delta plus | fixed | broken | skipped |
|---|---|---:|---:|---:|---:|---:|---:|
| Dream `maskr0.01 + gate0.012` fresh fixed | HumanEval | 70.1 | 75.0 | +0.0pp | 0 | 0 | 130 |
| Dream `maskr0.01 + gate0.012` fresh fixed | MBPP | 72.2 | 84.9 | +0.0pp | 0 | 0 | 338 |
| LLaDA `maskr0.01 + gate0.012` fixed | HumanEval | 70.1 | 75.0 | +0.0pp | 0 | 0 | 121 |
| LLaDA `maskr0.01 + gate0.012` fixed | MBPP | 72.2 | 84.9 | +0.0pp | 0 | 0 | 333 |

Compared with the harmful `tau=0.9` runs, the gate is clearly safer:

| Run | Dataset | plus% | Delta plus | fixed | broken |
|---|---|---:|---:|---:|---:|
| Dream `tau=0.9` | HumanEval | 65.2 | -4.9pp | 10 | 18 |
| Dream gated fresh fixed | HumanEval | 70.1 | +0.0pp | 0 | 0 |
| LLaDA `tau=0.9` | HumanEval | 62.8 | -7.3pp | 10 | 22 |
| LLaDA gated fixed | HumanEval | 70.1 | +0.0pp | 0 | 0 |
| LLaDA `tau=0.9` | MBPP | 64.6 | -7.6pp | 1 | 30 |
| LLaDA gated fixed | MBPP | 72.2 | +0.0pp | 0 | 0 |

This changes the claim: low-disagreement gating prevents degradation on fresh runs, but the observed +2.4pp HumanEval gain was tied to one previous stochastic Dream sample and should not be reported as stable yet.

## Next Steps

1. Validate stochastic stability on kept samples.

   Run the same low-disagreement gate with multiple Dream seeds or deterministic settings and report kept-sample transition variance. The key question is whether the 34 HumanEval kept samples have any stable positive repair signal or whether repairs are mostly sampling noise.

2. Test a stricter edit policy.

   Current diffusion refinement can still rewrite the masked position in a way that collapses to no effective repair or introduces noise. Try a one-token or minimal-span policy and record edit distance, so the method is closer to "surgical repair" than "small-region stochastic rewrite".

3. Separate safety gate from improvement mechanism.

   Keep `mask_ratio=0.01, gate_max_mask_fraction=0.012` as the safety baseline. Any improvement method should be compared against both Seed-Coder standalone and this safety-gated baseline.

4. Sweep gate threshold only after stability is measured.

   The off-policy sweep suggests `0.012` is the safest shared threshold and `0.045` can recover a small MBPP-only materialized gain. Do not promote either as an improvement setting until fresh multi-seed runs reproduce it.

5. Run granularity ablations last.

   Span/line granularity will expand planned edit size, so it should use the same `mask_fraction` gate and report actual skipped/kept counts. Start with `span --span_merge_gap 1`; line-level is likely too aggressive for Seed-Coder-Instruct.

6. Optional efficiency follow-up.

   The current gate still scores every draft with the refiner before skipping. If the gate becomes a standard safety mechanism, consider a cheaper locator-only path or cached mask plans to avoid paying full refiner setup cost for mostly skipped runs.
