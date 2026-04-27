# results.md Audit (2026-04-20)

Scope: verify that `docs/results.md` is reproducible from current result artifacts and that displayed metrics match their source summaries/eval files.

## Commands Run

- `PYTHONPATH=src python3 -m coder.scripts.gen_results_table --out docs/results.md`
- `python3 -m py_compile src/coder/scripts/gen_results_table.py`
- `git diff --check`
- EvalPlus CPU re-eval for deduped HumanEval Locate-AR-Rewrite audit files:
  - `outputs/base_tuteng/results_audit/llama31_humaneval_locate_ar_rewrite_t0.9_dedup_audit.jsonl`
  - `outputs/base_tuteng/results_audit/starcoder2_humaneval_locate_ar_rewrite_t0.9_dedup_audit.jsonl`

## Fixes Applied

- Added DeepSeek baseline timing to Table 4 from existing `_timed` artifacts:
  - HumanEval: `6.9s`
  - MBPP: `5.2s`
- Fixed pass@1 handling for EvalPlus summaries with duplicate task rows. The generator now recomputes table percentages from the first row per task when `n_samples_total != n_tasks`.
- Corrected affected Table 4 rows:
  - Llama-3.1 `Locate-AR-Rewrite` HumanEval: `53.7%` plus, `58.5%` base.
  - StarCoder2 `Locate-AR-Rewrite` HumanEval: `3.0%` plus, `3.7%` base.
- Updated BigCodeBench loader to read sample100 summaries stored under `pass_at_k.pass@1`.
- Replaced partial Dream-Coder LiveCodeBench row (`71` scored) with full sharded result (`1055` scored, `2.94%`).
- Rewrote `outputs/remask_kodai/remask_{humaneval,mbpp}_t*_summary.json` from local eval result files so `source_eval_file` paths are local and auditable.

## Final Audit Result

Final audit status: `issues=0`.

Checked:

- `88` EvalPlus summary files referenced by `results.md` sections.
- `76` timing files referenced by `results.md` sections.
- Table 3 transition counts from source eval files, except the two Seed-Coder-Instruct HumanEval rows intentionally overridden by pkgv2 analysis.
- Math summary problem counts (`GSM8K=1319`, `MATH500=500`).
- Research QA metrics range and file presence.
- LiveCodeBench / BigCodeBench displayed rows and sample counts.

Known notes:

- Llama-3.1 and StarCoder2 Locate-AR-Rewrite HumanEval source evals contain duplicate task rows (`175/164` and `174/164`). `results.md` now reports pass@1 after task-level deduplication, verified by CPU EvalPlus re-eval of deduped audit JSONL files.
- LLaDA LiveCodeBench remains `n_scored=0` and is displayed as incomplete in `results.md`.
- Table 4 still uses existing EvalPlus sanitized artifacts, not a full pkgv2 re-evaluation. Seed-Coder-Instruct HumanEval rows use pkgv2 where explicitly noted.
