## Known Pitfalls and Fixes

This document records pitfalls we have already encountered. When similar symptoms arise, check here first.

## LiveCodeBench Evaluation: `n_scored=0` / `accuracy=None`

### Symptom

- `outputs/base_tuteng/*_livecodebench_judgments.jsonl` has 1055 lines
- But `*_livecodebench_summary.json` shows:
  - `n_scored: 0`
  - `accuracy: null`
  - `by_task: {}`

### Root Cause

The LiveCodeBench official scorer expects an `original_json` field in the task dict; its absence triggers:

- `KeyError: 'original_json'`

This causes `score=None` for every task, and `n_scored=0` in the final summary.

### Fix

Already fixed in `src/coder/scripts/eval_livebench.py`: when loading tasks, add `row["original_json"] = original`.

After the fix, simply re-run:

```bash
python -m coder.scripts.eval_livebench --benchmark livecodebench \
  --samples outputs/base_tuteng/<model>_livecodebench.jsonl \
  --out_judgments outputs/base_tuteng/<model>_livecodebench_judgments.jsonl \
  --out_summary outputs/base_tuteng/<model>_livecodebench_summary.json
```

## BigCodeBench Evaluation Hangs: Overwrite Prompt on `*_eval_results.json`

### Symptom

In tmux/logs you see something like:

`<path>_eval_results.json already exists. Press [Y/N] to overwrite or exit...`

This causes the task to **hang indefinitely** when unattended.

### Fix

- **Before starting**, clean up old `*_eval_results.json` files
- Or avoid evaluation commands that pop interactive prompts (prefer our wrapper scripts)

Example (cleanup):

```bash
rm -f outputs/base_tuteng/*_bigcodebench_instruct_*_eval_results.json
```

## tmux Session "Instant Exit"

### Common Causes

- Quote nesting too complex in the start command; variables expanded prematurely in the outer shell, causing a syntax error in the final command
- The command errors out immediately, but without a log, you can't see why

### Fix

- Prefer writing a script first, then start the script in tmux
- Or force `2>&1 | tee -a <log>` in the tmux command, and add a trailing `sleep` to keep the session alive for debugging

## gen_remask Interrupted: JSONL Truncation / Resume Timing Unreliable

### Symptom

- `*.jsonl.lock` file exists, but `ps` / `tmux ls` shows no corresponding task
- `gen_remask --resume` fails immediately with `json.JSONDecodeError`
- `*.timing_summary.json` `n_records_written` equals only the newly added records in this resume run, not the total JSONL count
- Or `remask_generate_s_total=0.0`, causing the results table to incorrectly display `0.0s`

### Root Cause

When a task exits abnormally, the last line of the JSONL may be only partially written. The old `gen_remask.py` had two additional bugs:

- After normal completion, it only released `flock` but didn't delete the `.lock` file, making it easy to mistake a stale lock for a still-running task
- `--resume` timing summary only counted records added in this run, not representative of the full output

### Fix

The current code is fixed:

- `gen_remask.py --resume` aggregates `gen.timing` from the entire output file
- `gen_remask.py` deletes `.lock` upon normal completion
- `gen_results_table.py` ignores invalid resume-only timing where `remask_generate_s_total <= 0`

If you encounter a truncated tail line in old artifacts, the order of operations:

```bash
# 1. First confirm whether the lock is still held; checking file existence alone is unreliable
python3 -c "import fcntl, pathlib, sys
p = pathlib.Path('<out.jsonl>.lock')
with p.open('r+') as f:
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('locked')
        sys.exit(1)
    print('unlocked')"

# 2. Back up the original file, truncate the last incomplete JSONL line
# 3. Re-run the same command with --resume
```

Example instance:

- `deepseek_dream_remask_mbpp_t0.9_timed.jsonl` originally had 326 valid records with a truncated tail line
- Backed up to `deepseek_dream_remask_mbpp_t0.9_timed.jsonl.corrupt_tail_backup_20260413_2205`
- Truncated tail line, then `--resume` completed to 378 records

## dLLM Confidence Signal Fails on Math Reasoning

### Symptom

- `math_locator_analysis.py` ran three dLLM locators on GSM8K (Llama-3.1 8B generations, 200 problems); consistent results:

| Locator | Correct mean_conf | Incorrect mean_conf | Δ | worst_step Δ |
|---------|-------------------|---------------------|---|--------------|
| LLaDA 8B | 0.9485 | 0.9521 | −0.004 | +0.012 |
| Dream (general) | 0.9010 | 0.9078 | −0.007 | +0.010 |
| Dream-Coder | 0.9576 | 0.9570 | +0.001 | +0.024 |

Confidence scores for wrong answers are **nearly indistinguishable** from correct answers, and the direction is often opposite to expectations.

### Root Cause: Generation Mode Mismatch

The math benchmark capabilities demonstrated in dLLM papers (LLaDA, Dream) come from **native generation starting from fully masked state**:
- Bidirectional attention; the entire sequence evolves simultaneously during denoising
- Confidence signals control "which token to unmask first" (self-consistent)

CoCoder's usage is completely different:
- AR draft is already unidirectionally fixed; dLLM is asked to **detect errors on a fixed skeleton**
- dLLM confidence is calibrated under a "simultaneous global generation" assumption, which is meaningless in the "AR-fixed prefix + error detection" scenario

Additionally, [LogicDiff (2025)](https://arxiv.org/abs/2603.26771) points out: even in dLLM native generation, **standard confidence-based unmasking is suboptimal for math reasoning** — it skips high-entropy logical connectives ("therefore", equals signs), collapsing the solution space before the reasoning structure forms. Order-dependent unmasking can improve GSM8K from 22% to 60.7%.

### Experiment 2: LSO (Leave-Sentence-Out) Reconstruction NLL

To better utilize dLLM's bidirectional attention, we tried sentence-level LSO: mask all tokens on each line, let the model reconstruct, and compute recon_nll = −mean log P(orig|masked_context).

**Results (LLaDA, first 200 GSM8K problems)**:

| Metric | Correct | Incorrect | Δ | Cohen's d |
|--------|---------|-----------|---|-----------|
| worst_step recon_nll | 5.018 | 5.380 | +0.363 | 0.28 |
| mean_step recon_nll | 1.286 | 1.296 | +0.010 | — |
| worst_step recon_acc | 0.092 | 0.046 | −0.046 | — |

**Conclusion**: LSO is ~6–9× stronger than token confidence, but Cohen's d=0.28 is still too weak (a reliable locator needs ≥0.5). Also, the worst_step position distribution reveals a key confound: **both correct and incorrect concentrate in the last 25%** (59% vs 71%), suggesting LSO mainly captures "the answer line is naturally hard to reconstruct," not the truly erroneous step.

### Conclusion

dLLM confidence (token-level or sentence-level) as a math CoT error locator **is mechanistically flawed** — not fixable by tuning. Impact on CoCoder:

- **Code tasks**: errors have structural signals (syntax, types, call format); dLLM trained on code corpora can detect them; locator works
- **Math tasks**: errors are arithmetic/semantic — `28+15=42` is linguistically fluent; all language model signals are absent

### Directions to Explore (Bypassing dLLM Locator)

- **AR logprob**: use the AR model's own logprob for localization (AR is calibrated on math corpora)
- **Execution verification**: use Python/SymPy to execute intermediate expressions, find the first numerical error (applicable to GSM8K)
- **PRM (Process Reward Model)**: specifically trained for per-step math scoring

### Related Files

- Experiment scripts: `python -m coder.analysis.math_locator_analysis`, `python -m coder.analysis.math_lso_analysis`
- Artifacts: `outputs/math/llama31_gsm8k_locator_analysis_{llada,dream,dream_coder}.json`, `outputs/math/llama31_gsm8k_lso_llada.json`
- References: LLaDA (2502.09992), Dream (2508.15487), LogicDiff (2603.26771), The Flexibility Trap (2601.15165)

---

## DiffuLLaMA (diffusionfamily/diffullama) Generation Parameters Misaligned

### Symptom

- Outputs contain many repetitions/truncations/off-prompt responses (HumanEval even scores 0 pass@1)

### Official Quick-Start Parameters (HKUNLP/DiffuLLaMA)

`inf_diffullama.py` defaults:
- `diffusion_steps=64`
- `logits_temp=0.9`
- `topp_temp=0.9`
- `shift=True` (officially marked "do not change")
- Conditional generation requires `src_mask` to fix the prefix

### Fix

Adapters should:
- Pass `src_mask` (if the model interface supports it)
- Use steps/temp/top_p close to official defaults (steps is denoising steps, not max_new_tokens)
