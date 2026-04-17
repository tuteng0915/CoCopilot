# CoCoder Runbook (Condensed)

This `docs/` directory makes the "easy-to-forget / easy-to-trip-over" operational context explicit, avoiding repeated trial-and-error from relying solely on chat memory.

## Method Framework: Three-Role Decomposition

CoCoder consists of three independently replaceable roles:

```
AR drafter  ──►  Locator  ──►  dLLM rewriter
 generates draft   scores tokens, marks low-confidence   rewrites masked positions (diffusion)
```

**Default configuration**: Locator and Rewriter share the same dLLM (Dream-Coder / LLaDA).
The dLLM first performs a single forward pass on the draft, taking `P(token | full sequence)` as confidence (Locator), then performs diffusion rewriting on low-confidence tokens (Rewriter).

All three roles can be replaced independently, corresponding to ablation experiments:

| Role | Default | Can be replaced with | Ablation meaning |
|------|---------|---------------------|-----------------|
| Drafter | DeepSeek-Coder | Qwen / Llama / StarCoder2 | Effect of draft quality (Table 3) |
| Locator | dLLM (bidirectional) | AR logprob (unidirectional) / BERT (bidirectional lightweight) | Is bidirectional attention / scale necessary? |
| Rewriter | Dream-Coder | LLaDA | Effect of diffusion model choice (Table 3) |

Detailed ablation design: `docs/ablation_ideas.md`; completed Locator ablation spec: `docs/specs/done/spec_locator_ablation.md`.

---

## Most Common Entry Points

### Code Tasks

- EvalPlus (HumanEval/MBPP):
  - Generate: `python -m coder.scripts.gen_evalplus`
  - Sanitize: `python -m coder.scripts.postprocess_evalplus`
  - Evaluate: `python -m coder.scripts.eval_evalplus`
- LiveCodeBench / LiveBench-Coding (note legacy script naming):
  - Generate: `python -m coder.scripts.gen_livebench --benchmark livecodebench|livebench-coding`
  - Evaluate: `python -m coder.scripts.eval_livebench --benchmark livecodebench|livebench-coding`
- BigCodeBench:
  - Generate: `python -m coder.scripts.gen_bigcodebench`
  - Evaluate wrapper: `python -m coder.scripts.eval_bigcodebench`

### Math Tasks (Generalization Verification)

To examine generalization beyond code, we also test on math reasoning tasks:

- GSM8K / MATH-500:
  - Generate: `python -m coder.scripts.gen_math --dataset gsm8k|math500`
  - Evaluate: `python -m coder.scripts.eval_math --samples <out.jsonl>`

## Active Documents (Keep in Mind)

- `docs/runbook.md`: environment, command templates, artifact naming conventions
- `docs/tmux.md`: how to reliably run long tasks (and why "migration" requires restart)
- `docs/pitfalls.md`: known pitfalls and fixes (overwrite prompts, missing fields, etc.)
- `docs/ablation_ideas.md`: ablation experiment design (three-role framework, Locator model replacement, granularity, Reflexion, etc.)
- `docs/completion-checklist.md`: criteria for determining whether an experiment run is complete and trustworthy
- `docs/experiments-tracker.md`: pre-NeurIPS 2026 experiment summary (pending paper tables + appendix analysis)
- `docs/specs/`: short-term experiment execution specs and completion records (e.g., Table 3 timing, Group D/E)
- `docs/results.md`: **summary table of all completed experiments** (auto-generated; update with: `python -m coder.scripts.gen_results_table`)

## Archived Documents (Implementation Complete, No Longer Actively Maintained)

Stored in `docs/done/`:

- `docs/done/reflexion_evalplus_feedback.md`: implementation notes for EvalPlus real failure feedback Reflexion
- `docs/done/impl_progress_2026-03.md`: implementation progress from the 2026-03-30 session

The following files are kept as historical references with redirect notes:

- `docs/TODO_reflexion_evalplus_feedback.md` → see `done/reflexion_evalplus_feedback.md`
- `docs/rebuttal-experiments-tracker.md` → see `experiments-tracker.md`
