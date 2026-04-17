# Experiment Tracker (NeurIPS 2026 Submission Preparation)

This document summarizes experiments that need to be completed before submission, as well as work that is **already implemented in the repo but not yet incorporated into paper tables**, aligned with `docs/ablation_ideas.md` and `docs/completion-checklist.md`.

---

## 0. Rebuttal-CoCoder Branch Comparison (Updated 2026-04-13)

**Comparison conclusion**: The collaborator's Rebuttal-CoCoder branch and NeurIPS26-CoCoder are **fully consistent** on experimental numbers; no additional new data. Status of each rebuttal commitment:

| Commitment (rebuttal location) | Content | Current Status |
|-------------------------------|---------|---------------|
| Group 2 / Table 5 | Self-Refine/Reflexion/Reranking/dLLM-locate+AR-rewrite pass@1 | ✅ Written into paper |
| Group 4 / Table 2 | LiveCodeBench 12.0%, BigCodeBench 23.0% | ✅ Written into paper |
| Group 9 / Table 4 | DeepSeek + LLaDA-8B (HumanEval+ 65.9%) | ✅ Written into paper |
| Group 4 / Table 4 | Qwen2.5-Coder/Llama-3.1/StarCoder2 + Dream-Coder (HumanEval+) | ❌ Only MBPP done; HumanEval+ missing (see §3) |
| Group 2 / Table 5 | Wall-clock latency column | ❌ Not filled into table |
| Group 7 | Mask granularity ablation (token/span/line) | ❌ Not run |
| Group 11 | Token-level precision/recall | ❌ No script |
| Group 7 JwDe Q4 | Failure mode breakdown (over/under/algo-mismatch) | ❌ Not done |
| Group 5 XBny Q1 | τ over/under-mask qualitative cases (`qualitative_analysis_2.png` may already exist) | ❌ Not integrated |
| Group 12 YFoi Q2 | (τ, temperature, top-p) joint sensitivity | ❌ Not run |

**Note**: "Ghost content" already fixed in the paper: removed dangling `tab:granularity` reference, AR+Multi-round and MLM-style baseline descriptions (see NeurIPS26-CoCoder/docs/changelog.md).

---

## 1. Main Table Pending Numbers (Pipeline Implemented, Results Pending)

| Paper Location | Content | Status | Code/Artifact Pointer |
|---------------|---------|--------|-----------------------|
| Table 5 (`tab:baselines`) | Self-Refine, Reflexion, Reranking ($k{=}8$), dLLM-locate + AR-rewrite; pass@1 | ✅ Filled | — |
| Table 5 (`tab:baselines`) | **Overhead column** (s/sample, HE): Self-Refine 4.6s, Reflexion 9.5s, Reranking 61.7s, Locate-AR-Rewrite 7.2s, Collab ≈5s | ✅ Filled (2026-04-13) | results.md Table 4 |
| Table 2 (`tab:extended`) | LiveCodeBench full (LCB 1055 tasks: DeepSeek 11.4%, Dream 2.9%, Collab 11.6%); BigCodeBench full (BCB 1140 tasks: DeepSeek 24.7%, Dream 22.5%, Collab 24.6%) | ✅ Filled (2026-04-13) | results.md Extended Table Shards |
| Table 4 (`tab:model_pairs`) | DeepSeek+Dream-Coder (+15.9pp), DeepSeek+LLaDA (+9.2pp), Qwen+Dream (−0.6pp), Llama+Dream (−0.6pp), StarCoder2+Dream (±0.0pp on HE+) | ✅ Filled (2026-04-13) | results.md Table 3 |
| ~~Table 3 (tab:math)~~ | GSM8K/MATH — **commented out; numbers questionable** | ⚠️ Placeholder | results.md shows DeepSeek GSM8K=19.0%, inconsistent with paper's 48.7%; Dream has no math results |

---

## 2. Appendix Analysis (Not a Single pass@1 Number)

> Background framework: CoCoder consists of three independently replaceable roles: **Drafter → Locator → Rewriter**.
> Most ablations in the table below target the **Locator** role — verifying how "who decides what to mask" affects the final outcome.
> See `ablation_ideas.md` §Framework and `docs/specs/done/spec_locator_ablation.md`.

| Content | Priority | Notes | Related Implementation or Docs |
|---------|----------|-------|-------------------------------|
| **Locator model replacement ablation** | ✅ Done | dLLM locator vs AR logprob vs BERT; pass@1 + fault-detection ratio written into results | `gen_remask --locator {ar,bert}`; `locator_scoring.py`; `ablation_ideas.md` §B; `specs/done/spec_locator_ablation.md` |
| **Mask granularity ablation** | 🔴 High | token / span / line (and `span_merge_gap`); results go into appendix tab:granularity | `gen_remask --mask_granularity {token,span,line}`; design in `ablation_ideas.md` §A |
| **Seed-Coder-Instruct conservative remask** | 🔴 High | `tau=0.9` on strong drafter causes significant degradation; adding low-disagreement gate prevents degradation but fresh run not yet stably improving | `gen_remask --mask_ratio 0.01 --gate_max_mask_fraction 0.012 --record_mask_stats`; details in `specs/spec_seed_coder_instruct_conservative_remask.md` |
| **τ threshold qualitative cases** | 🟡 Medium | Comparison of conservative τ=0.7 missing corrections vs. aggressive τ=0.9 over-masking | Manual selection; `images/qualitative_analysis_2.png` may already have a figure, needs verification |
| **Failure mode breakdown** | 🟡 Medium | (i) over-masking, (ii) under-masking, (iii) algorithm mismatch + concrete examples | Filter "refinement-made-worse" cases from evaluated results; rule-based/manual classification |
| **Token-level precision/recall** | 🟡 Medium | Locator low-confidence tokens aligned with true error positions; precision/recall | Already have `locator_scoring.py` (fault-detection ratio); full P/R requires difflib diff alignment |
| **(τ, temperature, top-p) joint sensitivity** | 🟢 Low | Appendix sweep analysis | `gen_remask --temperature --top_p`; design in `ablation_ideas.md` §E3 |

---

## 3. Pending Runs

> Actual progress: see `docs/results.md` (auto-generated).

- [x] HumanEval / MBPP: `*_reflexion_feedback*.jsonl` — **results available**
- [x] HumanEval / MBPP: `*_rerank_logprob_k8*.jsonl` — **results available**
- [x] Table 3 HumanEval+: qwen/llama31/starcoder2 + dream remask — **all complete** (see results.md)
- [x] Table 3 MBPP: qwen/llama31/starcoder2 + dream remask — **all complete**
- [x] Extended table shards (dream + collab on LCB/BCB) — **all complete** (LCB: 1055✅, BCB: 1140✅)
- [x] DeepSeek + LLaDA on HumanEval+ — ✅ (65.9%)
- [ ] DeepSeek + LLaDA on MBPP — ❌ not yet run (no MBPP data in results.md Table 3 for this row)
- [ ] LiveCodeBench other models (qwen / llama31 / dream): `n_scored=0`; evaluation not working (original_json field missing, see pitfalls.md)
- [ ] BigCodeBench raw pass@1=0.0 investigation (deepseek_pass1_clean is normal; raw version is anomalous)
- [ ] GSM8K / MATH Collaborative Coding experiment — ❌ **not yet run** (numbers in paper are placeholders, commented out)

---

## 4. In-Repo Work Not Yet in Paper

| Topic | Docs |
|-------|------|
| Reflexion + EvalPlus **real failure feedback** pipeline | `ablation_ideas.md` §C |
| Reranking using **AR logprob** instead of heuristic scoring | `ablation_ideas.md` §B |
| Multi-round local patching, combined pipeline (T rounds) | `ablation_ideas.md` §C1 |

---

## 5. Completion Criteria and Artifact Paths

- EvalPlus / LiveCodeBench / BigCodeBench "complete and trustworthy" checks: see **`completion-checklist.md`**.
- Known interaction/evaluation pitfalls: see **`pitfalls.md`**.

---

## 6. Math Tasks (Generalization Verification)

To argue the approach is not limited to code generation, we add experiments on math reasoning tasks. Dataset selection rationale: clear answer verification (number/expression), widely cited in the community, complements code experiments.

| Benchmark | Scale | Notes | Script |
|-----------|-------|-------|--------|
| GSM8K | 1319 (test) | Elementary math, chain-of-thought reasoning, integer/decimal answers | `gen_math --dataset gsm8k` |
| MATH-500 | 500 (test) | Competition math representative subset (AMC/AIME difficulty), LaTeX answers, 5 difficulty levels and 7 subjects | `gen_math --dataset math500` |

### Completed Baselines

- [x] GSM8K: DeepSeek/Qwen/Llama-3.1 baseline — **done** (DeepSeek 19.0%, Qwen 30.6%, Llama-3.1 84.5%)
- [x] MATH-500: DeepSeek/Qwen/Llama-3.1 baseline + subject breakdown — **done** (DeepSeek 4.6%, Qwen 37.6%, Llama-3.1 38.6%)

### Why CoCoder Pipeline Does Not Work on Math (Mechanistic Analysis, Verified)

### Experiment 1: Token-Level Confidence (Single-Pass)

**Completed** (2026-04-16): `math_locator_analysis.py` scored first 200 GSM8K problems with three dLLM locators; consistent conclusion:

| Locator | Correct mean_conf | Incorrect mean_conf | Δ | Cohen's d |
|---------|-------------------|---------------------|---|-----------|
| LLaDA 8B | 0.9485 | 0.9521 | −0.004 | ~0.05 |
| Dream (general) | 0.9010 | 0.9078 | −0.007 | ~0.05 |
| Dream-Coder | 0.9576 | 0.9570 | +0.001 | ~0.01 |

**Conclusion: dLLM token-level confidence has almost no signal (Cohen's d≈0.05).**

Artifacts: `outputs/math/llama31_gsm8k_locator_analysis_{llada,dream,dream_coder}.json`

### Experiment 2: Leave-Sentence-Out (LSO) Reconstruction NLL

**Completed** (2026-04-16): `math_lso_analysis.py` masked all tokens on each reasoning step line, used LLaDA bidirectional attention to reconstruct, computed recon_nll.

| Metric | Correct | Incorrect | Δ | Cohen's d |
|--------|---------|-----------|---|-----------|
| worst_step recon_nll | 5.018 | 5.380 | **+0.363** | **0.28** |
| mean_step recon_nll | 1.286 | 1.296 | +0.010 | — |
| worst_step recon_acc | 0.092 | 0.046 | −0.046 | — |

LSO is ~**6–9× stronger** than token-level signal, but Cohen's d=0.28 is still a weak effect (reliable locator needs ≥0.5).

**Key confound**: worst_step in both correct and incorrect concentrates in the last 25% (59% vs 71%), indicating LSO mainly captures "the answer line is naturally hard to reconstruct," not the truly erroneous step.

Artifacts: `outputs/math/llama31_gsm8k_lso_llada.json`

### Root Cause (Mechanistic Problem, Not Fixable by Tuning)

dLLM's math capability in its own papers is effective because it generates **natively starting from fully masked state**, with the entire reasoning chain evolving simultaneously under bidirectional attention, and confidence signals controlling the unmasking order (self-consistent). CoCoder requires dLLM to **detect errors on a draft already fixed by unidirectional AR** — a completely different task that dLLM was never trained or calibrated for.

Furthermore, [LogicDiff (2025)](https://arxiv.org/abs/2603.26771) points out: even in dLLM native generation, standard confidence-based unmasking is suboptimal for math (skips high-entropy logical connectives), requiring order-dependent unmasking to be effective.

**Contrast with code tasks (explaining why code works)**:

- Code errors have **structural signals** (syntax, types, function call format); dLLM trained on code corpora can detect token-level anomalies
- Math errors are **arithmetic/semantic** — `28+15=42` is linguistically fluent; any language model will assign high confidence

### Directions to Explore (Still Investigating)

| Direction | Core Idea | Expected Feasibility | Status |
|-----------|-----------|---------------------|--------|
| **AR logprob as locator** | Use AR model's own logprob at generation time to find low-confidence steps; AR is calibrated on math corpora, unlike dLLM | Medium | ❌ Not run |
| **Execution verification locator (GSM8K)** | Use Python/SymPy to execute intermediate expressions, find first numerical error step; bypasses language model scoring | High (but requires structured steps) | ❌ Not done |
| **PRM as locator** | Use Process Reward Model (e.g., Qwen-Math-PRM) to score each step; locate errors | High (specifically trained for this) | ❌ Not done |
| **AR step-level truncation + continuation** | Even with a weak locator, oracle truncation experiment validates pipeline upper bound (if oracle doesn't improve, no hope) | Validating | ❌ Not run |

### Paper Writing Recommendation (Current Conclusion Version)

Write Experiments 1+2 as an Analysis section, conclusion:

> CoCoder cannot improve AR baseline on math reasoning. Token-level confidence (Cohen's d≈0.05) and sentence-level reconstruction NLL (Cohen's d=0.28) both fail to reliably localize erroneous steps. The root cause: language model probability signals (AR or dLLM) are insensitive to arithmetic errors — `28+15=42` is as fluent as `28+15=43` at the language level. This defines a natural boundary of CoCoder: effectiveness depends on errors having detectable structural signals (e.g., syntax/type violations in code); it does not apply to purely arithmetic-semantic error detection.

Artifact naming conventions same as code tasks; see `runbook.md` § Math Tasks.

---

## 7. Scope (Not in This Round's Experiment List)

- **Multi-file repo-level / SWE-bench-level**: current pipeline is single code segment + fixed window; full repo evaluation requires agents/retrieval/infrastructure, out of scope for this round (see `main.tex` Limitations).
