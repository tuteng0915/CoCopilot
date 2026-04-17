# Ablation Experiments and Mini-Study Ideas (CoCoder)

This document records "easy to run, conclusive" ablation / model study ideas to prevent them from remaining only in verbal discussion.

---

## Framework: Three-Role Decomposition

CoCoder's core pipeline consists of three independently replaceable roles:

```
AR drafter  ──►  Locator  ──►  dLLM rewriter
 generates draft   scores draft, marks low-confidence tokens   rewrites masked positions (diffusion)
```

- **Drafter**: AR model (DeepSeek-Coder, Qwen2.5-Coder, etc.) generates the initial code draft
- **Locator**: scores each token in the draft for confidence; low-scoring tokens are masked
- **Rewriter**: dLLM (Dream-Coder, LLaDA) diffusion-regenerates masked positions

**Default configuration**: Locator and Rewriter share the same dLLM. The dLLM first performs a single forward pass on the draft to score it (acting as Locator), then performs diffusion rewriting on low-confidence tokens (acting as Rewriter). These two sub-steps are already decoupled in the code (`coder/locators/`) and can be freely replaced.

Overview of ablation dimensions per role:

| Role | Ablation Dimension | Core Question | Section |
|------|-------------------|---------------|---------|
| **Locator** | Granularity: token / span / line | Fragmented mask vs. structured mask | §A |
| **Locator** | Model: dLLM / AR / BERT | Is bidirectional perception necessary? Is a lightweight model sufficient? | §B |
| **Rewriter** | Model: Dream-Coder / LLaDA | Effect of diffusion model choice | Table 3 (done) |
| **Drafter** | Model: DeepSeek / Qwen / Llama / StarCoder2 | Effect of AR draft quality | Table 3 (done) |

---

## A. Locator Granularity (Implemented)

**Core question**: After the Locator scores the draft, how are the mask "boundaries" determined?

Script: `python -m coder.scripts.gen_remask`, parameters: `--mask_granularity {token,span,line}`

| Granularity | Behavior | Use Case |
|-------------|----------|---------|
| `token` | Mask only low-confidence tokens (default, finest granularity) | Point errors (single variable name, operator) |
| `span` | Merge adjacent low-confidence tokens into contiguous spans (`--span_merge_gap K`) | Reduce fragmentation; suitable for local logic errors |
| `line` | If any token on a line is low-confidence, mask the entire line | Closer to "structured editing"; suitable for indentation/block-level errors |
| `step` | Find the least-confident step, **truncate** from that step, let Rewriter take over continuation | CoT/math reasoning: ensures causal consistency of subsequent steps; Rewriter can be AR or dLLM |

The `step` granularity is at the same abstraction level as the other three (all are "how to determine mask boundaries"), but semantically shifts from "fill holes" to "continue writing" — especially suited for chain-of-thought scenarios, since each CoT step depends on the previous result and parallel denoising cannot guarantee numerical consistency between steps.

Recommended sweep: `{token, span, line}` × `span_merge_gap ∈ {0, 1, 2, 4}`, comparing under aligned masked token ratios (otherwise unfair). `step` granularity evaluated separately (truncation position selection strategy can be ablated independently).

### A2. dLLM-locate + AR-rewrite (Implemented)

Script: `python -m coder.scripts.gen_locate_ar_rewrite`

Use dLLM as Locator, use AR model as Rewriter (constrained via prompt to "only modify masked positions").
Note: AR rewriter is not a hard-constraint edit; it can serve as one baseline after Locator decoupling (corresponding to §C2 Pending).

---

## B. Locator Model Replacement (Implemented)

**Core question**: When the Locator is replaced with models of different architectures/scales, how does pass@1 change?

This experiment directly tests dLLM's contribution in the localization stage: **Is bidirectional context perception necessary? Is it worth loading a 7B dLLM for scoring?**

### Three Locators

| Name | Params | Perception | Extra Inference Cost | How to Use |
|------|--------|------------|---------------------|-----------|
| `dream` (default) | 7B | **Bidirectional** (diffusion LM full-sequence forward) | 0 (shared with Rewriter) | default behavior |
| `ar` | 7B | **Unidirectional** (AR teacher-forced logprob) | 1 AR forward pass | `--locator ar` |
| `bert` | 125M | **Bidirectional** (CodeBERT MLM head, single forward) | Minimal | `--locator bert` |

Example runs:

```bash
# AR locator (test "is bidirectional perception necessary?")
python -m coder.scripts.gen_remask \
  --input  outputs/base_tuteng/deepseek_humaneval.jsonl \
  --out    outputs/ablation_locator/deepseek_dream_humaneval_t0.9_loc_ar.jsonl \
  --locator ar --mask_ratio 0.9

# BERT locator (test "is 125M sufficient?")
python -m coder.scripts.gen_remask \
  --input  outputs/base_tuteng/deepseek_humaneval.jsonl \
  --out    outputs/ablation_locator/deepseek_dream_humaneval_t0.9_loc_bert.jsonl \
  --locator bert --mask_ratio 0.9
```

Full experiment spec: `docs/specs/done/spec_locator_ablation.md`.

### Auxiliary Analysis: Fault Detection Ratio

Without running full pass@1, you can quickly evaluate Locator fault detection capability from existing artifacts:

```bash
python -m coder.analysis.locator_scoring \
  --remask_dir outputs/base_tuteng --dataset humaneval --device cuda
```

Outputs `P(fault) / P(non-fault)` ratio — higher ratio means the Locator better distinguishes truly erroneous tokens from correct ones.

### Expected Conclusions and Paper Significance

- **If AR < dLLM**: bidirectional context perception is effective; dLLM's Locator role has independent contribution
- **If BERT ≈ dLLM**: 125M lightweight bidirectional model is sufficient; no need to load an extra 7B model
- **If AR ≈ dLLM**: error localization mainly depends on local context; bidirectional perception is not critical

---

## C. Reranking with AR Logprobs (Basic Version Implemented)

**Background**: This section covers Best-of-N candidate reranking (`gen_rerank.py`), independent of the §B remask locator; it is a separate pipeline.

Script: `python -m coder.scripts.gen_rerank`

`--score_mode` options:
- `self_judge` (default): AR model does listwise selection, picks one from N candidates
- `logprob`: teacher-forced `sum/avg log p(completion | prompt)`, takes best by score
- `heuristic`: heuristic scoring (legacy fallback)

```bash
python -m coder.scripts.gen_rerank \
  --model deepseek-ai/deepseek-coder-6.7b-instruct \
  --dataset humaneval --num_samples 8 \
  --score_mode logprob --logprob_norm avg \
  --out outputs/ablation/deepseek_rerank_logprob_k8_humaneval.jsonl
```

---

## D. Reflexion Baseline (Simplified Version Implemented)

Script: `python -m coder.scripts.gen_reflexion`

Pipeline (default 1 round): problem + draft → `reflection` (verbal reflection) → `revised` (revised code).

Options:
- `--rounds T`: multi-round reflexion
- `--feedback_key KEY` / `--feedback_file FILE`: inject real eval failure feedback (EvalPlus supported)

Companion feedback extraction:
```bash
python -m coder.analysis.evalplus_feedback \
  --eval_results <..._eval_results.json> --out_feedback <...evalplus_feedback.jsonl>
```

---

## E. Pending: Follow-up Work (Not Yet Implemented)

### E1. Multi-Round Local Patching (T Rounds)

Locate → rewrite → re-locate → re-rewrite; observe gains and degradation at T=1/2/3.
Corresponds to §B three-role framework: each round can use a different Locator.

### E2. More Combinations of Decoupled Locator

The three-role framework theoretically allows arbitrary combinations:

| Drafter | Locator | Rewriter |
|---------|---------|---------|
| AR | dLLM (default) | dLLM |
| AR | AR (§B ablation) | dLLM |
| AR | BERT (§B ablation) | dLLM |
| AR | dLLM | AR (A2, implemented) |
| AR | Random (pure ablation baseline) | dLLM |

Random locator (randomly mask the same proportion of tokens) is the cleanest ablation baseline: if random ≈ dLLM, it means localization itself has no value.

### E3. (τ, temperature, top-p) Joint Sensitivity

`gen_remask --temperature --top_p` sweep; design in `ablation_ideas.md` §A.

### E4. "Edit Magnitude vs. Success Rate" Analysis Plot

Count mask count, diff distance, and pass/fail per problem; find the optimal edit magnitude range (cross-analyzed with Locator type).

### E5. Prompt / Output Format Robustness

Code fence sensitivity, system prompt strength, max_new_tokens truncation sensitivity.
