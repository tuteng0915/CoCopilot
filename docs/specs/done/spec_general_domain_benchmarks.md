# Spec: General-Domain Benchmarks（NLP Rewriting + Research QA + Writing）

> Merged from `spec_rewriting_benchmarks.md` + `spec_research_writing_benchmarks.md`

## Overview

Validates CoCoder generalization beyond code. All experiments use **Llama-3.1-8B-Instruct** as AR drafter and **Dream-v0-Instruct-7B** (DreamGeneral) as dLLM refiner, τ=0.9.

| Benchmark | Task | n | Metric | Status |
|-----------|------|---|--------|--------|
| ASSET | Sentence simplification | 359 | SARI + BLEU-4 | ✅ Done |
| CoEdIT | GEC + Paraphrase | 1012 | SARI + BLEU-4 | ✅ Done |
| FRAMES | Multi-hop research QA | 824 | EM + Token F1 | ✅ Done |
| HotpotQA | Multi-hop QA (distractor) | 1000 | EM + Token F1 | ✅ Done |
| WildBench Writing | Creative writing | 146 | LLM-as-judge checklist | ⏳ Pending eval |

---

## Current Results

### NLP Rewriting（SARI / BLEU-4）

| Benchmark | Model | SARI | BLEU-4 |
|-----------|-------|------|--------|
| ASSET | AR (Llama31) | 26.02 | 65.26 |
| ASSET | DreamGeneral | 25.21 | 70.05 |
| ASSET | CoCoder τ=0.9 | 25.50 | 65.80 |
| CoEdIT GEC | AR (Llama31) | 44.34 | 32.37 |
| CoEdIT GEC | DreamGeneral | 55.73 | 46.21 |
| CoEdIT GEC | CoCoder τ=0.9 | 44.27 | 32.64 |
| CoEdIT Paraphrase | AR (Llama31) | 41.11 | 15.62 |
| CoEdIT Paraphrase | DreamGeneral | 32.03 | 15.42 |
| CoEdIT Paraphrase | CoCoder τ=0.9 | 38.93 | 15.25 |

### Research QA（EM% / Token F1%）

| Benchmark | Model | EM% | Token F1% |
|-----------|-------|-----|-----------|
| FRAMES | AR (Llama31) | 0.0% | 4.4% |
| FRAMES | DreamGeneral | 1.1% | 11.4% |
| FRAMES | CoCoder τ=0.9 | 0.0% | 4.4% |
| HotpotQA | AR (Llama31) | 13.5% | 22.4% |
| HotpotQA | DreamGeneral | 16.4% | 25.4% |
| HotpotQA | CoCoder τ=0.9 | 9.7% | 18.9% |

### Creative Writing（WildBench, n=146）

Pending — generation artifacts exist, eval requires LLM-as-judge (ANTHROPIC_API_KEY).

---

## Interpretation

CoCoder does **not** improve over AR baseline on any of these benchmarks:
- Rewriting (ASSET/CoEdIT): CoCoder ≈ AR baseline; DreamGeneral standalone performs differently but CoCoder remask brings no lift
- Research QA (FRAMES/HotpotQA): CoCoder degrades AR baseline; the dLLM confidence signal doesn't localize factual errors
- Pattern is consistent with Math results: dLLM confidence is insensitive to semantic/factual errors, only works for local syntactic code bugs

---

## Remaining Work

### Writing Eval（WildBench Creative Writing）

Generation artifacts exist (`outputs/writing/writing_{llama31,dream_general,llama31_dream_general_t0.9}.jsonl`, 146 records each). Needs LLM-as-judge:

```bash
python -m coder.scripts.eval_writing \
    --input outputs/writing/writing_llama31.jsonl \
    --out   outputs/writing/writing_llama31_eval.json

python -m coder.scripts.eval_writing \
    --input outputs/writing/writing_dream_general.jsonl \
    --out   outputs/writing/writing_dream_general_eval.json

python -m coder.scripts.eval_writing \
    --input outputs/writing/writing_llama31_dream_general_t0.9.jsonl \
    --out   outputs/writing/writing_cocoder_eval.json
```

Requires `ANTHROPIC_API_KEY`. ~584 API calls total (146 × ~4 checklist items). Use `claude-haiku-4-5-20251001` to minimize cost.

**Decision point**: Given the consistent pattern of no improvement, Writing eval is low priority unless paper needs it for completeness.

---

## Artifact Paths

| Artifact | Path |
|----------|------|
| ASSET eval results | `outputs/rewrite/asset_{llama31,dream_general,cocoder}_eval.json` |
| CoEdIT eval results | `outputs/rewrite/coedit_{llama31,dream_general,cocoder}_eval.json` |
| FRAMES eval results | `outputs/research/frames_{llama31,dream_general,cocoder}_eval.json` |
| HotpotQA eval results | `outputs/research/hotpotqa_{llama31,dream_general,cocoder}_eval.json` |
| Writing generation | `outputs/writing/writing_{llama31,dream_general,llama31_dream_general_t0.9}.jsonl` |
