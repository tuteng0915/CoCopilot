# Coder Benchmark Playground (Collaborative Coding)

Implements **Collaborative Coding**: AR model drafts code → dLLM refines via confidence-based remasking.

**Models:**
- **Dream-Coder 7B** — diffusion LM (masked non-autoregressive)
- **DeepSeek-Coder 6.7B** — autoregressive LM
- **Qwen2.5-Coder 7B** — autoregressive LM (optional)
- **LLaDA** — autoregressive LM (optional)

**Benchmarks:** HumanEval+, MBPP+, LiveBench-Coding

---

## 1 Environment Setup

```bash
conda create -n code python=3.10 -y && conda activate code
pip install torch transformers==4.46.2 accelerate safetensors datasets tqdm evalplus numpy
pip install -e .
# For LiveBench:
pip install git+https://github.com/LiveBench/LiveBench.git
```

Always set before running scripts:
```bash
export PYTHONPATH=/path/to/CoCopilot/src
export CUDA_VISIBLE_DEVICES=<gpu_id>   # e.g. 0 or 1
```

---

## 2 File Structure

```
.
├── src/coder/
│   ├── models/
│   │   ├── base.py              # CoderModel(ABC): .name, .generate(req)
│   │   ├── deepseek_coder.py    # DeepSeekCoder: AutoModelForCausalLM
│   │   ├── dream_coder.py       # DreamCoder: AutoModel + diffusion_generate + score_tokens + generate_with_remask
│   │   ├── qwen_coder.py        # QwenCoder (optional)
│   │   └── llada_coder.py      # LLaDACoder (optional)
│   ├── datasets/
│   │   └── livebench_coding.py # LiveBench-Coding dataset loader
│   └── utils/
│       └── schema.py           # ModelRequest, SampleRecord
│
├── scripts/
│   │ — Generation —
│   ├── gen_evalplus.py          # Generate HumanEval / MBPP completions (dream|deepseek|qwen|llada)
│   ├── gen_remask.py            # DeepSeek draft → DreamCoder token remasking (Collaborative Coding)
│   ├── gen_self_refine.py       # AR + Self-Refine baseline (deepseek|qwen)
│   ├── gen_rerank.py            # AR + Reranking baseline (deepseek|qwen, heuristic verifier)
│   ├── gen_livebench.py         # Generate LiveBench-Coding completions
│   │ — Postprocess & Eval —
│   ├── postprocess_evalplus.py  # Syntax-check + sanitize for EvalPlus format
│   ├── eval_evalplus.py         # Evaluate sanitized JSONL with evalplus
│   ├── eval_livebench.py        # Score LiveBench-Coding outputs
│   │ — Pipeline scripts —
│   ├── 1_generate_deepseek.sh   # Step 1: DeepSeek generation (HumanEval + MBPP)
│   ├── 2_remask_dreamcoder.sh   # Step 2: DreamCoder remasking
│   ├── 3_sanitize.sh            # Step 3: Sanitize all outputs
│   ├── 4_evaluate.sh            # Step 4: Evaluate sanitized outputs
│   ├── 5_pipeline.sh            # Full pipeline (steps 1–4)
│   ├── gen.sh                   # Quick generation helper
│   ├── remask.sh                # Quick remask helper
│   ├── sanitize_and_eval.sh     # Combined sanitize + eval
│   ├── eval.sh                  # Quick eval helper
│   └── exp_examples.sh          # Example recipes: self_refine, rerank, livebench_remask, etc.
│
├── outputs/                     # Generated files (gitignored)
│   ├── <model>_<dataset>.jsonl
│   ├── remask_<dataset>_t<thresh>.jsonl
│   ├── selfrefine_*_<dataset>.jsonl
│   ├── rerank_*_<dataset>.jsonl
│   └── *-sanitized.jsonl, *_summary.json
│
├── requirements.txt
└── readme.md
```

---

## 3 Workflow

### 3.1 Collaborative Coding (main pipeline)

```bash
# Step 1: DeepSeek generates drafts
bash scripts/1_generate_deepseek.sh

# Step 2: DreamCoder remasks low-confidence tokens and denoises
bash scripts/2_remask_dreamcoder.sh

# Step 3 & 4: Sanitize and evaluate
bash scripts/3_sanitize.sh
bash scripts/4_evaluate.sh

# Or all-in-one:
bash scripts/5_pipeline.sh
```

### 3.2 Standalone generation → evaluate

```bash
# Generate (model: dream | deepseek | qwen | llada)
python scripts/gen_evalplus.py --model deepseek --dataset humaneval \
  --out outputs/deepseek_humaneval.jsonl

# Sanitize
python scripts/postprocess_evalplus.py --dataset humaneval \
  --samples outputs/deepseek_humaneval.jsonl

# Evaluate
python scripts/eval_evalplus.py --backend local --dataset humaneval \
  --samples outputs/deepseek_humaneval-sanitized.jsonl
```

### 3.3 Remasking (manual)

```bash
python scripts/gen_remask.py \
  --input  outputs/deepseek_humaneval.jsonl \
  --out    outputs/remask_humaneval_t0.8.jsonl \
  --confidence_threshold 0.8 \
  --temperature 0.0 --top_p 1.0 --seed 3407
# Use --resume to continue an interrupted run.
```

### 3.4 AR + Self-Refine baseline

```bash
python scripts/gen_self_refine.py \
  --input outputs/deepseek_humaneval.jsonl \
  --out   outputs/selfrefine_deepseek_humaneval.jsonl \
  --model deepseek \
  --temperature 0.2 --top_p 0.95 --seed 3407

python scripts/postprocess_evalplus.py --dataset humaneval \
  --samples outputs/selfrefine_deepseek_humaneval.jsonl
python scripts/eval_evalplus.py --backend local --dataset humaneval \
  --samples outputs/selfrefine_deepseek_humaneval-sanitized.jsonl
```

### 3.5 AR + Reranking baseline

```bash
python scripts/gen_rerank.py \
  --model deepseek --dataset humaneval \
  --out outputs/rerank_deepseek_humaneval.jsonl \
  --num_samples 8 --temperature 0.7 --top_p 0.95 --seed 3407

python scripts/postprocess_evalplus.py --dataset humaneval \
  --samples outputs/rerank_deepseek_humaneval.jsonl
python scripts/eval_evalplus.py --backend local --dataset humaneval \
  --samples outputs/rerank_deepseek_humaneval-sanitized.jsonl
```

### 3.6 LiveBench-Coding

```bash
python scripts/gen_livebench.py --model deepseek --out outputs/deepseek_livebench.jsonl
python scripts/gen_remask.py --input outputs/deepseek_livebench.jsonl \
  --out outputs/remask_livebench_t0.8.jsonl --confidence_threshold 0.8
python scripts/eval_livebench.py --samples outputs/deepseek_livebench.jsonl
python scripts/eval_livebench.py --samples outputs/remask_livebench_t0.8.jsonl
```

### 3.7 Example recipes (exp_examples.sh)

```bash
bash scripts/exp_examples.sh evalplus_single humaneval
bash scripts/exp_examples.sh evalplus_multi humaneval
bash scripts/exp_examples.sh livebench_remask
bash scripts/exp_examples.sh self_refine
bash scripts/exp_examples.sh rerank
```

---

## 4 Output File Conventions

| File | Description |
|------|-------------|
| `<model>_<dataset>.jsonl` | Raw: `task_id`, `prompt`, `raw_completion`, `solution`, `model`, `gen` |
| `remask_<dataset>_t<thresh>.jsonl` | Collaborative Coding: adds `draft_completion` |
| `selfrefine_*_<dataset>.jsonl` | Self-Refine: adds `draft_completion` |
| `rerank_*_<dataset>.jsonl` | Reranking: adds `rerank_candidates` |
| `*-sanitized.jsonl` | Cleaned for evalplus ingestion |
| `*_eval_results.json` | Per-task pass/fail from evalplus |
| `*_summary.json` | Aggregated pass@k metrics |

---

## 5 Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'coder'` | `export PYTHONPATH=/path/to/CoCopilot/src` or `pip install -e .` |
| Docker permission denied | Use `--backend local` in `eval_evalplus.py` |
| EvalPlus result file not found | `eval_evalplus.py` probes `_eval_results.json` and `-eval_results.json` |
| Subset evaluation | Pass `--override_out` to `gen_evalplus.py`; set `HUMANEVAL_OVERRIDE_PATH` / `MBPP_OVERRIDE_PATH` |
