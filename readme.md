# Coder Benchmark Playground (Dream-Coder vs DeepSeek-Coder)

Compares two code-generation paradigms on EvalPlus and LiveBench-Coding:

- **Dream-Coder 7B** — diffusion LM (masked non-autoregressive)
- **DeepSeek-Coder 6.7B** — autoregressive LM

Benchmarks: **HumanEval+**, **MBPP+**, **LiveBench-Coding**

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
export PYTHONPATH=/home/kodai/CoCopilot/src
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
│   │   └── dream_coder.py       # DreamCoder: AutoModel + diffusion_generate
│   ├── datasets/
│   │   └── livebench_coding.py  # LiveBench-Coding dataset loader
│   └── utils/
│       └── schema.py            # ModelRequest, SampleRecord
│
├── scripts/
│   │ — Python scripts —
│   ├── gen_evalplus.py          # Generate HumanEval / MBPP completions → JSONL
│   ├── gen_remask.py            # DeepSeek draft → DreamCoder token remasking
│   ├── gen_livebench.py         # Generate LiveBench-Coding completions
│   ├── postprocess_evalplus.py  # Syntax-check + sanitize for EvalPlus format
│   ├── eval_evalplus.py         # Evaluate sanitized JSONL with evalplus
│   ├── eval_livebench.py        # Score LiveBench-Coding outputs
│   │ — Convenience shell scripts —
│   ├── 1_generate_deepseek.sh   # Step 1: DeepSeek generation (HumanEval + MBPP)
│   ├── 2_remask_dreamcoder.sh   # Step 2: DreamCoder remasking
│   ├── 3_sanitize.sh            # Step 3: Sanitize all outputs
│   ├── 4_evaluate.sh            # Step 4: Evaluate sanitized outputs
│   ├── 5_pipeline.sh            # Full pipeline (steps 1–4)
│   ├── gen.sh                   # Quick generation helper
│   ├── remask.sh                # Quick remask helper
│   ├── sanitize_and_eval.sh     # Combined sanitize + eval
│   └── eval.sh                  # Quick eval helper
│
├── outputs/                     # Generated files (gitignored large files)
│   ├── <model>_<dataset>.jsonl                    # Raw completions
│   ├── <model>_<dataset>-sanitized.jsonl          # Sanitized for evalplus
│   ├── <model>_<dataset>-sanitized_eval_results.json
│   ├── <model>_<dataset>_summary.json
│   ├── remask_<dataset>_t<thresh>.jsonl           # Remasked completions
│   ├── remask_<dataset>_t<thresh>-sanitized.jsonl
│   └── remask_<dataset>_t<thresh>_summary.json
│
├── requirements.txt
└── readme.md
```

---

## 3 Workflow

### 3.1 Standalone generation → evaluate

```bash
# Generate
python scripts/gen_evalplus.py --model [dream|deepseek] --dataset [humaneval|mbpp] \
  --out outputs/<name>.jsonl

# Sanitize
python scripts/postprocess_evalplus.py --dataset [humaneval|mbpp] \
  --samples outputs/<name>.jsonl
# → outputs/<name>-sanitized.jsonl

# Evaluate
python scripts/eval_evalplus.py --backend local --dataset [humaneval|mbpp] \
  --samples outputs/<name>-sanitized.jsonl
# → outputs/<name>-sanitized_eval_results.json + <model>_<dataset>_summary.json
```

### 3.2 Remasking pipeline (DeepSeek draft → DreamCoder refinement)

```bash
# Refine DeepSeek outputs with DreamCoder token remasking
python scripts/gen_remask.py \
  --input  outputs/deepseek_humaneval.jsonl \
  --out    outputs/remask_humaneval_t0.8.jsonl \
  --confidence_threshold 0.8 \
  --temperature 0.0 --top_p 1.0 --seed 3407
# Use --resume to continue an interrupted run.

# Then sanitize and evaluate as in 3.1.
```

### 3.3 Bash file for full pipeline (steps 1–4)

1. Generates DeepSeek outputs for HumanEval and MBPP
```bash
bash scripts/1_generate_deepseek.sh
```

2. Remasks with DreamCoder
```bash
bash scripts/2_remask_dreamcoder.sh
```

3. Sanitizes all outputs for evalplus
```bash
bash scripts/3_sanitize.sh
```
4. Evaluates with evalplus (local backend)
```bash
bash scripts/4_evaluate.sh
```

5. All-in-one pipeline (steps 1–4)

```bash
bash scripts/5_pipeline.sh
```

### 3.4 LiveBench-Coding

```bash
python scripts/gen_livebench.py  --model [dream|deepseek] --out outputs/<name>.jsonl
python scripts/eval_livebench.py --samples outputs/<name>.jsonl
```

---

## 4 Output File Conventions

| File | Description |
|------|-------------|
| `<model>_<dataset>.jsonl` | Raw: `task_id`, `prompt`, `raw_completion`, `solution`, `model`, `gen` |
| `remask_<dataset>_t<thresh>.jsonl` | Remasked: adds `draft_completion` field |
| `*-sanitized.jsonl` | Cleaned for evalplus ingestion |
| `*_eval_results.json` | Per-task pass/fail written by evalplus |
| `*_summary.json` | Aggregated pass@k metrics |

---

## 5 Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'coder'` | `export PYTHONPATH=/home/kodai/CoCopilot/src` or `pip install -e .` |
| Docker permission denied | `--backend local` in `eval_evalplus.py` |
| EvalPlus result file not found | `eval_evalplus.py` probes `_eval_results.json` and `-eval_results.json` naming variants |
| Subset evaluation | Pass `--override_out` to `gen_evalplus.py`; set `HUMANEVAL_OVERRIDE_PATH` / `MBPP_OVERRIDE_PATH` before evaluating |
