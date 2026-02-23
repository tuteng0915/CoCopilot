# Coder Benchmark Playground (Dream-Coder vs DeepSeek-Coder)

This repo compares:

- **Dream-Coder 7B** (dLLM coder)
- **DeepSeek-Coder 6.7B** (AR LLM coder)

with a unified generation pipeline and benchmark evaluation wrappers for:

- **EvalPlus** (`humaneval+`, `mbpp+`)
- **LiveBench-Coding** 

---

## 1 Environment Setup

```bash
conda create -n code python=3.10 -y
conda activate code
pip install -U pip
pip install torch transformers datasets tqdm evalplus
pip install -e .
pip install git+https://github.com/LiveBench/LiveBench.git
```

---

## 2 Main Scripts

### Generation
- `scripts/gen_evalplus.py` — generate HumanEval / MBPP samples (JSONL)
- `scripts/gen_livebench.py` — generate LiveBench-Coding samples (JSONL)
- `scripts/postprocess_evalplus.py` — syntax/sanitize for EvalPlus

### Eval
- `scripts/eval_evalplus.py` — wrapper for `evalplus.evaluate` (docker/local)
- `scripts/eval_livebench.py` — wrapper around LiveBench official coding scoring

---

## 3 Troubleshooting

### `ModuleNotFoundError: No module named 'coder'`
Use:
```bash
pip install -e .
```
or temporarily:
```bash
PYTHONPATH=src python ...
```

### Docker permission denied
Use local backend for EvalPlus:
```bash
python scripts/eval_evalplus.py --backend local ...
```