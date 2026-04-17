# Spec: Mistral 7B + CodeLlama 7B 实验

## 目标

在两个"弱-中等"代码能力的 AR 模型上验证 CoCoder：

| 模型 | 预期 HE+ | 特点 |
|---|---|---|
| Mistral-7B-Instruct-v0.3 | ~31%（已测） | 通用 instruct，草稿已有，无需重新生成 |
| CodeLlama-7b-Instruct-hf | ~40-48%（估计） | 代码专用 instruct，需下载 + 新模型类 |

**Mistral** 的意义：快速零成本测一个通用弱模型；如果 CoCoder 对通用模型无效，进一步证实"代码专用模型才适合 CoCoder"的结论。  
**CodeLlama** 的意义：代码专用 instruct，能力低于 DeepSeek，测试 CoCoder 对弱代码模型是否同样有效。

---

## 前提条件

```bash
source /home/tteng/miniconda3/etc/profile.d/conda.sh
conda activate code
cd /model/tteng/CoCoder
export PYTHONPATH=src
```

---

## Part A：Mistral 7B（无需 AR 生成，直接跑 remask）

草稿已存在：
- `outputs/base_tuteng/mistral_humaneval.jsonl`（164 条）
- `outputs/base_tuteng/mistral_mbpp.jsonl`（378 条）

### Step A1: remask 四路

```bash
# mistral + Dream, HumanEval
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_remask \
  --refiner dream \
  --input outputs/base_tuteng/mistral_humaneval.jsonl \
  --out outputs/base_tuteng/mistral_dream_remask_humaneval_t0.9.jsonl \
  --confidence_threshold 0.9 --device cuda:0

# mistral + Dream, MBPP
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_remask \
  --refiner dream \
  --input outputs/base_tuteng/mistral_mbpp.jsonl \
  --out outputs/base_tuteng/mistral_dream_remask_mbpp_t0.9.jsonl \
  --confidence_threshold 0.9 --device cuda:0

# mistral + LLaDA, HumanEval
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_remask \
  --refiner llada \
  --input outputs/base_tuteng/mistral_humaneval.jsonl \
  --out outputs/base_tuteng/mistral_llada_remask_humaneval_t0.9.jsonl \
  --confidence_threshold 0.9 --device cuda:0

# mistral + LLaDA, MBPP
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_remask \
  --refiner llada \
  --input outputs/base_tuteng/mistral_mbpp.jsonl \
  --out outputs/base_tuteng/mistral_llada_remask_mbpp_t0.9.jsonl \
  --confidence_threshold 0.9 --device cuda:0
```

### Step A2: 评测四路 collab

```bash
for slug in \
  mistral_dream_remask_humaneval_t0.9 \
  mistral_dream_remask_mbpp_t0.9 \
  mistral_llada_remask_humaneval_t0.9 \
  mistral_llada_remask_mbpp_t0.9; do

  dataset=$(echo $slug | grep -oP "(humaneval|mbpp)")
  python -m coder.scripts.eval_evalplus \
    --samples outputs/base_tuteng/${slug}.jsonl \
    --dataset $dataset \
    --model $slug
done
```

产物：`outputs/base_tuteng/mistral_{dream,llada}_remask_{humaneval,mbpp}_t0.9_summary.json` × 4

### Step A3: 注册进 gen_results_table.py

文件：`src/coder/scripts/gen_results_table.py`

在 `_PAIR_TIMING` dict（约第 298 行）中添加：

```python
# HumanEval
"mistral_dream_humaneval_t0.9":
    OUTPUTS / "mistral_dream_remask_humaneval_t0.9.jsonl.timing_summary.json",
"mistral_llada_humaneval_t0.9":
    OUTPUTS / "mistral_llada_remask_humaneval_t0.9.jsonl.timing_summary.json",
# MBPP
"mistral_dream_mbpp_t0.9":
    OUTPUTS / "mistral_dream_remask_mbpp_t0.9.jsonl.timing_summary.json",
"mistral_llada_mbpp_t0.9":
    OUTPUTS / "mistral_llada_remask_mbpp_t0.9.jsonl.timing_summary.json",
```

### Step A4: 注册进 model_pairs_evalplus.py

文件：`src/coder/scripts/model_pairs_evalplus.py`

在 `PAIR_CONFIGS` 末尾添加：

```python
# ── Mistral 7B ─────────────────────────────────────────────────────────────
PairConfig(
    slug="mistral_dream_humaneval_t0.9",
    dataset="humaneval",
    tau_conf=0.9,
    ar_label="Mistral 7B",
    refiner_label="Dream-Coder 7B",
    ar_input="outputs/base_tuteng/mistral_humaneval.jsonl",
    ar_summary="outputs/base_tuteng/mistral_humaneval_summary.json",
    collab_output="outputs/base_tuteng/mistral_dream_remask_humaneval_t0.9.jsonl",
    collab_summary="outputs/base_tuteng/mistral_dream_remask_humaneval_t0.9_summary.json",
    refiner="dream",
    refiner_model_id="Dream-org/Dream-Coder-v0-Instruct-7B",
),
PairConfig(
    slug="mistral_dream_mbpp_t0.9",
    dataset="mbpp",
    tau_conf=0.9,
    ar_label="Mistral 7B",
    refiner_label="Dream-Coder 7B",
    ar_input="outputs/base_tuteng/mistral_mbpp.jsonl",
    ar_summary="outputs/base_tuteng/mistral_mbpp_summary.json",
    collab_output="outputs/base_tuteng/mistral_dream_remask_mbpp_t0.9.jsonl",
    collab_summary="outputs/base_tuteng/mistral_dream_remask_mbpp_t0.9_summary.json",
    refiner="dream",
    refiner_model_id="Dream-org/Dream-Coder-v0-Instruct-7B",
),
PairConfig(
    slug="mistral_llada_humaneval_t0.9",
    dataset="humaneval",
    tau_conf=0.9,
    ar_label="Mistral 7B",
    refiner_label="LLaDA 8B",
    ar_input="outputs/base_tuteng/mistral_humaneval.jsonl",
    ar_summary="outputs/base_tuteng/mistral_humaneval_summary.json",
    collab_output="outputs/base_tuteng/mistral_llada_remask_humaneval_t0.9.jsonl",
    collab_summary="outputs/base_tuteng/mistral_llada_remask_humaneval_t0.9_summary.json",
    refiner="llada",
    refiner_model_id="GSAI-ML/LLaDA-8B-Instruct",
),
PairConfig(
    slug="mistral_llada_mbpp_t0.9",
    dataset="mbpp",
    tau_conf=0.9,
    ar_label="Mistral 7B",
    refiner_label="LLaDA 8B",
    ar_input="outputs/base_tuteng/mistral_mbpp.jsonl",
    ar_summary="outputs/base_tuteng/mistral_mbpp_summary.json",
    collab_output="outputs/base_tuteng/mistral_llada_remask_mbpp_t0.9.jsonl",
    collab_summary="outputs/base_tuteng/mistral_llada_remask_mbpp_t0.9_summary.json",
    refiner="llada",
    refiner_model_id="GSAI-ML/LLaDA-8B-Instruct",
),
```

---

## Part B：CodeLlama 7B Instruct（需下载 + 代码变更 + 全流程）

### Step B0: 下载模型（~14GB）

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('codellama/CodeLlama-7b-Instruct-hf')
"
```

### Step B1: 新增 codellama_coder.py

新建文件：`src/coder/models/codellama_coder.py`

```python
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from coder.models.base import CoderModel
from coder.utils.schema import ModelRequest


class CodeLlamaCoder(CoderModel):
    def __init__(
        self,
        model_id: str = "codellama/CodeLlama-7b-Instruct-hf",
        device: str = "cuda",
    ):
        self.model_id = model_id
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        ).to(device).eval()
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token is None and self.tok.eos_token is not None:
            self.tok.pad_token = self.tok.eos_token

    @property
    def name(self) -> str:
        return f"codellama_coder::{self.model_id}"

    @torch.inference_mode()
    def generate(self, req: ModelRequest) -> str:
        messages = [{"role": "user", "content": req.prompt}]
        prompt = self.tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)

        if req.seed is not None:
            torch.manual_seed(req.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(req.seed)

        do_sample = req.temperature is not None and req.temperature > 0
        out = self.model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=do_sample,
            temperature=max(req.temperature or 0.0, 1e-6),
            top_p=req.top_p,
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.pad_token_id,
        )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        gen = self.tok.decode(gen_ids, skip_special_tokens=True)
        return gen.strip()
```

### Step B2: 注册进 src/coder/models/__init__.py

在 `from .llama31_coder import Llama31Coder` 下方添加：

```python
from .codellama_coder import CodeLlamaCoder
```

在 `__all__` 中添加 `"CodeLlamaCoder"`。

### Step B3: 注册进 gen_evalplus.py

文件：`src/coder/scripts/gen_evalplus.py`

在 import 列表中添加 `CodeLlamaCoder`，在 `build_model()` 中添加：

```python
if name == "codellama":
    return CodeLlamaCoder(
        model_id=model_id or "codellama/CodeLlama-7b-Instruct-hf",
        device=device,
    )
```

### Step B4: 生成 AR 草稿

```bash
# HumanEval
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_evalplus \
  --model codellama \
  --out outputs/base_tuteng/codellama_humaneval.jsonl \
  --dataset humaneval --device cuda:0

# MBPP
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_evalplus \
  --model codellama \
  --out outputs/base_tuteng/codellama_mbpp.jsonl \
  --dataset mbpp --device cuda:0
```

### Step B5: 评测 AR baseline

```bash
python -m coder.scripts.eval_evalplus \
  --samples outputs/base_tuteng/codellama_humaneval.jsonl \
  --dataset humaneval --model codellama

python -m coder.scripts.eval_evalplus \
  --samples outputs/base_tuteng/codellama_mbpp.jsonl \
  --dataset mbpp --model codellama
```

产物：`codellama_humaneval_summary.json`，`codellama_mbpp_summary.json`

### Step B6: remask 四路

```bash
# codellama + Dream, HumanEval
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_remask \
  --refiner dream \
  --input outputs/base_tuteng/codellama_humaneval.jsonl \
  --out outputs/base_tuteng/codellama_dream_remask_humaneval_t0.9.jsonl \
  --confidence_threshold 0.9 --device cuda:0

# codellama + Dream, MBPP
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_remask \
  --refiner dream \
  --input outputs/base_tuteng/codellama_mbpp.jsonl \
  --out outputs/base_tuteng/codellama_dream_remask_mbpp_t0.9.jsonl \
  --confidence_threshold 0.9 --device cuda:0

# codellama + LLaDA, HumanEval
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_remask \
  --refiner llada \
  --input outputs/base_tuteng/codellama_humaneval.jsonl \
  --out outputs/base_tuteng/codellama_llada_remask_humaneval_t0.9.jsonl \
  --confidence_threshold 0.9 --device cuda:0

# codellama + LLaDA, MBPP
CUDA_VISIBLE_DEVICES=<GPU> python -m coder.scripts.gen_remask \
  --refiner llada \
  --input outputs/base_tuteng/codellama_mbpp.jsonl \
  --out outputs/base_tuteng/codellama_llada_remask_mbpp_t0.9.jsonl \
  --confidence_threshold 0.9 --device cuda:0
```

### Step B7: 评测四路 collab

```bash
for slug in \
  codellama_dream_remask_humaneval_t0.9 \
  codellama_dream_remask_mbpp_t0.9 \
  codellama_llada_remask_humaneval_t0.9 \
  codellama_llada_remask_mbpp_t0.9; do

  dataset=$(echo $slug | grep -oP "(humaneval|mbpp)")
  python -m coder.scripts.eval_evalplus \
    --samples outputs/base_tuteng/${slug}.jsonl \
    --dataset $dataset \
    --model $slug
done
```

### Step B8: 注册进 gen_results_table.py

**Standalone 条目**（在 `_STANDALONE_ENTRIES` 中，接在 `Mistral 7B` 条目后）：

```python
(
    "CodeLlama 7B",
    "codellama_humaneval_summary.json",
    "codellama_mbpp_summary.json",
    None,  # no LCB
    None,  # no BCB
    "codellama_humaneval.jsonl.timing_summary.json",
    "codellama_mbpp.jsonl.timing_summary.json",
),
```

**`_PAIR_TIMING` 条目**（接在 mistral 条目后）：

```python
# HumanEval
"codellama_dream_humaneval_t0.9":
    OUTPUTS / "codellama_dream_remask_humaneval_t0.9.jsonl.timing_summary.json",
"codellama_llada_humaneval_t0.9":
    OUTPUTS / "codellama_llada_remask_humaneval_t0.9.jsonl.timing_summary.json",
# MBPP
"codellama_dream_mbpp_t0.9":
    OUTPUTS / "codellama_dream_remask_mbpp_t0.9.jsonl.timing_summary.json",
"codellama_llada_mbpp_t0.9":
    OUTPUTS / "codellama_llada_remask_mbpp_t0.9.jsonl.timing_summary.json",
```

### Step B9: 注册进 model_pairs_evalplus.py

```python
# ── CodeLlama 7B ────────────────────────────────────────────────────────────
PairConfig(
    slug="codellama_dream_humaneval_t0.9",
    dataset="humaneval",
    tau_conf=0.9,
    ar_label="CodeLlama 7B",
    refiner_label="Dream-Coder 7B",
    ar_input="outputs/base_tuteng/codellama_humaneval.jsonl",
    ar_summary="outputs/base_tuteng/codellama_humaneval_summary.json",
    collab_output="outputs/base_tuteng/codellama_dream_remask_humaneval_t0.9.jsonl",
    collab_summary="outputs/base_tuteng/codellama_dream_remask_humaneval_t0.9_summary.json",
    refiner="dream",
    refiner_model_id="Dream-org/Dream-Coder-v0-Instruct-7B",
),
PairConfig(
    slug="codellama_dream_mbpp_t0.9",
    dataset="mbpp",
    tau_conf=0.9,
    ar_label="CodeLlama 7B",
    refiner_label="Dream-Coder 7B",
    ar_input="outputs/base_tuteng/codellama_mbpp.jsonl",
    ar_summary="outputs/base_tuteng/codellama_mbpp_summary.json",
    collab_output="outputs/base_tuteng/codellama_dream_remask_mbpp_t0.9.jsonl",
    collab_summary="outputs/base_tuteng/codellama_dream_remask_mbpp_t0.9_summary.json",
    refiner="dream",
    refiner_model_id="Dream-org/Dream-Coder-v0-Instruct-7B",
),
PairConfig(
    slug="codellama_llada_humaneval_t0.9",
    dataset="humaneval",
    tau_conf=0.9,
    ar_label="CodeLlama 7B",
    refiner_label="LLaDA 8B",
    ar_input="outputs/base_tuteng/codellama_humaneval.jsonl",
    ar_summary="outputs/base_tuteng/codellama_humaneval_summary.json",
    collab_output="outputs/base_tuteng/codellama_llada_remask_humaneval_t0.9.jsonl",
    collab_summary="outputs/base_tuteng/codellama_llada_remask_humaneval_t0.9_summary.json",
    refiner="llada",
    refiner_model_id="GSAI-ML/LLaDA-8B-Instruct",
),
PairConfig(
    slug="codellama_llada_mbpp_t0.9",
    dataset="mbpp",
    tau_conf=0.9,
    ar_label="CodeLlama 7B",
    refiner_label="LLaDA 8B",
    ar_input="outputs/base_tuteng/codellama_mbpp.jsonl",
    ar_summary="outputs/base_tuteng/codellama_mbpp_summary.json",
    collab_output="outputs/base_tuteng/codellama_llada_remask_mbpp_t0.9.jsonl",
    collab_summary="outputs/base_tuteng/codellama_llada_remask_mbpp_t0.9_summary.json",
    refiner="llada",
    refiner_model_id="GSAI-ML/LLaDA-8B-Instruct",
),
```

---

## 最终验证

```bash
cd /model/tteng/CoCoder
PYTHONPATH=src python -m coder.scripts.model_pairs_evalplus
PYTHONPATH=src python -m coder.scripts.gen_results_table
```

查看 `docs/results.md`：
- Standalone 表中出现 `CodeLlama 7B` 行
- Table 3 中出现 `Mistral 7B` 和 `CodeLlama 7B` 各 4 行（× Dream/LLaDA × HE/MBPP）

---

## 验收标准

| 检查项 | 期望值 |
|---|---|
| `mistral_dream_remask_humaneval_t0.9_summary.json` n_tasks | 164 |
| `mistral_dream_remask_mbpp_t0.9_summary.json` n_tasks | 378 |
| `codellama_humaneval_summary.json` n_tasks | 164 |
| `codellama_mbpp_summary.json` n_tasks | 378 |
| Table 3 Mistral 行数 | 4（Dream+LLaDA × HE+MBPP） |
| Table 3 CodeLlama 行数 | 4 |
