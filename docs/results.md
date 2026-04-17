# 实验结果汇总

> 自动生成，勿手动编辑。更新命令：`python -m coder.scripts.gen_results_table`


## Standalone Models

| 模型 | HE+ plus% | HE+ base% | MBPP+ plus% | MBPP+ base% | s/sample (HE) | s/sample (MBPP) |
|---|---|---|---|---|---|---|
| DeepSeek-Coder 6.7B | 56.7% | 62.2% | 65.1% | 74.9% | 6.9s | 5.2s |
| Qwen2.5-Coder 7B | 77.4% | 82.3% | 73.0% | 83.1% | 2.4s | 1.7s |
| Llama-3.1 8B | 57.9% | 62.2% | 62.4% | 71.7% | 2.8s | 1.3s |
| Mistral 7B | 31.1% | 37.8% | 41.8% | 48.4% | 14.2s | 12.0s |
| CodeLlama 7B | 25.6% | 29.3% | 30.7% | 38.4% | 2.8s | 3.7s |
| StarCoder2 7B | 23.2% | 26.2% | 28.3% | 32.8% | 17.7s | 17.4s |
| Dream-Coder 7B | 43.3% | 45.1% | 63.8% | 72.8% | 131.3s | 121.4s |
| LLaDA 8B | 12.8% | 15.9% | 26.2% | 30.4% | 21.4s | 18.0s |
| Seed-Coder 8B | 12.2% | 14.0% | 71.4% | 84.4% | 3.0s | 3.7s |
| Seed-Coder-Instruct 8B | 70.1% | 75.0% | 72.2% | 84.9% | 2.8s | 2.0s |
| Seed-DiffCoder 8B | 65.9% | 70.7% | 73.3% | 85.7% | 11.3s | 8.4s |
| DiffuLLaMA 7B | 3.7% | 3.7% | 2.6% | 3.2% | 15.4s | 15.2s |

## Table 3 — Model Pairs（τ=0.9, pass@1 plus%）

> 注：当前所有实验 Locator 与 Rewriter 均为同一 dLLM（均等于"dLLM 精炼"列原始含义）。

| Dataset | AR 草稿 | Locator | Rewriter | AR-only | Collab | Δ | 修对(+) | 弄坏(-) | s/sample | 状态 |
|---|---|---|---|---|---|---|---|---|---|---|
| humaneval | DeepSeek-Coder 6.7B | Dream-Coder 7B | Dream-Coder 7B | 56.7% | 72.6% | +15.9pp | 27 | 1 | 21.6s | ✅ |
| humaneval | Qwen2.5-Coder 7B | Dream-Coder 7B | Dream-Coder 7B | 77.4% | 76.8% | -0.6pp | 0 | 1 | 8.9s | ✅ |
| humaneval | Llama-3.1 8B | Dream-Coder 7B | Dream-Coder 7B | 57.9% | 57.3% | -0.6pp | 0 | 1 | 16.1s | ✅ |
| humaneval | StarCoder2 7B | Dream-Coder 7B | Dream-Coder 7B | 23.2% | 23.2% | +0.0pp | 0 | 0 | 63.9s | ✅ |
| humaneval | Mistral 7B | Dream-Coder 7B | Dream-Coder 7B | 31.1% | 32.3% | +1.2pp | 2 | 0 | 21.8s | ✅ |
| humaneval | CodeLlama 7B | Dream-Coder 7B | Dream-Coder 7B | 25.6% | 34.1% | +8.5pp | 16 | 2 | 11.1s | ✅ |
| humaneval | DeepSeek-Coder 6.7B | LLaDA 8B | LLaDA 8B | 56.7% | 65.9% | +9.2pp | 21 | 6 | 22.2s | ✅ |
| humaneval | Qwen2.5-Coder 7B | LLaDA 8B | LLaDA 8B | 77.4% | 77.4% | +0.0pp | 0 | 0 | 8.8s | ✅ |
| humaneval | Llama-3.1 8B | LLaDA 8B | LLaDA 8B | 57.9% | 56.1% | -1.8pp | 0 | 3 | 16.4s | ✅ |
| humaneval | StarCoder2 7B | LLaDA 8B | LLaDA 8B | 23.2% | 23.2% | +0.0pp | 0 | 0 | 39.2s | ✅ |
| humaneval | Mistral 7B | LLaDA 8B | LLaDA 8B | 31.1% | 31.1% | +0.0pp | 2 | 2 | 22.1s | ✅ |
| humaneval | CodeLlama 7B | LLaDA 8B | LLaDA 8B | 25.6% | 32.3% | +6.7pp | 16 | 5 | 10.9s | ✅ |
| humaneval | Seed-Coder-Instruct 8B | Dream-Coder 7B | Dream-Coder 7B | 70.1% | 65.2% | -4.9pp | 10 | 18 | 10.2s | ✅ |
| humaneval | Seed-Coder-Instruct 8B | LLaDA 8B | LLaDA 8B | 70.1% | 62.8% | -7.3pp | 10 | 22 | 10.3s | ✅ |
|  |  |  |  |  |  |  |  |  |  |  |
| mbpp | DeepSeek-Coder 6.7B | Dream-Coder 7B | Dream-Coder 7B | 65.1% | 70.1% | +5.0pp | 20 | 1 | 15.2s | ✅ |
| mbpp | Qwen2.5-Coder 7B | Dream-Coder 7B | Dream-Coder 7B | 73.0% | 72.2% | -0.8pp | 1 | 4 | 10.5s | ✅ |
| mbpp | Llama-3.1 8B | Dream-Coder 7B | Dream-Coder 7B | 62.4% | 64.0% | +1.6pp | 8 | 2 | 5.8s | ✅ |
| mbpp | StarCoder2 7B | Dream-Coder 7B | Dream-Coder 7B | 28.3% | 33.1% | +4.8pp | 19 | 1 | 63.4s | ✅ |
| mbpp | Mistral 7B | Dream-Coder 7B | Dream-Coder 7B | 41.8% | 42.6% | +0.8pp | 4 | 1 | 17.0s | ✅ |
| mbpp | CodeLlama 7B | Dream-Coder 7B | Dream-Coder 7B | 30.7% | 43.4% | +12.7pp | 49 | 1 | 8.6s | ✅ |
| mbpp | DeepSeek-Coder 6.7B | LLaDA 8B | LLaDA 8B | 65.1% | 68.0% | +2.9pp | 12 | 1 | 11.7s | ✅ |
| mbpp | Qwen2.5-Coder 7B | LLaDA 8B | LLaDA 8B | 73.0% | 73.0% | +0.0pp | 0 | 0 | 7.1s | ✅ |
| mbpp | Llama-3.1 8B | LLaDA 8B | LLaDA 8B | 62.4% | 53.2% | -9.2pp | 3 | 38 | 6.7s | ✅ |
| mbpp | StarCoder2 7B | LLaDA 8B | LLaDA 8B | 28.3% | 28.8% | +0.5pp | 2 | 0 | 35.7s | ✅ |
| mbpp | Mistral 7B | LLaDA 8B | LLaDA 8B | 41.8% | 36.2% | -5.6pp | 0 | 21 | 18.1s | ✅ |
| mbpp | CodeLlama 7B | LLaDA 8B | LLaDA 8B | 30.7% | 39.2% | +8.5pp | 45 | 13 | 8.7s | ✅ |
| mbpp | Seed-Coder-Instruct 8B | Dream-Coder 7B | Dream-Coder 7B | 72.2% | 72.2% | +0.0pp | 2 | 2 | 7.5s | ✅ |
| mbpp | Seed-Coder-Instruct 8B | LLaDA 8B | LLaDA 8B | 72.2% | 64.6% | -7.6pp | 1 | 30 | 8.0s | ✅ |

> 产物：`outputs/base_tuteng/model_pairs_all_t0.9.json`  —  更新命令：`python -m coder.scripts.model_pairs_evalplus`

> s/sample = AR 草稿生成 + remask + dLLM denoising 的全流程平均每题耗时。

> Locator / Rewriter 拆分：当前实验均以同一 dLLM 同时充当两角色；后续 math 实验中可独立配置。

## Table 4 — AR Model Baselines（pass@1 plus%）

| AR 模型 | 方法 | HE+ plus% | HE+ base% | MBPP+ plus% | MBPP+ base% | s/sample (HE) | s/sample (MBPP) |
|---|---|---|---|---|---|---|---|
| DeepSeek-Coder 6.7B | DeepSeek baseline | 56.7% | 62.2% | 65.1% | 74.9% | — | — |
|  | + Self-Refine | 70.1% | 76.2% | 68.5% | 77.5% | 4.6s | 6.1s |
|  | + Reflexion (w/ feedback) | 59.1% | 67.1% | 49.5% | 67.2% | 9.5s | 14.2s |
|  | + Rerank logprob k=8 | 68.9% | 76.8% | 66.9% | 77.8% | 61.7s | 48.7s |
|  | + Locate-AR-Rewrite | 68.9% | 76.8% | 67.7% | 77.8% | 7.2s | 3.1s |
|  | + LLaDA remask τ=0.9 | 65.9% | 72.6% | 68.0% | 78.0% | 15.3s | 6.5s |
|  | + Dream remask τ=0.9 (ours) | 72.6% | 78.7% | 70.1% | 80.4% | 14.7s | 10.0s |
|  |  |  |  |  |  |  |  |
| Qwen2.5-Coder 7B | Qwen baseline | 77.4% | 82.3% | 73.0% | 83.1% | 2.4s | 1.7s |
|  | + Self-Refine | 79.9% | 86.6% | 71.2% | 82.8% | 4.9s | 3.5s |
|  | + Reflexion (w/ feedback) | 70.1% | 73.2% | 60.6% | 74.3% | 17.2s | 13.8s |
|  | + Rerank logprob k=8 | 79.9% | 84.1% | 73.8% | 84.9% | 29.2s | 14.9s |
|  | + Locate-AR-Rewrite | 72.6% | 77.4% | 69.3% | 78.6% | 2.2s | 1.6s |
|  | + LLaDA remask τ=0.9 | 77.4% | 82.3% | 73.0% | 83.1% | 6.4s | 5.4s |
|  | + Dream remask τ=0.9 (ours) | 76.8% | 81.7% | 72.2% | 82.3% | 6.5s | 8.8s |
|  |  |  |  |  |  |  |  |
| Llama-3.1 8B | Llama-3.1 baseline | 57.9% | 62.2% | 62.4% | 71.7% | 2.8s | 1.3s |
|  | + Self-Refine | 54.9% | 59.1% | 55.8% | 68.8% | — | 1.7s |
|  | + Reflexion (w/ feedback) | 39.0% | 48.8% | 37.3% | 51.9% | 14.3s | 11.8s |
|  | + Rerank logprob k=8 | — | — | — | — | — | — |
|  | + Locate-AR-Rewrite | — | — | — | — | — | — |
|  | + LLaDA remask τ=0.9 | 56.1% | 59.8% | 53.2% | 62.4% | 13.6s | 5.4s |
|  | + Dream remask τ=0.9 (ours) | 57.3% | 62.2% | 64.0% | 73.3% | 13.3s | 4.5s |
|  |  |  |  |  |  |  |  |
| StarCoder2 7B | StarCoder2 baseline | 23.2% | 26.2% | 28.3% | 32.8% | 17.7s | 17.4s |
|  | + Self-Refine | 7.9% | 7.9% | 12.2% | 16.9% | — | 16.3s |
|  | + Reflexion (w/ feedback) | — | — | — | — | — | — |
|  | + Rerank logprob k=8 | — | — | — | — | — | — |
|  | + Locate-AR-Rewrite | — | — | — | — | — | — |
|  | + LLaDA remask τ=0.9 | 23.2% | 26.2% | 28.8% | 33.3% | 21.5s | 18.3s |
|  | + Dream remask τ=0.9 (ours) | 23.2% | 26.2% | 33.1% | 38.1% | 46.2s | 46.0s |

> s/sample = 方法总耗时 / 题目数。DeepSeek baseline timing 来自 `_timed` 重跑产物。

## Locator Ablation（DeepSeek-Coder + Dream refine）

| Locator | HE+ plus% | HE+ base% | MBPP+ plus% | MBPP+ base% | s/sample (HE) | s/sample (MBPP) |
|---|---|---|---|---|---|---|
| dLLM locator (ours) | 72.6% | 78.7% | 70.1% | 80.4% | 14.7s | 10.0s |
| AR logprob locator | 71.3% | 78.7% | 68.5% | 78.8% | 7.8s | 5.7s |
| CodeBERT locator | 69.5% | 76.2% | 68.5% | 78.3% | 7.7s | 5.5s |

> AR / CodeBERT locator rows use `confidence_threshold=0.9`; refine model remains Dream-Coder 7B.

## Locator Fault-Detection Analysis

> "Surgical fault pairs": 草稿失败 → remask 后通过，且改动 ≤10 字符的样本。 对每个 fault token 和 non-fault token 计算模型置信度，ratio = P(non-fault) / P(fault)，越高说明 locator 对错误位置的感知越敏锐。

> AR 草稿：DeepSeek-Coder 6.7B，τ=0.9，dedupe_task=True。

| Locator | HE P(fault) | HE P(non-fault) | HE ratio | MBPP P(fault) | MBPP P(non-fault) | MBPP ratio |
|---|---|---|---|---|---|---|
| DLLM (Dream-Coder) | 0.042 | 0.985 | **23.21x** | 0.008 | 0.981 | **126.44x** |
| AR logprob (DeepSeek) | 0.719 | 0.957 | **1.33x** | 0.939 | 0.959 | **1.02x** |
| MLM (CodeBERT) | 0.686 | 0.813 | **1.18x** | 0.842 | 0.846 | **1.00x** |

> 产物：`outputs/ablation_locator/locator_fault_detection_summary.json`  —  源 log：`outputs/ablation_locator/locator_scoring_clean_t09_deepseek.log`

> dLLM locator 对 fault token 置信度极低（HE P≈0.04，MBPP P≈0.008），与 non-fault token 差距悬殊；AR 和 MLM locator 几乎无区分（ratio≈1x）。

## Math Benchmarks（AR 模型）

| 模型 | GSM8K acc% | s/sample | MATH500 acc% | s/sample |
|---|---|---|---|---|
| DeepSeek-Coder 6.7B | 19.0% | 8.0s | 4.6% | 11.6s |
| Qwen2.5-Coder 7B | 30.6% | 2.8s | 37.6% | 9.4s |
| Llama-3.1 8B | 84.5% | 7.9s | 38.6% | 17.9s |
| Mistral 7B | — | — | — | — |
| StarCoder2 7B | — | — | — | — |
|  |  |  |  |  |
| DeepSeek-Coder + Dream | 18.2% | — | 3.2% | — |
| Qwen2.5-Coder + Dream | 24.3% | — | — | — |
| Llama-3.1 + Dream | 83.7% | — | 38.6% | — |

> GSM8K：1319 道小学数学题（test set）。MATH500：500 道竞赛数学题（MATH 数据集子集）。

> 上半部分：AR 模型独立推理 baseline；下半部分：CoCoder（AR草稿 + Dream-Coder remask τ=0.9）协作结果，整体不提升。

### MATH500 Subject Breakdown

| 模型 | Algebra | Prealgebra | Precalculus | Intermediate Algebra | Number Theory | Geometry | C&P |
|---|---|---|---|---|---|---|---|
| DeepSeek-Coder 6.7B | 8/124 (6%) | 6/82 (7%) | 2/56 (4%) | 0/97 (0%) | 3/62 (5%) | 2/41 (5%) | 2/38 (5%) |
| Qwen2.5-Coder 7B | 62/124 (50%) | 38/82 (46%) | 11/56 (20%) | 27/97 (28%) | 23/62 (37%) | 12/41 (29%) | 15/38 (39%) |
| Llama-3.1 8B | 75/124 (60%) | 40/82 (49%) | 5/56 (9%) | 18/97 (19%) | 32/62 (52%) | 12/41 (29%) | 11/38 (29%) |
| Mistral 7B | — | — | — | — | — | — | — |
| StarCoder2 7B | — | — | — | — | — | — | — |

## General Domain Benchmarks（closed-book research QA）

> Dream-General = Dream-v0-Instruct-7B (text dLLM, not Dream-Coder). CoCoder = Llama-3.1 draft + Dream-General remask τ=0.9. Closed-book: no retrieval.

### FRAMES（multi-hop research QA, n=824）

| 模型 | n | EM% | Token F1% |
|---|---|---|---|
| Llama-3.1 8B (AR) | 824 | 0.0% | 4.4% |
| Dream-General 7B (dLLM) | 824 | 1.1% | 11.4% |
| CoCoder τ=0.9 (Llama+Dream) | 824 | 0.0% | 4.4% |

### HotpotQA（multi-hop QA distractor val, n=1000）

| 模型 | n | EM% | Token F1% |
|---|---|---|---|
| Llama-3.1 8B (AR) | 1000 | 13.5% | 22.4% |
| Dream-General 7B (dLLM) | 1000 | 16.4% | 25.4% |
| CoCoder τ=0.9 (Llama+Dream) | 1000 | 9.7% | 18.9% |

> WildBench Writing (n=146) 生成已完成（llama31 / dream_general），eval 需 LLM judge（API key），CoCoder run 仍在进行（37/146）。

## τ 敏感性分析（DeepSeek-Coder + Dream-Coder）

> AR baseline：HE+ plus=56.7%，MBPP+ plus=65.1%。

| τ | HE+ plus% | HE+ base% | MBPP+ plus% | MBPP+ base% |
|---|---|---|---|---|
| 0.7 | 71.3% | 77.4% | 70.1% | 80.4% |
| 0.8 | 71.3% | 77.4% | 70.1% | 80.4% |
| 0.9 | 72.6% | 78.7% | 70.1% | 80.4% |
| 0.93 | 72.6% | 78.7% | 70.1% | 80.4% |
| 0.95 | 72.6% | 78.7% | 70.1% | 80.4% |
| 0.97 | 72.6% | 79.3% | 70.1% | 80.4% |
| 0.99 | 72.6% | 79.3% | 69.8% | 80.2% |

> 产物来自 `outputs/remask_kodai/`（DeepSeek 草稿 + Dream-Coder 精炼）。

> 其他 AR 模型 × τ 组合尚未系统扫描。

## Table 2 — Extended Benchmarks

### LiveCodeBench (accuracy%)

| 模型 | n_scored | accuracy% | 状态 |
|---|---|---|---|
| DeepSeek-Coder 6.7B | 1055 | 11.37% | ✅ |
| Qwen2.5-Coder 7B | 1055 | 22.56% | ✅ |
| Llama-3.1 8B | 1055 | 7.96% | ✅ |
| Dream-Coder 7B | 71 | 0.00% | ✅ |
| LLaDA 8B | — | — | ❌ n_scored=0 |
| StarCoder2 7B | 1055 | 0.00% | ✅ |
| Collab τ=0.9 (n=100) | 100 | 12.00% | ✅ |
| Dream (n=100) | 100 | 4.00% | ✅ |
| DeepSeek (n=100) | 100 | 12.00% | ✅ |

### BigCodeBench（instruct, full, pass@1%）

| 模型 | pass@1% | 状态 |
|---|---|---|
| DeepSeek-Coder 6.7B | 24.7% | ✅ |
| Qwen2.5-Coder 7B | 38.0% | ✅ |
| Llama-3.1 8B | 19.0% | ✅ |
| Collab τ=0.9 (n=100) | — | ❌ |
| Dream (n=100) | — | ❌ |
| DeepSeek (n=100) | — | ❌ |

> 以上为 pass1_clean 结果（strip markdown fencing）。raw 版本均 0.0%（见 pitfalls.md）。

### Extended Table Shards 进度

| 实验 | 进度 (done/total) | pass@1% | 状态 |
|---|---|---|---|
| dream_livecodebench_pass1 | 1055/1055 | 2.9% | ✅ |
| dream_bigcodebench_instruct_full_pass1 | 1140/1140 | 22.5% | ✅ |
| collab_t0.9_livecodebench | 1055/1055 | 11.6% | ✅ |
| collab_t0.9_bigcodebench_instruct_full | 1140/1140 | 24.6% | ✅ |

> 更新命令：`python -m coder.scripts.run_extended_table --gpus <gpu_ids>`
