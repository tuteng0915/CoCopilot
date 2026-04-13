# 实验结果汇总

> 自动生成，勿手动编辑。更新命令：`python -m coder.scripts.gen_results_table`


## Standalone Models

| 模型 | HE+ plus% | HE+ base% | MBPP+ plus% | MBPP+ base% | s/sample (HE) | s/sample (MBPP) |
|---|---|---|---|---|---|---|
| DeepSeek-Coder 6.7B | 56.7% | 62.2% | 65.1% | 74.9% | 6.9s | 5.2s |
| Qwen2.5-Coder 7B | 77.4% | 82.3% | 73.0% | 83.1% | 2.4s | 1.7s |
| Llama-3.1 8B | 57.9% | 62.2% | 62.4% | 71.7% | 2.8s | 1.3s |
| Mistral 7B | 31.1% | 37.8% | 41.8% | 48.4% | 14.2s | 12.0s |
| StarCoder2 7B | 23.2% | 26.2% | 28.3% | 32.8% | 17.7s | 17.4s |
| Dream-Coder 7B | 43.3% | 45.1% | 63.8% | 72.8% | 131.3s | 121.4s |
| LLaDA 8B | 12.8% | 15.9% | 26.2% | 30.4% | 21.4s | 18.0s |
| Seed-Coder 8B | 12.2% | 14.0% | 71.4% | 84.4% | 3.0s | 3.7s |
| Seed-DiffCoder 8B | 65.9% | 70.7% | 73.3% | 85.7% | 11.3s | 8.4s |
| DiffuLLaMA 7B | 3.7% | 3.7% | 2.6% | 3.2% | 15.4s | 15.2s |

## Table 3 — Model Pairs（τ=0.9, pass@1 plus%）

| Dataset | AR 草稿 | dLLM 精炼 | AR-only | Collab | Δ | s/sample | 状态 |
|---|---|---|---|---|---|---|---|
| humaneval | DeepSeek-Coder 6.7B | Dream-Coder 7B | 56.7% | 72.6% | +15.9pp | — | ✅ |
| humaneval | Qwen2.5-Coder 7B | Dream-Coder 7B | 77.4% | 76.8% | -0.6pp | — | ✅ |
| humaneval | Llama-3.1 8B | Dream-Coder 7B | 57.9% | 57.3% | -0.6pp | 13.3s | ✅ |
| humaneval | StarCoder2 7B | Dream-Coder 7B | 23.2% | 23.2% | +0.0pp | — | ✅ |
| humaneval | DeepSeek-Coder 6.7B | LLaDA 8B | 56.7% | 65.9% | +9.2pp | 15.3s | ✅ |
|  |  |  |  |  |  |  |  |
| mbpp | DeepSeek-Coder 6.7B | Dream-Coder 7B | 65.1% | 70.1% | +5.0pp | — | ✅ |
| mbpp | Qwen2.5-Coder 7B | Dream-Coder 7B | 73.0% | 72.2% | -0.8pp | 8.8s | ✅ |
| mbpp | Llama-3.1 8B | Dream-Coder 7B | 62.4% | 64.0% | +1.6pp | 4.5s | ✅ |
| mbpp | StarCoder2 7B | Dream-Coder 7B | 28.3% | 33.1% | +4.8pp | 46.0s | ✅ |

> 产物：`outputs/base_tuteng/model_pairs_all_t0.9.json`  —  更新命令：`python -m coder.scripts.model_pairs_evalplus`

> s/sample = collab 生成阶段（remask + denoising）的平均每题耗时。AR 草稿生成耗时未单独计入（gen_evalplus 尚未统计 timing）。

## Table 4 — DeepSeek-Coder Baselines（pass@1 plus%）

| 方法 | HE+ plus% | HE+ base% | MBPP+ plus% | MBPP+ base% | s/sample (HE) | s/sample (MBPP) |
|---|---|---|---|---|---|---|
| DeepSeek baseline | 56.7% | 62.2% | 65.1% | 74.9% | — | — |
| + Self-Refine | 70.1% | 76.2% | 68.5% | 77.5% | 4.6s | 6.1s |
| + Reflexion (w/ feedback) | 59.1% | 67.1% | 49.5% | 67.2% | 9.5s | 14.2s |
| + Rerank logprob k=8 | 68.9% | 76.8% | 66.9% | 77.8% | 61.7s | 48.7s |
| + Locate-AR-Rewrite | 68.9% | 76.8% | 67.7% | 77.8% | 7.2s | 3.1s |
| + LLaDA remask τ=0.9 | 65.9% | 72.6% | — | — | 15.3s | — |
| + Dream remask τ=0.9 (ours) | 72.6% | 78.7% | 70.1% | 80.4% | — | — |

> s/sample = 方法总耗时 / 题目数。baseline timing 来自 `_timed` 重跑产物。

## Table 4b — Qwen2.5-Coder 7B Baselines（pass@1 plus%）

| 方法 | HE+ plus% | HE+ base% | MBPP+ plus% | MBPP+ base% | s/sample (HE) | s/sample (MBPP) |
|---|---|---|---|---|---|---|
| Qwen baseline | 77.4% | 82.3% | 73.0% | 83.1% | 2.4s | 1.7s |
| + Self-Refine | 79.9% | 86.6% | 71.2% | 82.8% | 4.9s | 3.5s |
| + Reflexion (w/ feedback) | 70.1% | 73.2% | 60.6% | 74.3% | 17.2s | 13.8s |
| + Rerank logprob k=8 | 79.9% | 84.1% | 73.8% | 84.9% | 29.2s | 14.9s |
| + Locate-AR-Rewrite | — | — | — | — | — | — |
| + LLaDA remask τ=0.9 | — | — | — | — | — | — |
| + Dream remask τ=0.9 (ours) | 76.8% | 81.7% | 72.2% | 82.3% | — | 8.8s |

> s/sample = 方法总耗时 / 题目数。

## Math Benchmarks（AR 模型）

| 模型 | GSM8K acc% | s/sample | MATH500 acc% | s/sample |
|---|---|---|---|---|
| DeepSeek-Coder 6.7B | 19.0% | 8.0s | 4.6% | 11.6s |
| Qwen2.5-Coder 7B | 30.6% | 2.8s | 37.6% | 9.4s |
| Llama-3.1 8B | 84.5% | 7.9s | 38.6% | 17.9s |
| Mistral 7B | — | — | — | — |
| StarCoder2 7B | — | — | — | — |

> GSM8K：1319 道小学数学题（test set）。MATH500：500 道竞赛数学题（MATH 数据集子集）。

> 仅列 AR 模型；dLLM（Dream-Coder、LLaDA）不适用于此评测。

### MATH500 Subject Breakdown

| 模型 | Algebra | Prealgebra | Precalculus | Intermediate Algebra | Number Theory | Geometry | C&P |
|---|---|---|---|---|---|---|---|
| DeepSeek-Coder 6.7B | 8/124 (6%) | 6/82 (7%) | 2/56 (4%) | 0/97 (0%) | 3/62 (5%) | 2/41 (5%) | 2/38 (5%) |
| Qwen2.5-Coder 7B | 62/124 (50%) | 38/82 (46%) | 11/56 (20%) | 27/97 (28%) | 23/62 (37%) | 12/41 (29%) | 15/38 (39%) |
| Llama-3.1 8B | 75/124 (60%) | 40/82 (49%) | 5/56 (9%) | 18/97 (19%) | 32/62 (52%) | 12/41 (29%) | 11/38 (29%) |
| Mistral 7B | — | — | — | — | — | — | — |
| StarCoder2 7B | — | — | — | — | — | — | — |

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
| Qwen2.5-Coder 7B | — | — | ❌ n_scored=0 |
| Llama-3.1 8B | — | — | ❌ n_scored=0 |
| Dream-Coder 7B | — | — | ❌ n_scored=0 |
| LLaDA 8B | — | — | ❌ n_scored=0 |
| StarCoder2 7B | 1055 | 0.00% | ✅ |
| Collab τ=0.9 (n=100) | 100 | 12.00% | ✅ |
| Dream (n=100) | 100 | 4.00% | ✅ |
| DeepSeek (n=100) | 100 | 12.00% | ✅ |

### BigCodeBench（instruct, full, pass@1%）

| 模型 | pass@1% | 状态 |
|---|---|---|
| DeepSeek-Coder 6.7B (pass1_clean) | 24.7% | ✅ |
| DeepSeek-Coder 6.7B (raw) | 0.0% | ❌ 0.0%⚠️ |
| Qwen2.5-Coder 7B (raw) | 0.0% | ❌ 0.0%⚠️ |
| Llama-3.1 8B (raw) | 0.0% | ❌ 0.0%⚠️ |
| Collab τ=0.9 (n=100) | — | ❌ |
| Dream (n=100) | — | ❌ |
| DeepSeek (n=100) | — | ❌ |

> ⚠️ raw 结果全部 0.0%，疑似评测时交互提示卡住（见 pitfalls.md）。pass1_clean 版本正常。

### Extended Table Shards 进度

| 实验 | 进度 (done/total) | pass@1% | 状态 |
|---|---|---|---|
| dream_livecodebench_pass1 | 1055/1055 | 2.9% | ✅ |
| dream_bigcodebench_instruct_full_pass1 | 1140/1140 | 22.5% | ✅ |
| collab_t0.9_livecodebench | 1055/1055 | 11.6% | ✅ |
| collab_t0.9_bigcodebench_instruct_full | 1140/1140 | 24.6% | ✅ |

> 更新命令：`python -m coder.scripts.run_extended_table --gpus <gpu_ids>`
