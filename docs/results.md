# 实验结果汇总

> 自动生成，勿手动编辑。更新命令：`python -m coder.scripts.gen_results_table`


## Standalone Models

| 模型 | HE+ plus% | HE+ base% | MBPP+ plus% | MBPP+ base% | LCB acc% | BCB pass@1% | s/sample (HE) | s/sample (MBPP) |
|---|---|---|---|---|---|---|---|---|
| DeepSeek-Coder 6.7B | 56.7% | 62.2% | 65.1% | 74.9% | 11.4% | 24.7% | — | — |
| Qwen2.5-Coder 7B | 77.4% | 82.3% | 73.0% | 83.1% | ❌ | 0.0%⚠️ | — | — |
| Llama-3.1 8B | 57.9% | 62.2% | 62.4% | 71.7% | ❌ | 0.0%⚠️ | — | — |
| Mistral 7B | 31.1% | 37.8% | 41.8% | 48.4% | ❌ | — | — | — |
| StarCoder2 7B | 23.2% | 26.2% | 28.3% | 32.8% | 0.0% | — | — | — |
| Dream-Coder 7B | 43.3% | 45.1% | 63.8% | 72.8% | ❌ | — | — | — |
| LLaDA 8B | 12.8% | 15.9% | 26.2% | 30.4% | ❌ | — | — | — |
| Seed-Coder 8B | 12.2% | 14.0% | 71.4% | 84.4% | ❌ | — | — | — |
| Seed-DiffCoder 8B | 65.9% | 70.7% | 73.3% | 85.7% | ❌ | — | — | — |
| DiffuLLaMA 7B | 3.7% | 3.7% | 2.6% | 3.2% | — | — | 15.4s | 15.2s |

> ⚠️ BCB 带 ⚠️ 标记的结果为 0.0%，原始评测结果可疑（疑似交互提示卡住），pass1_clean 版本正常。

> LCB ❌ = n_scored=0，评测未跑通（original_json 字段问题），仅 DeepSeek 有可信结果。

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
| mbpp | StarCoder2 7B | Dream-Coder 7B | 28.3% | — | — | — | 🔄 141/378 |

> 产物：`outputs/base_tuteng/model_pairs_all_t0.9.json`  —  更新命令：`python -m coder.scripts.model_pairs_evalplus`

> s/sample = collab 生成阶段（remask + denoising）的平均每题耗时。AR 草稿生成耗时未单独计入（gen_evalplus 尚未统计 timing）。

## Table 4 — DeepSeek-Coder Baselines（pass@1 plus%）

| 方法 | HE+ plus% | HE+ base% | MBPP+ plus% | MBPP+ base% | s/sample (HE) | s/sample (MBPP) |
|---|---|---|---|---|---|---|
| DeepSeek baseline | 56.7% | 62.2% | 65.1% | 74.9% | — | — |
| + Self-Refine | 70.1% | 76.2% | 68.5% | 77.5% | 4.6s | 6.1s |
| + Reflexion (w/ feedback) | 59.1% | 67.1% | 49.5% | 67.2% | 9.5s | 14.2s |
| + Rerank logprob k=8 | 68.9% | 76.8% | 66.9% | 77.8% | — | — |
| + Locate-AR-Rewrite | 68.9% | 76.8% | 67.7% | 77.8% | 7.2s | 3.1s |
| + LLaDA remask τ=0.9 | 65.9% | 72.6% | — | — | 15.3s | — |
| + Dream remask τ=0.9 (ours) | 72.6% | 78.7% | 70.1% | 80.4% | — | — |

> s/sample = 方法总耗时 / 题目数。baseline（gen_evalplus）尚未统计 timing，显示为 —。

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
| dream_livecodebench_pass1 | 69/1055 | — | 🔄 69/1055 |
| dream_bigcodebench_instruct_full_pass1 | 33/1140 | — | 🔄 33/1140 |
| collab_t0.9_livecodebench | 72/1055 | — | 🔄 72/1055 |
| collab_t0.9_bigcodebench_instruct_full | 85/1140 | — | 🔄 85/1140 |

> 更新命令：`python -m coder.scripts.run_extended_table --gpus <gpu_ids>`
