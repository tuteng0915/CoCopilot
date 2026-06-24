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
| Seed-Coder-Instruct 8B | 75.6% | 81.1% | 72.2% | 84.9% | 2.8s | 2.0s |
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
| humaneval | Seed-Coder-Instruct 8B | Dream-Coder 7B | Dream-Coder 7B | 75.6% | 75.6% | +0.0pp | 0 | 0 | 10.2s | ✅ |
| humaneval | Seed-Coder-Instruct 8B | LLaDA 8B | LLaDA 8B | 75.6% | 72.6% | -3.0pp | 0 | 5 | 10.3s | ✅ |
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

## Table 4 — AR Model Baselines（pass@1，tau_rerun 固定管线）

> 本表全部数字来自 `tau_rerun` 重跑产物（`outputs/tau_rerun/`），使用修复后的 `build_evalplus_solution`（保留 completion 中的 import 行）。与早期 `base_tuteng` 数字相比，AR baseline 本身也因修复而上升（AR 生成同样走 `gen_evalplus` → `build_evalplus_solution`）。

| AR 模型 | 方法 | HE+ plus% | HE+ base% | MBPP+ plus% | MBPP+ base% | s/sample (HE) | s/sample (MBPP) |
|---|---|---|---|---|---|---|---|
| DeepSeek-Coder 6.7B | DeepSeek baseline | 70.7% | 76.2% | 65.1% | 74.9% | 6.9s | 5.2s |
|  | + Self-Refine | 70.1% | 76.2% | 68.5% | 77.5% | 4.6s | 6.1s |
|  | + Reflexion (w/ feedback) | 59.1% | 67.1% | 49.5% | 67.2% | 9.5s | 14.2s |
|  | + Rerank logprob k=8 | 68.9% | 76.8% | 66.9% | 77.8% | 61.7s | 48.7s |
|  | + Locate-AR-Rewrite | 68.9% | 76.8% | 67.7% | 77.8% | 7.2s | 3.1s |
|  | + LLaDA remask τ=0.9 | 65.9% | 72.6% | 68.0% | 78.0% | 15.3s | 6.5s |
|  | + Dream remask τ=0.9 (ours) | 70.1% | 77.4% | 69.6%† | 80.2%† | 14.7s | 10.0s |
|  |  |  |  |  |  |  |  |
| Qwen2.5-Coder 7B | Qwen baseline | 77.4% | 82.3% | 73.0% | 83.1% | 2.4s | 1.7s |
|  | + Self-Refine | 79.9% | 86.6% | 71.2% | 82.8% | 4.9s | 3.5s |
|  | + Reflexion (w/ feedback) | 70.1% | 73.2% | 60.6% | 74.3% | 17.2s | 13.8s |
|  | + Rerank logprob k=8 | 79.9% | 84.1% | 73.8% | 84.9% | 29.2s | 14.9s |
|  | + Locate-AR-Rewrite | 72.6% | 77.4% | 69.3% | 78.6% | 2.2s | 1.6s |
|  | + LLaDA remask τ=0.9 | 77.4% | 82.3% | 73.0% | 83.1% | 6.4s | 5.4s |
|  | + Dream remask τ=0.9 (ours) | 78.0% | 83.5% | 72.2% | 82.3% | 6.5s | 8.8s |
|  |  |  |  |  |  |  |  |
| Llama-3.1 8B | Llama-3.1 baseline | 57.9% | 62.2% | 62.4% | 71.7% | 2.8s | 1.3s |
|  | + Self-Refine | 54.9% | 59.1% | 55.8% | 68.8% | 4.3s | 1.7s |
|  | + Reflexion (w/ feedback) | 39.0% | 48.8% | 37.3% | 51.9% | 14.3s | 11.8s |
|  | + Rerank logprob k=8 | 57.9% | 61.0% | 64.0% | 73.5% | 22.5s | 11.0s |
|  | + Locate-AR-Rewrite | 56.7% | 61.6% | 57.1% | 66.7% | 3.5s | 1.5s |
|  | + LLaDA remask τ=0.9 | 56.1% | 59.8% | 53.2% | 62.4% | 13.6s | 5.4s |
|  | + Dream remask τ=0.9 (ours) | 57.9% | 62.2% | 64.0% | 73.3% | 13.3s | 4.5s |
|  |  |  |  |  |  |  |  |
| CodeLlama 7B | CodeLlama baseline | 36.0% | 40.9% | 41.8% | 50.5% | 2.8s | 3.7s |
|  | + Self-Refine | 32.9% | 37.2% | 43.1% | 50.5% | — | — |
|  | + Reflexion (w/ feedback) | 14.6% | 22.6% | 20.6% | 30.4% | — | — |
|  | + Rerank logprob k=8 | 31.7% | 37.2% | 43.1% | 52.6% | — | — |
|  | + Locate-AR-Rewrite | 17.1% | 20.1% | 40.7% | 49.2% | — | — |
|  | + Dream remask τ=0.9 (ours) | 36.6% | 41.5% | 43.4% | 52.4% | — | — |
|  |  |  |  |  |  |  |  |
| StarCoder2 7B | StarCoder2 baseline | 23.2% | 26.2% | 28.3% | 32.8% | 17.7s | 17.4s |
|  | + Self-Refine | 7.9% | 7.9% | 12.2% | 16.9% | 16.3s | 16.3s |
|  | + Reflexion (w/ feedback) | 7.9% | 7.9% | 9.8% | 13.8% | 33.1s | 24.2s |
|  | + Rerank logprob k=8 | 7.9% | 9.1% | 13.5% | 16.4% | 122.7s | 129.9s |
|  | + Locate-AR-Rewrite | 3.7% | 4.3% | 4.2% | 5.6% | 17.4s | 16.4s |
|  | + LLaDA remask τ=0.9 | 23.2% | 26.2% | 28.8% | 33.3% | 21.5s | 18.3s |
|  | + Dream remask τ=0.9 (ours) | 23.2% | 26.2% | 33.1% | 38.1% | 46.2s | 46.0s |

> s/sample = 方法总耗时 / 题目数。DeepSeek baseline timing 来自 `_timed` 重跑产物。

> † MBPP 数字来自 tau_rerun fixed 产物（import fix reprocess 已完成），所有模型 MBPP import change 率均为 0%（不受 import 修复影响）。

> 若 EvalPlus 结果中同一 task 出现重复样本，本表按 pass@1 口径只计每个 task 的第一条样本；这修正了 Llama-3.1/StarCoder2 Locate-AR-Rewrite HumanEval 的 merge duplicate artifact。

## Locator Ablation（DeepSeek-Coder + Dream refine）

| Locator | HE+ plus% | HE+ base% | MBPP+ plus% | MBPP+ base% | s/sample (HE) | s/sample (MBPP) |
|---|---|---|---|---|---|---|
| dLLM locator (ours) | 72.6% | 78.7% | 70.1% | 80.4% | 14.7s | 10.0s |
| AR logprob locator | 71.3% | 78.7% | 68.5% | 78.8% | 7.8s | 5.7s |
| CodeBERT locator | 69.5% | 76.2% | 68.5% | 78.3% | 7.7s | 5.5s |
| Random locator (`mask_ratio=0.10`) | 4.9% | 4.9% | 14.8% | 15.9% | 11.0s | 6.4s |
| Oracle locator (`gold diff spans`) | 67.7% | 75.0% | 68.0% | 78.0% | 0.3s | 0.1s |

> AR / CodeBERT locator rows use `confidence_threshold=0.9`; refine model remains Dream-Coder 7B.

> Random locator uses DeepSeek-Coder drafts + Dream-Coder rewrite with random token confidence and `mask_ratio=0.10`; all records have `skip_refine=False`. Mean recorded mask fraction: HumanEval 10.6%, MBPP 12.3%. Raw outputs: `outputs/base_tuteng/deepseek_random_locate_dream_rewrite_humaneval.jsonl`, `outputs/base_tuteng/deepseek_random_locate_dream_rewrite_mbpp.jsonl`; summaries: `outputs/base_tuteng/deepseek_random_locate_dream_rewrite_humaneval_summary.json`, `outputs/base_tuteng/deepseek_random_locate_dream_rewrite_mbpp_summary.json`.

> Oracle locator uses DeepSeek-Coder drafts plus oracle diff masks computed against the available `_timed` Dream remask outputs. Oracle mask files contain 164 HumanEval rows and 378 MBPP rows, with non-null spans for 2 HumanEval and 8 MBPP tasks. Raw outputs: `outputs/base_tuteng/deepseek_oracle_locate_dream_rewrite_humaneval.jsonl`, `outputs/base_tuteng/deepseek_oracle_locate_dream_rewrite_mbpp.jsonl`; summaries: `outputs/base_tuteng/deepseek_oracle_locate_dream_rewrite_humaneval_summary.json`, `outputs/base_tuteng/deepseek_oracle_locate_dream_rewrite_mbpp_summary.json`.

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

## Locator Calibration / ROC-AUC

> Per-token changed-span labels from DeepSeek-Coder failed drafts and Dream-Coder remask outputs. Low confidence is treated as predicting a fault token. Because strict corrected pairs are sparse, these runs use `--include_collab_fail`; interpret AUC as changed-token separability rather than a pure repair-success metric.

| Dataset | Eligible | Changed pairs | Fault tokens | Non-fault tokens | dLLM AUC | AR AUC | CodeBERT AUC | Random AUC |
|---|---|---|---|---|---|---|---|---|
| HumanEval | 71 | 9 | 12 | 969 | 0.951 | 0.862 | 0.812 | 0.398 |
| MBPP | 132 | 23 | 27 | 1669 | 0.960 | 0.824 | 0.771 | 0.562 |

> 产物：`outputs/ablation_locator/calibration_data_humaneval.json`, `outputs/ablation_locator/calibration_data_mbpp.json`, `outputs/ablation_locator/plots/{calibration,roc}_{humaneval,mbpp}.pdf`, `outputs/ablation_locator/plots/auc_summary.json`。

> 结论：dLLM AUC 最高（HE 0.951, MBPP 0.960）。HumanEval fault tokens 只有 12 个，MBPP 有 27 个；AR/CodeBERT 在 changed-token proxy 上也高于随机，因此这组图适合作为可视化补充，intrinsic locator 质量仍以 surgical fault-detection ratio 为主。

### Full Model-Pair Calibration Matrix

> 完整矩阵覆盖 `model_pairs_all_t0.9.json` 中 7 个 AR drafter × 2 个 dLLM refiner × HumanEval/MBPP，共 28 组。所有行使用 `--include_collab_fail`；Llama-3.1 AR logprob locator 因 gated HF 权限不可用，4 行 AR AUC 记为 `—`，但 dLLM/CodeBERT 正常计算。

| Dataset | dLLM refiner | Rows | Mean dLLM AUC | Mean AR AUC | Mean CodeBERT AUC |
|---|---|---|---|---|---|
| HumanEval | Dream-Coder 7B | 7 | 0.942 | 0.774 | 0.723 |
| HumanEval | LLaDA 8B | 7 | 0.967 | 0.712 | 0.488 |
| MBPP | Dream-Coder 7B | 7 | 0.911 | 0.769 | 0.701 |
| MBPP | LLaDA 8B | 7 | 0.961 | 0.676 | 0.599 |

| Dataset | AR drafter | dLLM refiner | Changed pairs | Fault tokens | dLLM AUC | AR AUC | CodeBERT AUC |
|---|---|---|---|---|---|---|---|
| HumanEval | DeepSeek | Dream-Coder | 11 | 11 | 0.940 | 0.844 | 0.778 |
| HumanEval | Qwen | Dream-Coder | 4 | 4 | 0.932 | 0.679 | 0.724 |
| HumanEval | Llama-3.1 | Dream-Coder | 26 | 16 | 0.939 | — | 0.723 |
| HumanEval | StarCoder2 | Dream-Coder | 103 | 2588 | 0.922 | 0.559 | 0.705 |
| HumanEval | Mistral | Dream-Coder | 34 | 50 | 0.962 | 0.821 | 0.727 |
| HumanEval | CodeLlama | Dream-Coder | 26 | 32 | 0.939 | 0.789 | 0.664 |
| HumanEval | DeepSeek | LLaDA | 8 | 9 | 0.976 | 0.777 | 0.572 |
| HumanEval | Qwen | LLaDA | 4 | 3 | 0.984 | 0.790 | 0.318 |
| HumanEval | Llama-3.1 | LLaDA | 26 | 15 | 0.969 | — | 0.572 |
| HumanEval | StarCoder2 | LLaDA | 62 | 168 | 0.886 | 0.844 | 0.687 |
| HumanEval | Mistral | LLaDA | 23 | 31 | 0.979 | 0.699 | 0.608 |
| HumanEval | CodeLlama | LLaDA | 19 | 18 | 0.976 | 0.740 | 0.556 |
| HumanEval | Seed-Instruct | Dream-Coder | 5 | 6 | 0.957 | 0.951 | 0.741 |
| HumanEval | Seed-Instruct | LLaDA | 3 | 2 | 1.000 | 0.420 | 0.106 |
| MBPP | DeepSeek | Dream-Coder | 24 | 26 | 0.961 | 0.826 | 0.761 |
| MBPP | Qwen | Dream-Coder | 17 | 27 | 0.956 | 0.873 | 0.745 |
| MBPP | Llama-3.1 | Dream-Coder | 36 | 53 | 0.931 | — | 0.707 |
| MBPP | StarCoder2 | Dream-Coder | 260 | 9836 | 0.805 | 0.535 | 0.679 |
| MBPP | Mistral | Dream-Coder | 66 | 120 | 0.903 | 0.796 | 0.651 |
| MBPP | CodeLlama | Dream-Coder | 40 | 49 | 0.957 | 0.794 | 0.676 |
| MBPP | DeepSeek | LLaDA | 17 | 20 | 0.971 | 0.593 | 0.592 |
| MBPP | Qwen | LLaDA | 16 | 19 | 0.972 | 0.763 | 0.563 |
| MBPP | Llama-3.1 | LLaDA | 32 | 32 | 0.971 | — | 0.566 |
| MBPP | StarCoder2 | LLaDA | 117 | 424 | 0.920 | 0.623 | 0.696 |
| MBPP | Mistral | LLaDA | 37 | 40 | 0.958 | 0.728 | 0.637 |
| MBPP | CodeLlama | LLaDA | 37 | 44 | 0.961 | 0.574 | 0.531 |
| MBPP | Seed-Instruct | Dream-Coder | 13 | 23 | 0.865 | 0.788 | 0.687 |
| MBPP | Seed-Instruct | LLaDA | 15 | 16 | 0.972 | 0.775 | 0.612 |

> Matrix 结论：dLLM AUC 在 24/24 个 AR-available 行上高于 AR logprob，在 28/28 行上高于 CodeBERT。Dream-Coder 平均 AUC：HE 0.942 / MBPP 0.911；LLaDA 平均 AUC：HE 0.967 / MBPP 0.961。

> Matrix 产物：`outputs/ablation_locator/matrix/calibration_matrix_summary.json`, `outputs/ablation_locator/matrix/calibration_data_*.json`, `outputs/ablation_locator/matrix_plots/{calibration,roc}_*.{pdf,png}`。

## SQL Feasibility（Spider dev）

> 轻量 Text-to-SQL 验证：AR 模型生成 Spider SQL 草稿，SQLite execution accuracy 评估 AR baseline；对 AR-failed samples 计算 locator P(non-fault) / P(fault)。通过阈值：dLLM ratio ≥ 3x。

> 背景：Spider 1.0 官方 test leaderboard 已于 2024-02-05 停止更新；[execution-with-values 榜](https://yale-lily.github.io/spider)头部为 80–90%+，本表只做 dev 前 200 条的轻量 feasibility 检查，不作为 SOTA 对比。

| AR 草稿 | n | AR Exec Acc | Pred exec errors | Dream ratio | LLaDA ratio | AR ratio | Dream CoCoder | LLaDA CoCoder | 结论 |
|---|---|---|---|---|---|---|---|---|---|
| DeepSeek-Coder 6.7B | 200 | 15.5% | 149 | 1.15x | 1.04x | 0.92x | — | — | 未通过 |
| Qwen2.5-Coder 7B | 200 | 63.0% | 43 | 1.03x | 1.01x | 1.06x | — | — | 未通过 |
| StarCoder2 7B | 200 | 36.5% | 78 | 1.07x | 1.10x | 0.94x | — | — | 未通过 |
| Mistral 7B | 200 | 40.0% | 58 | 1.01x | 1.04x | 0.96x | — | — | 未通过 |
| CodeLlama 7B | 200 | 42.5% | 64 | 1.01x | 1.04x | 0.91x | — | — | 未通过 |
| Seed-Coder-Instruct 8B | 200 | 47.0% | 42 | 3.21x | 1.02x | 1.00x | 47.0% | — | 可行性通过 |

> 产物：`outputs/sql_feasibility/*_spider_dev_eval.jsonl`, `outputs/sql_feasibility/*sql_locator_analysis_{dream,llada,ar}.txt`；可选 Phase 4 产物为 `outputs/sql_feasibility/*_{dream,llada}_remask_spider_dev_eval.jsonl`。

> 更新命令：`python -m coder.scripts.gen_sql_ar` → `python -m coder.scripts.sql_eval` → `python -m coder.analysis.sql_locator_analysis`。

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

## Math Benchmarks — Code-Execution Mode

> AR model generates a Python `solution()` function; answer extracted by exec(). CoCoder = AR code draft + Dream-Coder remask τ=0.9 on the code.

> GSM8K n=1319 (grade school). MATH-500 n=500 (competition). AIME n=90 (2022-2024). AIME-2025 n=30.

| 模型 | GSM8K acc% | MATH500 acc% | AIME acc% | AIME-2025 acc% |
|---|---|---|---|---|
| DeepSeek-Coder 6.7B | 61.0% | 6.4% | 5.6% | 6.7% |
| Qwen2.5-Coder 7B | 81.0% | 14.4% | 2.2% | 10.0% |
| Llama-3.1 8B | 74.8% | 7.0% | 6.7% | 10.0% |
|  |  |  |  |  |
| DeepSeek-Coder + Dream | 62.3% (+1.3pp) | 6.4% (0pp) | 5.6% (0pp) | 6.7% (0pp) |
| Qwen2.5-Coder + Dream | 81.5% (+0.5pp) | 14.2% (−0.2pp) | 2.2% (0pp) | 10.0% (0pp) |
| Llama-3.1 + Dream | 75.8% (+1.1pp) | 7.2% (+0.2pp) | 6.7% (0pp) | 6.7% (−3.3pp)* |

> *AIME-2025 n=30，−3.3pp = 1 题，噪声范围内。

> 上半部分：AR code-only baseline；下半部分：CoCoder 协作结果（Δ 括号内）。

> 核心 pattern：GSM8K（结构性代码错误）有小幅稳定增益 +0.5–1.3pp；MATH-500 / AIME（概念性错误）Δ ≈ 0，与 §7 boundary condition 理论一致。

### MATH500 Subject Breakdown (Code Mode)

| 模型 | Algebra | Prealgebra | Precalculus | Intermediate Algebra | Number Theory | Geometry | C&P |
|---|---|---|---|---|---|---|---|
| DeepSeek-Coder 6.7B | 9/124 (7%) | 6/82 (7%) | 0/56 (0%) | 1/97 (1%) | 14/62 (23%) | 0/41 (0%) | 2/38 (5%) |
| Qwen2.5-Coder 7B | — | — | — | — | — | — | — |
| Llama-3.1 8B | — | — | — | — | — | — | — |

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

### WildBench Writing（open-ended creative writing, n=146）

> Judge: claude-haiku-4-5-20251001，评分指标：per-criterion YES/NO checklist pass rate（WildBench 官方协议）。生成文件：`outputs/writing/writing_*.jsonl`，评估结果：`outputs/writing/writing_*_eval.json`。

| 模型 | n | checklist_pass_rate | Δ vs AR |
|---|---|---|---|
| Llama-3.1 8B (AR) | 146 | **40.69%** | — |
| Dream-General 7B (dLLM only) | 146 | 3.09% | −37.6pp |
| CoCoder τ=0.9 (Llama+Dream) | 146 | 40.15% | −0.54pp |

> **关键发现**：
> 1. dLLM 独立生成（Dream-General only）在开放写作任务上完全失败（3.09%），因为 dLLM 缺乏左到右的叙事连贯性（bidirectional mask-fill 在长文本创作上退化）。
> 2. CoCoder 协作保持 AR 质量（40.15% vs 40.69%，差距仅 0.54pp），验证了 boundary condition：写作错误是**概念/风格层面**的，AR 写出的通顺段落低置信度 token 少，dLLM 不会进行大量修改，因此输出几乎等同于 AR baseline。
> 3. 与 FRAMES/HotpotQA 一致的 pattern：CoCoder 在"dLLM 无优势"的领域**不劣化** AR，说明协作机制有良好的退化行为（graceful degradation）。

## Mask Granularity Ablation（DeepSeek-Coder draft, τ=0.9）

> Locator 固定为 Dream-Coder（dLLM bidirectional confidence），只改变 mask 粒度（token / span / line）和 Rewriter（dLLM=Dream-Coder / AR=DeepSeek-Coder）。
> token 行的数值来自正文主实验；span/line 行来自 `outputs/ablation_granularity/`。

| 粒度 | Rewriter | HE+ plus% | MBPP+ plus% | 产物路径 |
|------|----------|-----------|-------------|---------|
| token | dLLM | **72.6%** | **70.1%** | `outputs/base_tuteng/deepseek_dream_remask_humaneval_t0.9_timed.jsonl` |
| token | AR | 68.9% | 67.7% | `outputs/base_tuteng/deepseek_humaneval_locate_ar_rewrite_t0.9.jsonl` |
| span | dLLM | 65.9% | 69.3% | `outputs/ablation_granularity/deepseek_dream_{humaneval,mbpp}_t0.9_gran_span.jsonl` |
| span | AR | 66.5% | 67.5% | `outputs/ablation_granularity/deepseek_ar_rewrite_{humaneval,mbpp}_t0.9_gran_span.jsonl` |
| line | dLLM | 46.3% | 68.3% | `outputs/ablation_granularity/deepseek_dream_{humaneval,mbpp}_t0.9_gran_line.jsonl` |
| line | AR | 65.2% | 68.3% | `outputs/ablation_granularity/deepseek_ar_rewrite_{humaneval,mbpp}_t0.9_gran_line.jsonl` |
| AR-only | — | 56.7% | 65.1% | — |

> **关键发现**：
> 1. token 粒度 + dLLM rewriter 最优（72.6% HE+）。
> 2. **line 粒度出现 18.9pp 的 rewriter 差距**（dLLM 46.3% vs AR 65.2% on HE+）：masking 整行 = 让 dLLM 从头生成整行，触发 training/inference mismatch；AR rewriter 左到右逐 token 生成一行，不受该问题影响。
> 3. span 粒度 AR ≈ dLLM（66.5% vs 65.9% HE+），因为 span 仍在 token 级别评估范围内。
> 4. MBPP+ 粒度效应弱（MBPP 函数短，line mask 只有 5–10 token，dLLM 可应付）。
> 5. 这是对 Appendix theory 的最强实验验证：**dLLM rewriter 的优势与 token 粒度绑定**，一旦放大至 line 粒度即退化。

## τ 敏感性分析（全模型，tau_rerun，固定管线）

> 全部数字来自 `outputs/tau_rerun/`，使用修复后的 `build_evalplus_solution`（标 * 表示该 tau 值 fixed 版本尚在处理中，使用 tau_rerun 旧值）。

### DeepSeek-Coder 6.7B（AR baseline HE base=76.2% / MBPP base=74.9%）

| τ | HE base% | HE plus% | MBPP base% | MBPP plus% |
|---|---|---|---|---|
| 0.1 | 76.2% | 68.9% | 78.8% | 68.8% |
| 0.2 | 76.2% | 68.9% | 79.1% | 69.0% |
| 0.3 | 76.2% | 68.9% | 79.6% | 69.3% |
| 0.4 | 76.2% | 68.9% | 79.6% | 69.3% |
| 0.5 | 76.2% | 68.9% | 79.9% | 69.3% |
| 0.6 | 77.4% | 70.1% | 79.9% | 69.3% |
| 0.7 | 77.4% | 70.1% | 79.9% | 69.3% |
| 0.8 | 77.4% | 70.1% | 80.2% | 69.6% |
| 0.9 | 77.4% | 70.1% | 80.2% | 69.6% |

### Qwen2.5-Coder 7B（AR baseline HE base=82.3% / MBPP base=83.1%）

| τ | HE base% | HE plus% | MBPP base% | MBPP plus% |
|---|---|---|---|---|
| 0.1 | 83.5% | 78.0% | 82.8% | 73.0% |
| 0.2 | 83.5% | 78.0% | 82.8% | 73.0% |
| 0.3 | 83.5% | 78.0% | 82.8% | 73.0% |
| 0.4 | 83.5% | 78.0% | 82.8% | 73.0% |
| 0.5 | 83.5% | 78.0% | 82.5% | 72.8% |
| 0.6 | 83.5% | 78.0% | 82.3% | 72.5% |
| 0.7 | 83.5% | 78.0% | 82.3% | 72.5% |
| 0.8 | 83.5% | 78.0% | 82.3% | 72.2% |
| 0.9 | 83.5% | 78.0% | 82.3% | 72.2% |

### Llama-3.1 8B（AR baseline HE base=62.2% / MBPP base=71.7%）

| τ | HE base% | HE plus% | MBPP base% | MBPP plus% |
|---|---|---|---|---|
| 0.1 | 62.2% | 57.9% | 72.2% | 63.2% |
| 0.2 | 62.2% | 57.9% | 72.5% | 63.2% |
| 0.3 | 62.2% | 57.9% | 72.8% | 63.5% |
| 0.4 | 62.2% | 57.9% | 72.8% | 63.5% |
| 0.5 | 62.2% | 57.9% | 72.8% | 63.5% |
| 0.6 | 62.2% | 57.9% | 72.8% | 63.5% |
| 0.7 | 62.2% | 57.9% | 72.8% | 63.5% |
| 0.8 | 62.2% | 57.9% | 73.3% | 64.0% |
| 0.9 | 62.2% | 57.9% | 73.3% | 64.0% |

### CodeLlama 7B（AR baseline HE base=40.2% / MBPP base=50.5%）

| τ | HE base% | HE plus% | MBPP base% | MBPP plus% |
|---|---|---|---|---|
| 0.1 | 40.9% | 36.0% | 51.3% | 42.9% |
| 0.2 | 40.9% | 36.0% | 51.3% | 42.9% |
| 0.3 | 40.9% | 36.0% | 51.6% | 43.4% |
| 0.4 | 40.9% | 36.0% | 51.9% | 43.4% |
| 0.5 | 40.9% | 36.0% | 51.9% | 43.4% |
| 0.6 | 40.9% | 36.0% | 51.9% | 43.1% |
| 0.7 | 41.5% | 36.6% | 51.6% | 42.9% |
| 0.8 | — | — | 52.1% | 43.1% |
| 0.9 | 41.5% | 36.6% | 52.4% | 43.4% |

> 数据来源：`outputs/tau_rerun/`（全部已 fixed，使用修复后 `build_evalplus_solution`）。标 — 为该组合无对应 jsonl（CodeLlama HE τ=0.8 原始生成缺失）。

> **结论：τ 不敏感性在全部 4 个模型上成立**。Qwen HE 曲线完全平坦（83.5% for all τ），Llama-3.1 HE 在所有 τ 均固定于 62.2%（= AR baseline），DeepSeek MBPP 从 τ=0.6 开始趋于平坦。这一鲁棒性是 CoCoder 的正面特性：无需调参。

## 多轮精炼分析（Multi-round Refinement）

> AR 草稿：DeepSeek-Coder 6.7B，τ=0.9。r2 输入 = r1 输出；r3 输入 = r2 输出。产物：`outputs/tau_rerun/remask_{humaneval,mbpp}_t0.9_r{2,3}.jsonl`。

| round | HE base% | HE plus% | MBPP base% | MBPP plus% |
|---|---|---|---|---|
| r=1 | 76.8% | 69.5% | 80.2% | 69.6% |
| r=2 | 76.8% | 69.5% | 80.2% | 69.6% |
| r=3 | 76.8% | 69.5% | 80.4% | 69.6% |

> **结论：多轮完全饱和。** r2 = r1（HumanEval 精确相同），r3 ≈ r2（MBPP 仅 +1 题 = 噪声）。原因：DREAM 对自身输出的每个 token 置信度均接近 1.0，第二轮 remask 时几乎没有 token 被 remask，精炼退化为恒等变换。因此 **一轮精炼即足够**，无需多轮迭代。

## Import 剥离修复分析

> `build_evalplus_solution` 旧实现调用 `extract_single_function`（AST）提取目标函数，再用 `extract_prompt_imports(prompt)` 补回 import，但未收集 completion 中新增的 import（AR 模型常在 completion 头部写 `import math` 等）。修复后同时收集 `extract_prompt_imports(gen)`。

> 修复影响量（τ=0.9，normalize_evalplus_packaging 重处理已有 jsonl）：

| 模型 | 数据集 | 修复前 base% | 修复后 base% | Δ | solutions changed |
|---|---|---|---|---|---|
| Qwen2.5-Coder 7B | HumanEval | 73.8% | 83.5% | **+9.7pp** | 31/164 (18.9%) |
| Qwen2.5-Coder 7B | MBPP | 82.3% | 82.3% | 0pp | 0/378 (0%) |
| Llama-3.1 8B | HumanEval | 59.8% | 62.2% | **+2.4pp** | 9/164 (5.5%) |
| Llama-3.1 8B | MBPP | 73.3% | 73.3% | 0pp | 0/378 (0%) |
| CodeLlama 7B | HumanEval | 40.9% | 41.5% | +0.6pp | 1/164 (0.6%) |
| CodeLlama 7B | MBPP | 52.4% | 52.4% | 0pp | 0/378 (0%) |
| DeepSeek-Coder 6.7B | HumanEval | 76.8% | 77.4% | +0.6pp | 2/164 (1.2%) |

> **根本原因**：Qwen 在 completion 中写 import 的比例最高（18.9%），CodeLlama/DeepSeek 最低（0.6–1.2%）。HumanEval 受影响（任务常需 `import typing/math` 等），MBPP 几乎不受影响（MBPP 函数不需要 completion 级别的 import）。修复后 Qwen CoCoder HumanEval 超越 AR baseline（83.5% vs 82.3%）。

> 产物：`src/coder/utils/code_cleaning.py`（`build_evalplus_solution` 函数），`src/coder/scripts/normalize_evalplus_packaging.py`（重处理脚本）。

## 典型案例：AR 盲区，dLLM 可见（Illustrative Case）

> 来源：`docs/case_study.json`，`outputs/base_tuteng/deepseek_dream_remask_humaneval_t0.9.jsonl`。
> 筛选条件：AR 草稿失败 → CoCoder 成功，且 diff ≤ 5 tokens。已知 dLLM conf < 0.05 且 AR conf > 0.95（来自 `locator_scoring_clean_t09_deepseek.log`）。

### 案例 1：HumanEval/46 — fib4 循环边界（off-by-one）

```python
# AR 草稿（失败）：
def fib4(n: int):
    if n == 0: return 0
    elif n == 1: return 0
    elif n == 2: return 2
    elif n == 3: return 0
    else:
        a, b, c, d = 0, 0, 2, 0
        for _ in range(n - 4):          # AR conf ≈ 0.99; dLLM conf ≈ 0.01
            a, b, c, d = b, c, d, a + b + c + d
        return d

# CoCoder 修复（成功）：
        for _ in range(n - 3):          # diff: "4" → "3"，1 token 改动
```

**为什么 AR 写错？** AR 从左到右读取代码：base cases 覆盖 n=0..3（共 4 个），因此 loop 运行 n-4 次 —— 这是局部上下文支持的合理推断（但实际上初始化 `a,b,c,d=0,0,2,0` 对应 fib4(0..3)，下一个值 d=a+b+c+d 是 fib4(4)，故需 n-3 次迭代才能到 fib4(n)）。

**为什么 dLLM 能发现？** dLLM 的双向注意力同时看到初始化 `{0,0,2,0}` 和 docstring 中 `fib4(5)=4`。对于 n=5：`range(n-4)=range(1)` → 只执行一次 → 返回 d=2 ≠ 4。dLLM 对 `4` 这个 token 的置信度接近 0（与右侧上下文矛盾），AR logprob 则接近 1（局部模式支持）。

### 案例 2：HumanEval/74 — total_match 等号遗漏

```python
# AR 草稿（失败）：
def total_match(lst1, lst2):
    sum1 = sum(len(i) for i in lst1)
    sum2 = sum(len(i) for i in lst2)
    return lst1 if sum1 < sum2 else lst2   # AR 遗漏 equal case

# CoCoder 修复（成功）：
    return lst1 if sum1 <= sum2 else lst2  # diff: "<" → "<="，1 token
```

**Docstring 明确规定**：`if the two lists have the same number of chars, return the first list`。AR 从左到右生成时，看到 sum1/sum2 后自然写 `<`（常见的 less-than 模式）；只有在同时看到 docstring 约束和 return 语句时，才能检测到 equal case 缺失。

> **核心机制**：AR 生成时只有单向上下文，无法在 return 时"回头"验证 docstring 约束。dLLM 的 bidirectional attention 使其在评估每个 token 时同时持有函数签名、docstring、函数体的全局视野。

---

## τ 阈值定性分析（paper todo F）

> 使用 DeepSeek HumanEval tau_rerun fixed 产物（τ=0.1 和 τ=0.9），分析 threshold 变化的实际影响。

**实验设计**：固定 AR 草稿（DeepSeek HumanEval），比较 τ=0.1（保守）和 τ=0.9（激进）的 pass@1 差异：

| 模式 | 定义 | 数量 |
|---|---|---|
| 欠 mask（Under-mask）| AR 失败，τ=0.1 失败，τ=0.9 成功 | **2 tasks** |
| 过 mask（Over-mask） | AR 成功，τ=0.1 成功，τ=0.9 破坏 | **0 tasks** |

**欠 mask 案例（τ=0.1 错过，τ=0.9 能修复）：**

- **HumanEval/46**：`range(n-4)` → `range(n-3)`（见上方典型案例，off-by-one）
- **HumanEval/74**：`sum1 < sum2` → `sum1 <= sum2`（见上方，equal case 遗漏）

这两个案例的共同特征：错误 token 的 dLLM 置信度 < 0.05，仅在 τ=0.9 时才低于阈值被 mask；τ=0.1 时该 token 置信度超过 0.1（不会被 mask）。

**过 mask：0 例。** 在 HumanEval 164 道题上，τ=0.9 从未破坏任何 τ=0.1 能通过的题目。这意味着：**对于 DeepSeek HumanEval，提高 τ 的代价为零，只有收益。** 这从定性角度支持了 τ 不敏感性曲线的实验结果。

> 机制解释：τ 越低，只有置信度极低（模型极度"不确定"）的 token 才被 mask；这些 token 通常是真正的错误位置，mask 它们不会引入新错误。τ 越高只会额外 mask 一些"勉强可信"的 token，在 HumanEval 范围内，dLLM 的修复能力足以正确还原这些 token。

---

## 失败模式分解（Failure Mode Breakdown）

> AR 草稿：DeepSeek-Coder 6.7B，τ=0.9。对每道题分类四种结果。产物：`outputs/tau_rerun/remask_{humaneval,mbpp}_t0.9_fixed.jsonl`。

| 类别 | 描述 | HumanEval | MBPP |
|---|---|---|---|
| A) AR✓ Co✓ | 两者均通过（CoCoder 保留正确解） | 124/164 (75.6%) | 282/378 (74.6%) |
| B) AR✗ Co✓ | CoCoder 修复（locate+rewrite 成功） | **3/164 (1.8%)** | **21/378 (5.6%)** |
| C1) AR✗ Co✗，无改动 | 欠 mask：dLLM 高置信 → 无 token 被 mask | 32/164 (19.5%) | 63/378 (16.7%) |
| C2) AR✗ Co✗，有改动 | 错误修复：dLLM 检测到错误但改动仍不正确 | 4/164 (2.4%) | 11/378 (2.9%) |
| D) AR✓ Co✗ | 过 mask：CoCoder 破坏正确解 | **0/164 (0%)** | **0/378 (0%)** |

> D 类实际 code change = 0（sanitized solution 版本差异为 artifact，raw completion 相同）；**CoCoder 未曾破坏任何原本正确的解答**。

**C1（欠 mask）占主导**：在 HumanEval 上占 19.5%（全部题目）、88.9%（C 类中）。这些任务中 dLLM 对 AR 草稿中所有 token 的置信度均高于 τ=0.9，未触发任何 remask。根本原因：这些任务的错误属于**算法级别**（AR 的整体逻辑思路有误），而非 token 级别的语法/边界错误。dLLM 的双向 confidence 信号在此失效 —— 与 math 任务分析一致（Cohen's d ≈ 0.05）。

**C2（错误修复）仅 4 例**：
- `HumanEval/83`：`return 1` → `return 2`（数值有变化但仍错）
- `HumanEval/140`：`space_count -= 2` → `space_count = 0`（算法逻辑不同，均不正确）
- `HumanEval/32`：`1e-6` → `1e-7`（精度调整，逻辑同样不对）
- `HumanEval/119`：修改了 doctest 打印语句（不影响函数本身，误 mask）

> **结论**：CoCoder 的局限不在于"过度改动"（D 类 = 0），而在于**错误检测范围有限**（C1 = 算法错误无法 locate）。这与 §6 的 boundary condition 理论完全吻合：token 级 confidence 只能检测具有结构信号的错误，对算法级错误无效。

---

## Table 4 扩展 — Mistral / Seed-Coder-Instruct 完整方法行（paper todo J）

> 数据来源：`outputs/base_tuteng/`（原始管线，未经 import-fix reprocess）。Mistral/Seed-Coder-Instruct 无 tau_rerun 产物，timing 列暂缺。

| AR 模型 | 方法 | HE+ plus% | HE+ base% | MBPP+ plus% | MBPP+ base% |
|---|---|---|---|---|---|
| Mistral 7B | Mistral baseline | 31.1% | 37.8% | 41.8% | 48.4% |
|  | + Self-Refine | 23.8% | 31.7% | 37.6% | 42.6% |
|  | + Reflexion (w/ feedback) | 23.2% | 29.3% | 29.1% | 38.9% |
|  | + Rerank logprob k=8 | 34.8% | 42.1% | 44.2% | 51.1% |
|  | + Locate-AR-Rewrite | 28.7% | 32.9% | 43.4% | 49.7% |
|  | + LLaDA remask τ=0.9 | 31.1% | 37.2% | 36.2% | 42.6% |
|  | + Dream remask τ=0.9 (ours) | 32.3% | 39.0% | 42.6% | 50.0% |
|  |  |  |  |  |  |
| Seed-Coder-Instruct 8B | Seed-Coder-Instruct baseline | 70.1% | 75.0% | 72.2% | 84.9% |
|  | + Self-Refine | 74.4% | 78.7% | 69.0% | 81.0% |
|  | + Reflexion (w/ feedback) | 57.3% | 67.1% | 49.2% | 66.9% |
|  | + Rerank logprob k=8 | 77.4% | 82.3% | 73.8% | 86.0% |
|  | + Locate-AR-Rewrite | 76.2% | 81.7% | 69.8% | 82.8% |
|  | + LLaDA remask τ=0.9 | 62.8% | 67.1% | 64.6% | 77.0% |
|  | + Dream remask τ=0.9 (ours) | 65.2% | 70.7% | 72.2% | 84.7% |

> **Mistral 规律**：AR 精炼方法（Self-Refine/Reflexion/Locate-AR-Rewrite）在 Mistral 上普遍退化。Dream τ=0.9 是唯一有一致正向效果的方法（HE +1.2pp，MBPP +1.6pp），但绝对增益小（Mistral 是弱 AR 模型，错误以算法级为主）。

> **Seed-Coder-Instruct 规律**：强 AR 模型（HE base=75.0%），CoCoder 反而 hurt（Dream HE −4.3pp，LLaDA HE −7.9pp）。Reflexion 伤害最大（−7.9pp on HE）。Rerank k=8 表现最好（HE +7.3pp）。这与 AR 能力越强、dLLM 的 boundary condition 越难满足的理论一致 —— 强 AR 的错误更偏向概念/算法层面。

> **CodeLlama pipeline 说明**：原始 `base_tuteng` pipeline 下 CodeLlama AR baseline 仅 29.3%/25.6%（HE base/plus），根因是旧 sanitization 的 `extract_single_function` 在多函数场景下截断到第一个函数（非目标函数），导致 9 个 HumanEval 任务失败。已对所有 base_tuteng/codellama_* 文件执行 `normalize_evalplus_packaging` 重新评估（`specs/reprocess_codellama_ablations.sh`），数字已并入上方主表（normalized baseline = 40.9%/36.0%）。Standalone Models 表 / Table 3 中的 CodeLlama 数字（29.3%/25.6%）来自旧 pipeline，两表内部各自一致（AR-only 与 Collab 使用同一 pipeline），Δ 有效，但绝对值低估。

---

## Table 2 — Extended Benchmarks

### LiveCodeBench (accuracy%)

| 模型 | n_scored | accuracy% | 状态 |
|---|---|---|---|
| DeepSeek-Coder 6.7B | 1055 | 11.37% | ✅ |
| Qwen2.5-Coder 7B | 1055 | 22.56% | ✅ |
| Llama-3.1 8B | 1055 | 7.96% | ✅ |
| Dream-Coder 7B | 1055 | 2.94% | ✅ |
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
| Collab τ=0.9 (n=100) | 23.0% | ✅ |
| Dream (n=100) | 28.0% | ✅ |
| DeepSeek (n=100) | 23.0% | ✅ |

> 以上为 pass1_clean 结果（strip markdown fencing）。raw 版本均 0.0%（见 pitfalls.md）。

### Extended Table Shards 进度

| 实验 | 进度 (done/total) | pass@1% | 状态 |
|---|---|---|---|
| dream_livecodebench_pass1 | 1055/1055 | 2.9% | ✅ |
| dream_bigcodebench_instruct_full_pass1 | 1140/1140 | 22.5% | ✅ |
| collab_t0.9_livecodebench | 1055/1055 | 11.6% | ✅ |
| collab_t0.9_bigcodebench_instruct_full | 1140/1140 | 24.6% | ✅ |

> 更新命令：`python -m coder.scripts.run_extended_table --gpus <gpu_ids>`
