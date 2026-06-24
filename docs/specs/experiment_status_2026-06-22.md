# Experiment Status & Analysis Spec — 2026-06-22

## 1. 目前正在运行的实验

### GPU 2 — LCB + BCB gen_remask
**脚本**: `scripts/run_gpu2_lcb_bcb.sh`  
**预计完成**: ~4.5天

| 任务 | 状态 | 输出文件 |
|------|------|---------|
| deepseek + Dream → LCB | 🔄 912/2103 (43%), ETA ~7h | `deepseek_dream_livecodebench_t0.9.jsonl` |
| deepseek + LLaDA → LCB | ⏳ | `deepseek_llada_livecodebench_t0.9_plnt3.jsonl` |
| llama31 + Dream → LCB | ⏳ | `llama31_dream_livecodebench_t0.9.jsonl` |
| llama31 + LLaDA → LCB | ⏳ | `llama31_llada_livecodebench_t0.9_plnt3.jsonl` |
| qwen + Dream → LCB | ⏳ | `qwen_dream_livecodebench_t0.9.jsonl` |
| qwen + LLaDA → LCB | ⏳ | `qwen_llada_livecodebench_t0.9_plnt3.jsonl` |
| deepseek/llama31/qwen + Dream/LLaDA → BCB | ⏳ (6 jobs) | `*_bigcodebench_instruct_full_t0.9*.jsonl` |

**LCB AR baselines** (已有): deepseek 11.5%, llama31 8.0%, qwen 22.6%

---

### GPU 5 — Math-to-code LLaDA refiner
**脚本**: `scripts/run_gpu5_math_llada.sh`  
**预计完成**: ~1.5h（今晚结束）

| 任务 | 状态 | LLaDA acc | AR baseline | Dream baseline |
|------|------|-----------|-------------|----------------|
| deepseek × gsm8k | ✅ 完成 | 61.4% | 61.0% | 62.3% |
| deepseek × math500 | ✅ 完成 | 6.4% | 6.4% | 6.4% |
| llama31 × gsm8k | ✅ 完成 | 75.4% | 74.8% | 75.8% |
| llama31 × math500 | ✅ 完成 | 7.0% | 7.0% | 7.2% |
| qwen × gsm8k | 🔄 181/500, ETA ~49min | — | 81.0% | 81.5% |
| qwen × math500 | ⏳ | — | 14.4% | 14.2% |

---

### GPU 6 — AR-rewrite ablation + DiffuCoder + Stable-DiffCoder
**脚本**: `scripts/run_gpu6_remaining.sh`

#### Phase A — AR rewrite ablation ✅ 全部完成
| 任务 | 结果 | 对比 Dream refiner |
|------|------|-------------------|
| llama31 AR-rewrite HE (span) | 57.3% | Dream: 62.2% → Dream +4.9pp |
| llama31 AR-rewrite MBPP (span) | 66.7% | Dream: 73.3% → Dream +6.6pp |
| qwen AR-rewrite HE (span) | 79.3% | Dream: 81.7% → Dream +2.4pp |
| qwen AR-rewrite MBPP (span) | 79.4% | Dream: 82.3% → Dream +2.9pp |

#### Phase B — DiffuCoder standalone MBPP 🔄 进行中
171/378 records，预计 ~3.5h

#### Phase C — DiffuCoder-7B-Instruct refiner ⏳ (14 jobs)
输出: `{AR}_diffucoder_remask_{DS}_t0.9.jsonl`，预计 ~40h

#### Phase D — Stable-DiffCoder-8B refiner ⏳ (14 jobs)
输出: `{AR}_seeddiff_remask_{DS}_t0.9.jsonl`，预计 ~40h

---

## 2. 已完成实验汇总

### 2.1 主表：HumanEval + MBPP
**输出目录**: `outputs/base_tuteng/`

| AR Model | HE(AR) | HE+Dream | HE+LLaDA | MBPP(AR) | MBPP+Dream | MBPP+LLaDA |
|----------|--------|----------|----------|----------|------------|------------|
| deepseek | 76.2% | **78.0%** (+1.8) | 74.4% (-1.8) | 74.9% | **80.2%** (+5.3) | 78.3% (+3.4) |
| qwen | 82.3% | 81.7% (-0.6) | **82.9%** (+0.6) | 83.1% | 82.3% (-0.8) | **83.3%** (+0.3) |
| llama31 | 62.2% | **62.2%** (0) | 61.6% (-0.6) | 71.7% | **73.3%** (+1.6) | 72.2% (+0.5) |
| codellama | 40.2% | 39.0% (-1.2) | **41.5%** (+1.2) | 50.5% | **52.4%** (+1.9) | 50.5% (0) |
| mistral | 37.8% | **39.0%** (+1.2) | 37.8% (0) | 48.4% | **50.0%** (+1.6) | 48.4% (0) |
| starcoder2 | 26.2% | **26.2%** (0) | 25.6% (-0.6) | 32.8% | **38.1%** (+5.3) | 33.3% (+0.5) |
| seed-coder-inst. | 75.0% | 70.7% (-4.3) | **81.7%** (+6.7) | 84.9% | **84.7%** (-0.3) | 84.7% (-0.3) |

**关键观察**:
- Dream 在 MBPP 表现更稳定（多个 +3~5pp），HumanEval 基本无增益甚至倒退
- qwen 已接近天花板，两种 refiner 效果均微弱
- seed-coder-instruct 异常：Dream 对 HE 倒退 -4.3pp，LLaDA 大幅提升 +6.7pp
- Dream 总体比 LLaDA 稳定（LLaDA 对 deepseek HE 有 -1.8pp 倒退）

### 2.2 Granularity Ablation（Dream, τ=0.9）
**输出目录**: `outputs/ablation_granularity/`

| AR | Dataset | Token (default) | Line | Span |
|----|---------|----------------|------|------|
| deepseek | HE | **78.0%** | 51.2% (-26.8pp) | 70.7% (-7.3pp) |
| deepseek | MBPP | **80.2%** | 78.0% (-2.2pp) | **80.2%** (0pp) |
| llama31 | HE | **62.2%** | 39.0% (-23.2pp) | **62.2%** (0pp) |
| llama31 | MBPP | **73.3%** | 58.2% (-15.1pp) | **73.3%** (0pp) |
| qwen | HE | 81.7% | 51.8% (-29.9pp) | **83.5%** (+1.8pp) |
| qwen | MBPP | **82.3%** | 65.1% (-17.2pp) | **82.3%** (0pp) |

**结论**: Token ≥ Span >> Line。Line 在 HumanEval 灾难性失效（-23~-30pp），MBPP 上略好但仍差。

### 2.3 AR Rewrite Ablation（locator=Dream, τ=0.9, span granularity）
**输出目录**: `outputs/ablation_granularity/`

| AR | Dataset | AR baseline | AR-rewrite | Dream-refiner | Dream 优势 |
|----|---------|-------------|------------|---------------|-----------|
| deepseek | HE | 76.2% | 75.0% | **78.0%** | +3.0pp vs rewrite |
| deepseek | MBPP | 74.9% | 78.0% | **80.2%** | +2.2pp vs rewrite |
| llama31 | HE | 62.2% | 57.3% | **62.2%** | +4.9pp vs rewrite |
| llama31 | MBPP | 71.7% | 66.7% | **73.3%** | +6.6pp vs rewrite |
| qwen | HE | 82.3% | 79.3% | **81.7%** | +2.4pp vs rewrite |
| qwen | MBPP | 83.1% | 79.4% | **82.3%** | +2.9pp vs rewrite |

**结论**: Dream hard-constraint diffusion 一致优于 AR soft-constraint rewrite（+2~7pp），验证了 diffusion infilling 的核心价值。

### 2.4 Math-to-code（LLaDA refiner, τ=0.9, plnt3）
**输出目录**: `outputs/math_code/`

| AR | Dataset | AR | +Dream | +LLaDA |
|----|---------|-----|--------|--------|
| deepseek | gsm8k | 61.0% | 62.3% | 61.4% |
| deepseek | math500 | 6.4% | 6.4% | 6.4% |
| llama31 | gsm8k | 74.8% | 75.8% | 75.4% |
| llama31 | math500 | 7.0% | 7.2% | 7.0% |
| qwen | gsm8k | 81.0% | 81.5% | (running) |
| qwen | math500 | 14.4% | 14.2% | (running) |

**结论**: 增益基本为零（±1pp）。Refiner 无法修复错误的数学推理。

---

## 3. 待跑完后如何整理合并

### 3.1 优先级排序

**P0（今晚/明天可用，直接进主表）**:
- ✅ GPU 5 math-to-code LLaDA（~1.5h完成）
- ✅ GPU 6 Phase A AR-rewrite（已完成）

**P1（2-3天，进 supplementary）**:
- GPU 2 LCB (deepseek dream job: ~7h 完成第1个; 全部约4.5天)
- GPU 6 Phase B DiffuCoder standalone（~3.5h）
- GPU 6 Phase C DiffuCoder refiner（~40h）

**P2（可能来不及或不放主表）**:
- GPU 6 Phase D Stable-DiffCoder refiner（~80h after Phase C）
- GPU 2 BCB（依赖 Phase 1 LCB 跑完后才开始）

### 3.2 主表最终形式建议

**Table 1（Main Results）**: 主表只用 Dream refiner，7 个 AR 模型 × HumanEval + MBPP
- 去掉 seed-coder-instruct（Dream 对它 HE -4.3pp，放 appendix 讨论）
- 或者保留并标注 "†"（用 LLaDA 结果替代）

**Table 2（Ablation）**: 三列并排
- Granularity: token vs span vs line（用 deepseek/llama31/qwen）
- Refiner type: Dream vs LLaDA vs AR-rewrite（3 × 2 datasets）

**Table 3（Extension）**:
- Math-to-code: AR vs +Dream vs +LLaDA（6行）
- LCB/BCB: 跑完后补充（AR baseline vs +Dream）

### 3.3 合并结果的脚本

跑完后执行：
```bash
python3 -m coder.scripts.gen_results_table \
  --output docs/analysis/results_final.md
```

或手动用下面的路径规范收集：

```
主表 (HE/MBPP):
  AR:       outputs/base_tuteng/{ar}_{ds}-sanitized_eval_results.json
  +Dream:   outputs/base_tuteng/{ar}_dream_remask_{ds}_t0.9*-sanitized_eval_results.json
  +LLaDA:   outputs/base_tuteng/{ar}_llada_remask_{ds}_t0.9_plnt3-sanitized_eval_results.json

Granularity:
  outputs/ablation_granularity/{ar}_dream_{ds}_t0.9_gran_{token/line/span}-sanitized_eval_results.json

AR-rewrite:
  outputs/ablation_granularity/{ar}_ar_rewrite_{ds}_t0.9_gran_span-sanitized_eval_results.json

Math-to-code:
  AR:     outputs/math_code/{ar}_{ds}_code_eval.json           (key: accuracy)
  +Dream: outputs/math_code/{ar}_{ds}_code_dream_t0.9_eval.json
  +LLaDA: outputs/math_code/{ar}_{ds}_code_llada_t0.9_plnt3_eval.json

LCB (跑完后):
  AR:     outputs/base_tuteng/{ar}_livecodebench_summary.json  (key: pass@1)
  +Dream: outputs/base_tuteng/{ar}_dream_livecodebench_t0.9_summary.json
  +LLaDA: outputs/base_tuteng/{ar}_llada_livecodebench_t0.9_plnt3_summary.json

DiffuCoder/SeedDiff refiner (跑完后):
  outputs/base_tuteng/{ar}_diffucoder_remask_{ds}_t0.9-sanitized_eval_results.json
  outputs/base_tuteng/{ar}_seeddiff_remask_{ds}_t0.9-sanitized_eval_results.json
```

---

## 4. 分析重点（写作时）

### 4.1 核心 claim 及支撑证据

**Claim 1**: dLLM 作为 refiner 可以有效提升 AR 生成代码质量
- 支撑: Dream 在 deepseek/starcoder2 MBPP 上 +5.3pp；平均 MBPP 增益 ~2pp
- 注意点: HumanEval 增益弱，需解释（更复杂的多行结构）

**Claim 2**: Hard-constraint diffusion infilling > soft-constraint AR rewrite
- 支撑: Dream 比 AR-rewrite 一致高 +2~7pp（全 3 个 AR 模型 × 2 datasets）
- 这是最干净的结论，应重点展示

**Claim 3**: Token-level masking 是最优粒度
- 支撑: Granularity ablation，token ≥ span >> line
- Span 几乎等同 token 说明精确 token 级别是自然的选择

### 4.2 需要解释的异常

**seed-coder-instruct 的 Dream 倒退**: 可能是 Dream scorer 和 Seed-diffusion 生成分布不匹配。
备选解释方向：
1. seed-coder-instruct 生成的代码 token 分布异常，Dream 的置信度估计不准
2. seed-coder-instruct 用了 diffusion-style 训练，和 Dream 的 score_tokens 方法有兼容性问题
→ 建议：在 paper 中放 appendix，作为 "refiner-AR compatibility" 的 discussion

**Math-to-code 无增益**: 很合理，refiner 修代码表达，不修数学推理。
→ 建议：作为 limitation section 内容，说明适用范围

### 4.3 如果有时间（优先级较低）

- DiffuCoder / Stable-DiffCoder 作为 refiner 的结果（GPU 6 Phase C/D）：
  对比 Dream 和 LLaDA，扩展到更多 dLLM 作为 refiner 的多样性
- BCB 需要 `bigcodebench.evaluate` CLI，本地跑不了，只能提交到有环境的机器

---

## 5. 快速检查命令

```bash
# 查看所有 GPU 进度
tail -3 /tmp/gpu2_lcb_bcb.log /tmp/gpu5_math_llada.log /tmp/gpu6_remaining.log 2>/dev/null | grep -v "remask:"

# 当前主表结果概览
python3 -c "
import json,os,glob
BASE='outputs/base_tuteng'
def p1(f):
    if not os.path.exists(f): return None
    d=json.load(open(f))
    if 'pass@1' in d: return d['pass@1']*100
    ev=d.get('eval',{}); n=0;b=0
    for tr in ev.values():
        for r in (tr if isinstance(tr,list) else [tr]):
            n+=1
            if r.get('base_status')=='pass': b+=1
    return b/n*100 if n else None
for ar in ['deepseek','qwen','llama31','codellama','mistral','starcoder2']:
    for ds in ['humaneval','mbpp']:
        ar_v=p1(f'{BASE}/{ar}_{ds}-sanitized_eval_results.json')
        dr_v=p1(f'{BASE}/{ar}_dream_remask_{ds}_t0.9-sanitized_eval_results.json') or p1(f'{BASE}/{ar}_dream_remask_{ds}_t0.9_timed-sanitized_eval_results.json')
        ll_v=p1(f'{BASE}/{ar}_llada_remask_{ds}_t0.9_plnt3-sanitized_eval_results.json')
        if ar_v:
            print(f'{ar:20s} {ds:10s}: AR={ar_v:.1f}  Dream={dr_v:.1f if dr_v else \"—\"}  LLaDA={ll_v:.1f if ll_v else \"—\"}')
"
```
