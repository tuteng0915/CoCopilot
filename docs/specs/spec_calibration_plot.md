# Spec: Calibration Plot — Paper Integration

> 完整背景见 `done/spec_locator_calibration_analysis.md`（数据收集 + 绘图已执行完，2026-06-09）。
> 本 spec 只做最后一步：将已有图片加入论文 `app:calibration` 附录。

---

## 现状

| 文件 | 状态 |
|------|------|
| `outputs/ablation_locator/plots/calibration_humaneval.png` | ✅ 已存在 |
| `outputs/ablation_locator/plots/calibration_mbpp.png` | ✅ 已存在 |
| `NeurIPS26-CoCoder/images/roc_humaneval.png` | ✅ 已在论文 |
| `NeurIPS26-CoCoder/images/calibration_humaneval.png` | ❌ 尚未复制 |

`app:calibration` 目前只有 ROC 曲线图 + AUC matrix 表，**缺少 calibration bar chart**。

---

## 步骤（无需 GPU）

### Step 1：复制图片到论文 images 目录

```bash
cd /home/wjzhang/tt_workspace/model/CoCoder
cp CoCoder/outputs/ablation_locator/plots/calibration_humaneval.png NeurIPS26-CoCoder/images/
cp CoCoder/outputs/ablation_locator/plots/calibration_mbpp.png      NeurIPS26-CoCoder/images/
```

验证：
```bash
ls NeurIPS26-CoCoder/images/calibration_*.png
# 期望输出：calibration_humaneval.png  calibration_mbpp.png
```

---

### Step 2：在 `appendix.tex` 的 `app:calibration` 段加入图

打开 `NeurIPS26-CoCoder/section/appendix.tex`，找到 `\label{app:calibration}` 之后、`\begin{figure}[h]`（ROC 图那个 figure）之前，插入以下内容：

```latex
Figure~\ref{fig:calibration} shows confidence calibration for the dLLM (Dream-Coder) locator.
Each bar corresponds to one decile of confidence scores; the y-axis shows the fraction of tokens
in that bin that are fault tokens (changed in corrected pairs).
The monotone decreasing trend confirms that low-confidence tokens are disproportionately fault tokens,
providing direct visual evidence of the dLLM's discriminative power beyond the summary AUC statistic.

\begin{figure}[h]
  \centering
  \begin{subfigure}[t]{0.48\linewidth}
    \includegraphics[width=\linewidth]{images/calibration_humaneval.png}
  \end{subfigure}
  \hfill
  \begin{subfigure}[t]{0.48\linewidth}
    \includegraphics[width=\linewidth]{images/calibration_mbpp.png}
  \end{subfigure}
  \caption{Calibration plot for the dLLM locator on HumanEval+ (left) and MBPP+ (right).
    Tokens are binned by dLLM confidence score (10 bins, 0--1).
    The y-axis shows the fraction of tokens in each bin that are fault tokens
    (differ between AR draft and CoCoder output in corrected pairs).
    Low-confidence bins contain a substantially higher proportion of fault tokens,
    confirming that dLLM confidence is well-calibrated as a fault signal.}
  \label{fig:calibration}
\end{figure}
```

---

### Step 3：（可选）更新 `06_analysis.tex` 中引用 calibration 的句子

当前 analysis 第 37 行只提到 ROC-AUC 数字，再引用 `app:calibration`。可以将最后半句改为：

```latex
the advantage holds across all 21 AR--dLLM pairs on both benchmarks, and calibration plots
(Figure~\ref{fig:calibration} in Appendix~\ref{app:calibration}) confirm the monotone
relationship between dLLM confidence and fault rate at the token level.
```

---

## 验收

- [ ] `images/calibration_humaneval.png` 和 `calibration_mbpp.png` 存在
- [ ] appendix `app:calibration` 段中有 `fig:calibration` 的 figure 环境
- [ ] 两张图在 PDF 编译后正常显示（单调递减的 bar chart）
- [ ] 标签 `\label{fig:calibration}` 和 `\ref{fig:calibration}` 匹配

---

## 时间估计

15–20 分钟（纯 paper 编辑，无需实验）。
