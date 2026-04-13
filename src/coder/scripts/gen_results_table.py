#!/usr/bin/env python3
"""
Generate docs/results.md — a human-readable table of all completed experiments.

Usage:
    python -m coder.scripts.gen_results_table [--out docs/results.md]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = REPO_ROOT / "outputs" / "base_tuteng"
REMASK_KODAI = REPO_ROOT / "outputs" / "remask_kodai"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path | str) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _resolve(fname: str | None, base: Path = OUTPUTS) -> Path | None:
    """Resolve a filename, handling remask_kodai/ prefix specially."""
    if fname is None:
        return None
    if "remask_kodai" in fname:
        return REPO_ROOT / "outputs" / fname
    return base / fname


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_evalplus_summary(path: Path | None) -> dict[str, Any] | None:
    data = _load_json(path) if path else None
    if data is None:
        return None
    s = data.get("summary", {})
    n_tasks = s.get("n_tasks")
    n_plus = s.get("n_plus_pass")
    n_base = s.get("n_base_pass")
    if n_tasks is None:
        return None
    return {
        "n_tasks": n_tasks,
        "plus_pct": round(100.0 * n_plus / n_tasks, 1) if n_plus is not None else None,
        "base_pct": round(100.0 * n_base / n_tasks, 1) if n_base is not None else None,
    }


def _load_timing(timing_path: Path | None, expected_n: int | None = None) -> float | None:
    """Load avg s/sample from a *.timing_summary.json file.

    Returns None if file missing, timing field absent, or n_records doesn't
    match expected_n (prevents returning garbage from partial runs).
    """
    data = _load_json(timing_path) if timing_path else None
    if data is None:
        return None
    t = data.get("timing", {})
    if not t or not t.get("total_s"):
        return None
    n = data.get("n_records_written")
    if expected_n is not None and n is not None and abs(n - expected_n) > 2:
        return None  # partial run, don't trust
    if n:
        return round(t["total_s"] / n, 1)
    return None


def _load_livecodebench(fname: str) -> dict[str, Any] | None:
    data = _load_json(OUTPUTS / fname)
    if data is None:
        return None
    n = data.get("n_scored", 0)
    acc = data.get("accuracy")
    return {"n_scored": n, "accuracy": acc, "ok": bool(acc is not None and n and n > 0)}


def _load_bigcodebench(fname: str) -> dict[str, Any] | None:
    data = _load_json(OUTPUTS / fname)
    if data is None:
        return None
    p1 = data.get("pass@1")
    if p1 is None:
        # try nested summary (sample100 format)
        p1 = (data.get("summary") or {}).get("pass@1")
    return {"pass1": p1, "calibrated": data.get("calibrated", "")}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _pct(val: float | None, suffix: str = "") -> str:
    if val is None:
        return "—"
    return f"{val:.1f}%{suffix}"


def _delta(ar: float | None, collab: float | None) -> str:
    if ar is None or collab is None:
        return "—"
    d = collab - ar
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.1f}pp"


def _sps(val: float | None) -> str:
    """Format seconds-per-sample."""
    if val is None:
        return "—"
    return f"{val:.1f}s"


def _fmt_row(*cells: Any) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _hr(n_cols: int) -> str:
    return "|" + "|".join(["---"] * n_cols) + "|"


# ---------------------------------------------------------------------------
# Section: Standalone Models
# ---------------------------------------------------------------------------

# (label, he_summary, mbpp_summary, lcb_summary, bcb_pass_at_k, timing_he, timing_mbpp)
# timing files: basename of *.timing_summary.json (None if not tracked)
_STANDALONE_ENTRIES = [
    (
        "DeepSeek-Coder 6.7B",
        "deepseek_humaneval_summary.json",
        "deepseek_mbpp_summary.json",
        "deepseek_livecodebench_pass1_clean_summary.json",
        "deepseek_bigcodebench_instruct_full_pass1_clean_pass_at_k.json",
        "deepseek_humaneval_timed.jsonl.timing_summary.json",
        "deepseek_mbpp_timed.jsonl.timing_summary.json",
    ),
    (
        "Qwen2.5-Coder 7B",
        "qwen_humaneval_summary.json",
        "qwen_mbpp_summary.json",
        "qwen_livecodebench_summary.json",
        "qwen_bigcodebench_instruct_full_pass_at_k.json",
        "qwen_humaneval_timed.jsonl.timing_summary.json",
        "qwen_mbpp_timed.jsonl.timing_summary.json",
    ),
    (
        "Llama-3.1 8B",
        "llama31_humaneval_summary.json",
        "llama31_mbpp_summary.json",
        "llama31_livecodebench_summary.json",
        "llama31_bigcodebench_instruct_full_pass_at_k.json",
        "llama31_humaneval_timed.jsonl.timing_summary.json",
        "llama31_mbpp_timed.jsonl.timing_summary.json",
    ),
    (
        "Mistral 7B",
        "mistral_humaneval_summary.json",
        "mistral_mbpp_summary.json",
        "mistral_livecodebench_summary.json",
        None,
        "mistral_humaneval_timed.jsonl.timing_summary.json",
        "mistral_mbpp_timed.jsonl.timing_summary.json",
    ),
    (
        "StarCoder2 7B",
        "starcoder2_humaneval_summary.json",
        "starcoder2_mbpp_summary.json",
        "starcoder2_livecodebench_summary.json",
        None,
        "starcoder2_humaneval_timed.jsonl.timing_summary.json",
        "starcoder2_mbpp_timed.jsonl.timing_summary.json",
    ),
    (
        "Dream-Coder 7B",
        "dream_humaneval_summary.json",
        "dream_mbpp_summary.json",
        "dream_livecodebench_summary.json",
        None,
        "dream_humaneval_timed.jsonl.timing_summary.json",
        "dream_mbpp_timed.jsonl.timing_summary.json",
    ),
    (
        "LLaDA 8B",
        "llada_humaneval_summary.json",
        "llada_mbpp_summary.json",
        "llada_livecodebench_summary.json",
        None,
        "llada_humaneval_timed.jsonl.timing_summary.json",
        "llada_mbpp_timed.jsonl.timing_summary.json",
    ),
    (
        "Seed-Coder 8B",
        "seed-coder_humaneval_summary.json",
        "seed-coder_mbpp_summary.json",
        "seed-coder_livecodebench_summary.json",
        None,
        "seed-coder_humaneval_timed.jsonl.timing_summary.json",
        "seed-coder_mbpp_timed.jsonl.timing_summary.json",
    ),
    (
        "Seed-DiffCoder 8B",
        "seed-diffcoder_humaneval_summary.json",
        "seed-diffcoder_mbpp_summary.json",
        "seed-diffcoder_livecodebench_summary.json",
        None,
        "seed-diffcoder_humaneval_timed.jsonl.timing_summary.json",
        "seed-diffcoder_mbpp_timed.jsonl.timing_summary.json",
    ),
    (
        "DiffuLLaMA 7B",
        "diffullama_humaneval_summary.json",
        "diffullama_mbpp_summary.json",
        None,
        None,
        "diffullama_humaneval_fix2.jsonl.timing_summary.json",
        "diffullama_mbpp_fix2.jsonl.timing_summary.json",
    ),
]


def section_standalone(out: list[str]) -> None:
    out.append("## Standalone Models\n")
    headers = [
        "模型",
        "HE+ plus%", "HE+ base%",
        "MBPP+ plus%", "MBPP+ base%",
        "s/sample (HE)", "s/sample (MBPP)",
    ]
    out.append(_fmt_row(*headers))
    out.append(_hr(len(headers)))

    for (label, he_f, mbpp_f, lcb_f, bcb_f, t_he_f, t_mb_f) in _STANDALONE_ENTRIES:
        he = _load_evalplus_summary(OUTPUTS / he_f if he_f else None)
        mb = _load_evalplus_summary(OUTPUTS / mbpp_f if mbpp_f else None)
        n_he = he["n_tasks"] if he else None
        n_mb = mb["n_tasks"] if mb else None

        t_he = _load_timing(OUTPUTS / t_he_f if t_he_f else None, n_he)
        t_mb = _load_timing(OUTPUTS / t_mb_f if t_mb_f else None, n_mb)

        out.append(_fmt_row(
            label,
            _pct(he["plus_pct"] if he else None),
            _pct(he["base_pct"] if he else None),
            _pct(mb["plus_pct"] if mb else None),
            _pct(mb["base_pct"] if mb else None),
            _sps(t_he), _sps(t_mb),
        ))
    out.append("")


# ---------------------------------------------------------------------------
# Section: Table 3 — Model Pairs
# ---------------------------------------------------------------------------

def section_table3_model_pairs(out: list[str]) -> None:
    pairs_path = OUTPUTS / "model_pairs_all_t0.9.json"
    if not pairs_path.exists():
        pairs_path = OUTPUTS / "model_pairs_humaneval_t0.9.json"
    data = _load_json(pairs_path)

    out.append("## Table 3 — Model Pairs（τ=0.9, pass@1 plus%）\n")
    headers = [
        "Dataset", "AR 草稿", "dLLM 精炼",
        "AR-only", "Collab", "Δ",
        "s/sample", "状态",
    ]
    out.append(_fmt_row(*headers))
    out.append(_hr(len(headers)))

    if data is None or "rows" not in data:
        out.append(_fmt_row("*（model_pairs_all_t0.9.json 不存在）*", *[""] * 7))
        out.append("")
        return

    # timing file map: slug → timing_summary path
    _PAIR_TIMING: dict[str, Path] = {
        "deepseek_llada_humaneval_t0.9":
            OUTPUTS / "deepseek_llada_remask_humaneval_t0.9.jsonl.timing_summary.json",
        "llama31_dream_humaneval_t0.9":
            OUTPUTS / "llama31_dream_remask_humaneval_t0.9.jsonl.timing_summary.json",
        "llama31_dream_mbpp_t0.9":
            OUTPUTS / "llama31_dream_remask_mbpp_t0.9.jsonl.timing_summary.json",
        "qwen_dream_mbpp_t0.9":
            OUTPUTS / "qwen_dream_remask_mbpp_t0.9.jsonl.timing_summary.json",
        "starcoder2_dream_mbpp_t0.9":
            OUTPUTS / "starcoder2_dream_remask_mbpp_t0.9.jsonl.timing_summary.json",
    }
    _PAIR_EXPECTED_N = {
        "humaneval": 164,
        "mbpp": 378,
    }

    prev_dataset = None
    for row in data["rows"]:
        dataset = row.get("dataset", "?")
        ar = row.get("ar_drafter", "?")
        dllm = row.get("dllm_refiner", "?")
        slug = row.get("slug", "")
        ar_pct = row.get("ar_only_pass_at_1_pct")
        collab_pct = row.get("collab_pass_at_1_pct")
        cs = row.get("collab_status", {})
        is_complete = cs.get("is_fully_complete", False)
        is_gen = cs.get("is_generation_complete", False)

        if is_complete:
            status = "✅"
        elif is_gen:
            status = "🔄 eval待跑"
        else:
            n_done = cs.get("unique_task_ids", 0)
            n_exp = cs.get("expected_tasks", "?")
            status = f"🔄 {n_done}/{n_exp}"

        # timing
        t_path = _PAIR_TIMING.get(slug)
        exp_n = _PAIR_EXPECTED_N.get(dataset)
        t = _load_timing(t_path, exp_n)

        # separator between datasets
        if prev_dataset and dataset != prev_dataset:
            out.append(_fmt_row(*[""] * len(headers)))
        prev_dataset = dataset

        out.append(_fmt_row(
            dataset, ar, dllm,
            _pct(ar_pct), _pct(collab_pct), _delta(ar_pct, collab_pct),
            _sps(t), status,
        ))

    out.append("")
    out.append(f"> 产物：`{pairs_path.relative_to(REPO_ROOT)}`"
               f"  —  更新命令：`python -m coder.scripts.model_pairs_evalplus`\n")
    out.append("> s/sample = collab 生成阶段（remask + denoising）的平均每题耗时。"
               "AR 草稿生成耗时未单独计入（gen_evalplus 尚未统计 timing）。\n")


# ---------------------------------------------------------------------------
# Section: Table 4 — Baselines
# ---------------------------------------------------------------------------

# (label, he_summary, mbpp_summary, he_timing_file, mbpp_timing_file)
_BASELINE_ENTRIES = [
    (
        "DeepSeek baseline",
        "deepseek_humaneval_summary.json",
        "deepseek_mbpp_summary.json",
        None, None,  # gen_evalplus no timing
    ),
    (
        "+ Self-Refine",
        "deepseek_humaneval_selfrefine_r1_summary.json",
        "deepseek_mbpp_selfrefine_r1_summary.json",
        "deepseek_humaneval_selfrefine_r1.jsonl.timing_summary.json",
        "deepseek_mbpp_selfrefine_r1.jsonl.timing_summary.json",
    ),
    (
        "+ Reflexion (w/ feedback)",
        "deepseek_humaneval_reflexion_feedback_r1_summary.json",
        "deepseek_mbpp_reflexion_feedback_r1_summary.json",
        "deepseek_humaneval_reflexion_feedback_r1.jsonl.timing_summary.json",
        "deepseek_mbpp_reflexion_feedback_r1.jsonl.timing_summary.json",
    ),
    (
        "+ Rerank logprob k=8",
        "deepseek_humaneval_rerank_logprob_k8_summary.json",
        "deepseek_mbpp_rerank_logprob_k8_summary.json",
        "deepseek_humaneval_rerank_logprob_k8_timed.jsonl.timing_summary.json",
        "deepseek_mbpp_rerank_logprob_k8_timed.jsonl.timing_summary.json",
    ),
    (
        "+ Locate-AR-Rewrite",
        "deepseek_humaneval_locate_ar_rewrite_t0.9_summary.json",
        "deepseek_mbpp_locate_ar_rewrite_t0.9_summary.json",
        "deepseek_humaneval_locate_ar_rewrite_t0.9.jsonl.timing_summary.json",
        "deepseek_mbpp_locate_ar_rewrite_t0.9.jsonl.timing_summary.json",
    ),
    (
        "+ LLaDA remask τ=0.9",
        "deepseek_llada_remask_humaneval_t0.9_summary.json",
        "deepseek_llada_remask_mbpp_t0.9_summary.json",
        "deepseek_llada_remask_humaneval_t0.9.jsonl.timing_summary.json",
        "deepseek_llada_remask_mbpp_t0.9.jsonl.timing_summary.json",
    ),
    (
        "+ Dream remask τ=0.9 (ours)",
        "remask_kodai/remask_humaneval_t0.9_summary.json",
        "remask_kodai/remask_mbpp_t0.9_summary.json",
        "deepseek_dream_remask_humaneval_t0.9_timed.jsonl.timing_summary.json",
        "deepseek_dream_remask_mbpp_t0.9_timed.jsonl.timing_summary.json",
    ),
]


def section_table4_baselines(out: list[str]) -> None:
    out.append("## Table 4 — DeepSeek-Coder Baselines（pass@1 plus%）\n")
    headers = [
        "方法",
        "HE+ plus%", "HE+ base%",
        "MBPP+ plus%", "MBPP+ base%",
        "s/sample (HE)", "s/sample (MBPP)",
    ]
    out.append(_fmt_row(*headers))
    out.append(_hr(len(headers)))

    for (label, he_f, mbpp_f, t_he_f, t_mb_f) in _BASELINE_ENTRIES:
        he = _load_evalplus_summary(_resolve(he_f))
        mb = _load_evalplus_summary(_resolve(mbpp_f))
        n_he = he["n_tasks"] if he else None
        n_mb = mb["n_tasks"] if mb else None

        t_he = _load_timing(OUTPUTS / t_he_f if t_he_f else None, n_he)
        t_mb = _load_timing(OUTPUTS / t_mb_f if t_mb_f else None, n_mb)

        out.append(_fmt_row(
            label,
            _pct(he["plus_pct"] if he else None),
            _pct(he["base_pct"] if he else None),
            _pct(mb["plus_pct"] if mb else None),
            _pct(mb["base_pct"] if mb else None),
            _sps(t_he), _sps(t_mb),
        ))
    out.append("")
    out.append("> s/sample = 方法总耗时 / 题目数。baseline timing 来自 `_timed` 重跑产物。\n")


# ---------------------------------------------------------------------------
# Section: Table 4b — Qwen2.5-Coder 7B Baselines
# ---------------------------------------------------------------------------

# (label, he_summary, mbpp_summary, he_timing_file, mbpp_timing_file)
_QWEN_BASELINE_ENTRIES = [
    (
        "Qwen baseline",
        "qwen_humaneval_summary.json",
        "qwen_mbpp_summary.json",
        "qwen_humaneval_timed.jsonl.timing_summary.json",
        "qwen_mbpp_timed.jsonl.timing_summary.json",
    ),
    (
        "+ Self-Refine",
        "qwen_humaneval_selfrefine_r1_summary.json",
        "qwen_mbpp_selfrefine_r1_summary.json",
        "qwen_humaneval_selfrefine_r1.jsonl.timing_summary.json",
        "qwen_mbpp_selfrefine_r1.jsonl.timing_summary.json",
    ),
    (
        "+ Reflexion (w/ feedback)",
        "qwen_humaneval_reflexion_feedback_r1_summary.json",
        "qwen_mbpp_reflexion_feedback_r1_summary.json",
        "qwen_humaneval_reflexion_feedback_r1.jsonl.timing_summary.json",
        "qwen_mbpp_reflexion_feedback_r1.jsonl.timing_summary.json",
    ),
    (
        "+ Rerank logprob k=8",
        "qwen_humaneval_rerank_logprob_k8_summary.json",
        "qwen_mbpp_rerank_logprob_k8_summary.json",
        "qwen_humaneval_rerank_logprob_k8.jsonl.timing_summary.json",
        "qwen_mbpp_rerank_logprob_k8.jsonl.timing_summary.json",
    ),
    (
        "+ Locate-AR-Rewrite",
        "qwen_humaneval_locate_ar_rewrite_t0.9_summary.json",
        "qwen_mbpp_locate_ar_rewrite_t0.9_summary.json",
        "qwen_humaneval_locate_ar_rewrite_t0.9.jsonl.timing_summary.json",
        "qwen_mbpp_locate_ar_rewrite_t0.9.jsonl.timing_summary.json",
    ),
    (
        "+ LLaDA remask τ=0.9",
        "qwen_llada_remask_humaneval_t0.9_summary.json",
        "qwen_llada_remask_mbpp_t0.9_summary.json",
        "qwen_llada_remask_humaneval_t0.9.jsonl.timing_summary.json",
        "qwen_llada_remask_mbpp_t0.9.jsonl.timing_summary.json",
    ),
    (
        "+ Dream remask τ=0.9 (ours)",
        "qwen_dream_remask_humaneval_t0.9_summary.json",
        "qwen_dream_remask_mbpp_t0.9_summary.json",
        "qwen_dream_remask_humaneval_t0.9.jsonl.timing_summary.json",
        "qwen_dream_remask_mbpp_t0.9.jsonl.timing_summary.json",
    ),
]


def section_table4_qwen_baselines(out: list[str]) -> None:
    out.append("## Table 4b — Qwen2.5-Coder 7B Baselines（pass@1 plus%）\n")
    headers = [
        "方法",
        "HE+ plus%", "HE+ base%",
        "MBPP+ plus%", "MBPP+ base%",
        "s/sample (HE)", "s/sample (MBPP)",
    ]
    out.append(_fmt_row(*headers))
    out.append(_hr(len(headers)))

    for (label, he_f, mbpp_f, t_he_f, t_mb_f) in _QWEN_BASELINE_ENTRIES:
        he = _load_evalplus_summary(OUTPUTS / he_f if he_f else None)
        mb = _load_evalplus_summary(OUTPUTS / mbpp_f if mbpp_f else None)
        n_he = he["n_tasks"] if he else None
        n_mb = mb["n_tasks"] if mb else None

        t_he = _load_timing(OUTPUTS / t_he_f if t_he_f else None, n_he)
        t_mb = _load_timing(OUTPUTS / t_mb_f if t_mb_f else None, n_mb)

        out.append(_fmt_row(
            label,
            _pct(he["plus_pct"] if he else None),
            _pct(he["base_pct"] if he else None),
            _pct(mb["plus_pct"] if mb else None),
            _pct(mb["base_pct"] if mb else None),
            _sps(t_he), _sps(t_mb),
        ))
    out.append("")
    out.append("> s/sample = 方法总耗时 / 题目数。\n")


# ---------------------------------------------------------------------------
# Section: Math Benchmarks (GSM8K + MATH500)
# ---------------------------------------------------------------------------

def _load_math_summary(path: Path | None) -> dict[str, Any] | None:
    data = _load_json(path) if path else None
    if data is None:
        return None
    acc = data.get("accuracy")
    n = data.get("n_problems")
    t_avg = (data.get("timing") or {}).get("generate_s_avg")
    return {
        "n": n,
        "acc_pct": round(acc * 100, 1) if acc is not None else None,
        "s_per_sample": round(t_avg, 1) if t_avg is not None else None,
        "subject_breakdown": data.get("subject_breakdown"),
    }


# (label, gsm8k_summary, math500_summary)  — 仅 AR 模型
_MATH_ENTRIES = [
    ("DeepSeek-Coder 6.7B", "deepseek_gsm8k_summary.json", "deepseek_math500_summary.json"),
    ("Qwen2.5-Coder 7B",    "qwen_gsm8k_summary.json",     "qwen_math500_summary.json"),
    ("Llama-3.1 8B",        "llama31_gsm8k_summary.json",  "llama31_math500_summary.json"),
    ("Mistral 7B",          None,                           None),  # 待跑
    ("StarCoder2 7B",       None,                           None),  # 待跑
]

_MATH500_SUBJECTS = [
    "Algebra", "Prealgebra", "Precalculus",
    "Intermediate Algebra", "Number Theory", "Geometry",
    "Counting & Probability",
]


def section_math(out: list[str]) -> None:
    out.append("## Math Benchmarks（AR 模型）\n")
    headers = ["模型", "GSM8K acc%", "s/sample", "MATH500 acc%", "s/sample"]
    out.append(_fmt_row(*headers))
    out.append(_hr(len(headers)))

    for label, gsm_f, math_f in _MATH_ENTRIES:
        gsm = _load_math_summary(OUTPUTS / gsm_f if gsm_f else None)
        math = _load_math_summary(OUTPUTS / math_f if math_f else None)
        out.append(_fmt_row(
            label,
            _pct(gsm["acc_pct"] if gsm else None),
            _sps(gsm["s_per_sample"] if gsm else None),
            _pct(math["acc_pct"] if math else None),
            _sps(math["s_per_sample"] if math else None),
        ))

    out.append("")
    out.append("> GSM8K：1319 道小学数学题（test set）。MATH500：500 道竞赛数学题（MATH 数据集子集）。\n")
    out.append("> 仅列 AR 模型；dLLM（Dream-Coder、LLaDA）不适用于此评测。\n")

    # MATH500 subject breakdown
    out.append("### MATH500 Subject Breakdown\n")
    subj_headers = ["模型"] + [s.replace("Counting & Probability", "C&P") for s in _MATH500_SUBJECTS]
    out.append(_fmt_row(*subj_headers))
    out.append(_hr(len(subj_headers)))

    for label, _, math_f in _MATH_ENTRIES:
        math = _load_math_summary(OUTPUTS / math_f if math_f else None)
        if math is None or not math.get("subject_breakdown"):
            out.append(_fmt_row(label, *["—"] * len(_MATH500_SUBJECTS)))
            continue
        bd = math["subject_breakdown"]
        cells = [label]
        for subj in _MATH500_SUBJECTS:
            s = bd.get(subj)
            cells.append(f"{s['correct']}/{s['total']} ({s['accuracy']*100:.0f}%)" if s else "—")
        out.append(_fmt_row(*cells))
    out.append("")


# ---------------------------------------------------------------------------
# Section: τ Sensitivity（DeepSeek + Dream-Coder, remask_kodai）
# ---------------------------------------------------------------------------

_TAU_VALUES = ["0.7", "0.8", "0.9", "0.93", "0.95", "0.97", "0.99"]


def section_tau_sweep(out: list[str]) -> None:
    out.append("## τ 敏感性分析（DeepSeek-Coder + Dream-Coder）\n")
    out.append("> AR baseline：HE+ plus=56.7%，MBPP+ plus=65.1%。\n")

    headers = ["τ", "HE+ plus%", "HE+ base%", "MBPP+ plus%", "MBPP+ base%"]
    out.append(_fmt_row(*headers))
    out.append(_hr(len(headers)))

    for tau in _TAU_VALUES:
        he_path  = REMASK_KODAI / f"remask_humaneval_t{tau}_summary.json"
        mb_path  = REMASK_KODAI / f"remask_mbpp_t{tau}_summary.json"
        he = _load_evalplus_summary(he_path)
        mb = _load_evalplus_summary(mb_path)
        out.append(_fmt_row(
            tau,
            _pct(he["plus_pct"] if he else None),
            _pct(he["base_pct"] if he else None),
            _pct(mb["plus_pct"] if mb else None),
            _pct(mb["base_pct"] if mb else None),
        ))
    out.append("")
    out.append("> 产物来自 `outputs/remask_kodai/`（DeepSeek 草稿 + Dream-Coder 精炼）。\n")
    out.append("> 其他 AR 模型 × τ 组合尚未系统扫描。\n")


# ---------------------------------------------------------------------------
# Section: Table 2 — Extended Benchmarks
# ---------------------------------------------------------------------------

def section_table2_extended(out: list[str]) -> None:
    out.append("## Table 2 — Extended Benchmarks\n")

    # LiveCodeBench
    out.append("### LiveCodeBench (accuracy%)\n")
    lcb_entries = [
        ("DeepSeek-Coder 6.7B",  "deepseek_livecodebench_pass1_clean_summary.json"),
        ("Qwen2.5-Coder 7B",     "qwen_livecodebench_summary.json"),
        ("Llama-3.1 8B",         "llama31_livecodebench_summary.json"),
        ("Dream-Coder 7B",       "dream_livecodebench_summary.json"),
        ("LLaDA 8B",             "llada_livecodebench_summary.json"),
        ("StarCoder2 7B",        "starcoder2_livecodebench_summary.json"),
        ("Collab τ=0.9 (n=100)", "sample100_collab_t0.9_livecodebench_seed3407_summary.json"),
        ("Dream (n=100)",        "sample100_dream_livecodebench_seed3407_summary.json"),
        ("DeepSeek (n=100)",     "sample100_livecodebench_seed3407_summary.json"),
    ]
    headers = ["模型", "n_scored", "accuracy%", "状态"]
    out.append(_fmt_row(*headers))
    out.append(_hr(len(headers)))
    for label, fname in lcb_entries:
        lcb = _load_livecodebench(fname)
        if lcb is None:
            out.append(_fmt_row(label, "—", "—", "❌ 无文件"))
        elif lcb["ok"]:
            out.append(_fmt_row(label, str(lcb["n_scored"]),
                                f"{lcb['accuracy']*100:.2f}%", "✅"))
        else:
            out.append(_fmt_row(label, lcb["n_scored"] or "—", "—", "❌ n_scored=0"))
    out.append("")

    # BigCodeBench
    out.append("### BigCodeBench（instruct, full, pass@1%）\n")
    bcb_entries = [
        ("DeepSeek-Coder 6.7B (pass1_clean)",
         "deepseek_bigcodebench_instruct_full_pass1_clean_pass_at_k.json"),
        ("DeepSeek-Coder 6.7B (raw)",
         "deepseek_bigcodebench_instruct_full_pass_at_k.json"),
        ("Qwen2.5-Coder 7B (raw)",
         "qwen_bigcodebench_instruct_full_pass_at_k.json"),
        ("Llama-3.1 8B (raw)",
         "llama31_bigcodebench_instruct_full_pass_at_k.json"),
        ("Collab τ=0.9 (n=100)",
         "sample100_collab_t0.9_bigcodebench_instruct_full_seed3407_summary.json"),
        ("Dream (n=100)",
         "sample100_dream_bigcodebench_instruct_full_seed3407_summary.json"),
        ("DeepSeek (n=100)",
         "sample100_bigcodebench_instruct_full_seed3407_summary.json"),
    ]
    headers = ["模型", "pass@1%", "状态"]
    out.append(_fmt_row(*headers))
    out.append(_hr(len(headers)))
    for label, fname in bcb_entries:
        bcb = _load_bigcodebench(fname)
        if bcb is None:
            out.append(_fmt_row(label, "—", "❌ 无文件"))
            continue
        p1 = bcb["pass1"]
        if p1 is not None:
            p1_str = f"{p1*100:.1f}%" if p1 < 1.0 else f"{p1:.1f}%"
            status = "✅" if p1 > 0 else "❌ 0.0%⚠️"
        else:
            p1_str, status = "—", "❌"
        out.append(_fmt_row(label, p1_str, status))
    out.append("")
    out.append("> ⚠️ raw 结果全部 0.0%，疑似评测时交互提示卡住（见 pitfalls.md）。pass1_clean 版本正常。\n")

    # Shards progress
    ext_path = OUTPUTS / "extended_table_t0.9_status.json"
    ext = _load_json(ext_path)
    if ext:
        out.append("### Extended Table Shards 进度\n")
        headers = ["实验", "进度 (done/total)", "pass@1%", "状态"]
        out.append(_fmt_row(*headers))
        out.append(_hr(len(headers)))
        for row in ext.get("rows", []):
            name = row["name"]
            total = sum(s["expected_rows"] for s in row["shards"])
            done = sum(s["actual_unique_rows"] for s in row["shards"])
            p1 = row.get("pass_at_1_pct")
            p1_str = f"{p1:.1f}%" if p1 is not None else "—"
            ok = "✅" if done == total else f"🔄 {done}/{total}"
            out.append(_fmt_row(name, f"{done}/{total}", p1_str, ok))
        out.append("")
        out.append("> 更新命令：`python -m coder.scripts.run_extended_table --gpus <gpu_ids>`\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate docs/results.md from experiment output files.")
    ap.add_argument("--out", default="docs/results.md")
    args = ap.parse_args()

    out_path = REPO_ROOT / args.out
    lines: list[str] = []

    lines.append("# 实验结果汇总\n")
    lines.append("> 自动生成，勿手动编辑。更新命令：`python -m coder.scripts.gen_results_table`\n")
    lines.append("")

    section_standalone(lines)
    section_table3_model_pairs(lines)
    section_table4_baselines(lines)
    section_table4_qwen_baselines(lines)
    section_math(lines)
    section_tau_sweep(lines)
    section_table2_extended(lines)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
