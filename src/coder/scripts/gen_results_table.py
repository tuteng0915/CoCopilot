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
ABLATION_LOCATOR = REPO_ROOT / "outputs" / "ablation_locator"
RESEARCH_OUT = REPO_ROOT / "outputs" / "research"
WRITING_OUT = REPO_ROOT / "outputs" / "writing"
MATH_CODE_OUT = REPO_ROOT / "outputs" / "math_code"


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


def _is_pass(status: Any) -> bool:
    return isinstance(status, str) and status.lower() == "pass"


def _dedup_evalplus_counts(summary_data: dict[str, Any]) -> dict[str, int | None] | None:
    """Recompute pass@1 counts when an EvalPlus result has duplicate task rows."""
    source_eval_file = summary_data.get("source_eval_file")
    if not source_eval_file:
        return None

    eval_data = _load_json(Path(source_eval_file))
    eval_map = (eval_data or {}).get("eval") or {}
    if not eval_map:
        return None

    n_tasks = 0
    n_samples_total = 0
    n_base_pass = 0
    n_plus_pass = 0
    plus_seen = False

    for rows in eval_map.values():
        n_tasks += 1
        rows = rows or []
        n_samples_total += len(rows)
        if not rows:
            continue

        # These tables are pass@1. Duplicate rows are merge artifacts, so use
        # the first evaluated sample for each task instead of counting all rows.
        first = rows[0]
        n_base_pass += int(_is_pass(first.get("base_status")))
        if first.get("plus_status") is not None:
            plus_seen = True
            n_plus_pass += int(_is_pass(first.get("plus_status")))

    return {
        "n_tasks": n_tasks,
        "n_samples_total": n_samples_total,
        "n_base_pass": n_base_pass,
        "n_plus_pass": n_plus_pass if plus_seen else None,
    }


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
    if s.get("n_samples_total") and s.get("n_samples_total") != n_tasks:
        dedup = _dedup_evalplus_counts(data)
        if dedup is not None:
            n_tasks = dedup["n_tasks"]
            n_plus = dedup["n_plus_pass"]
            n_base = dedup["n_base_pass"]
    return {
        "n_tasks": n_tasks,
        "plus_pct": round(100.0 * n_plus / n_tasks, 1) if n_plus is not None else None,
        "base_pct": round(100.0 * n_base / n_tasks, 1) if n_base is not None else None,
    }


def _load_timing(timing_path: Path | None, expected_n: int | None = None) -> float | None:
    """Load avg s/sample from a *.timing_summary.json file.

    Returns None if file missing, timing field absent, n_records doesn't
    match expected_n, or gen_remask reports zero generation time (resume-only
    timing files can have complete row counts but no real generation timing).
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
    if "remask_generate_s_total" in t and t["remask_generate_s_total"] <= 0:
        return None  # resume-only run, don't trust
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
    if p1 is None:
        # sample100 BigCodeBench summaries store the metric under pass_at_k.
        p1 = (data.get("pass_at_k") or {}).get("pass@1")
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
        "qwen_bigcodebench_instruct_full_pass1_clean_pass_at_k.json",
        "qwen_humaneval_timed.jsonl.timing_summary.json",
        "qwen_mbpp_timed.jsonl.timing_summary.json",
    ),
    (
        "Llama-3.1 8B",
        "llama31_humaneval_summary.json",
        "llama31_mbpp_summary.json",
        "llama31_livecodebench_summary.json",
        "llama31_bigcodebench_instruct_full_pass1_clean_pass_at_k.json",
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
        "CodeLlama 7B",
        "codellama_humaneval_summary.json",
        "codellama_mbpp_summary.json",
        None,
        None,
        "codellama_humaneval.jsonl.timing_summary.json",
        "codellama_mbpp.jsonl.timing_summary.json",
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
        "Seed-Coder-Instruct 8B",
        "packaging_v2/seed-coder-instruct_humaneval_pkgv2_summary.json",
        "seed-coder-instruct_mbpp_summary.json",
        None,
        None,
        "seed-coder-instruct_humaneval.jsonl.timing_summary.json",
        "seed-coder-instruct_mbpp.jsonl.timing_summary.json",
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
    out.append('> 注：当前所有实验 Locator 与 Rewriter 均为同一 dLLM（均等于"dLLM 精炼"列原始含义）。\n')
    headers = [
        "Dataset", "AR 草稿", "Locator", "Rewriter",
        "AR-only", "Collab", "Δ",
        "修对(+)", "弄坏(-)",
        "s/sample", "状态",
    ]
    out.append(_fmt_row(*headers))
    out.append(_hr(len(headers)))

    if data is None or "rows" not in data:
        out.append(_fmt_row("*（model_pairs_all_t0.9.json 不存在）*", *[""] * 8))
        out.append("")
        return

    # timing file map: slug → timing_summary path
    _PAIR_TIMING: dict[str, Path] = {
        # HumanEval
        "deepseek_dream_humaneval_t0.9":
            OUTPUTS / "deepseek_dream_remask_humaneval_t0.9_timed.jsonl.timing_summary.json",
        "qwen_dream_humaneval_t0.9":
            OUTPUTS / "qwen_dream_remask_humaneval_t0.9_timed.jsonl.timing_summary.json",
        "llama31_dream_humaneval_t0.9":
            OUTPUTS / "llama31_dream_remask_humaneval_t0.9.jsonl.timing_summary.json",
        "starcoder2_dream_humaneval_t0.9":
            OUTPUTS / "starcoder2_dream_remask_humaneval_t0.9_timed.jsonl.timing_summary.json",
        "deepseek_llada_humaneval_t0.9":
            OUTPUTS / "deepseek_llada_remask_humaneval_t0.9.jsonl.timing_summary.json",
        "qwen_llada_humaneval_t0.9":
            OUTPUTS / "qwen_llada_remask_humaneval_t0.9.jsonl.timing_summary.json",
        "llama31_llada_humaneval_t0.9":
            OUTPUTS / "llama31_llada_remask_humaneval_t0.9.jsonl.timing_summary.json",
        "starcoder2_llada_humaneval_t0.9":
            OUTPUTS / "starcoder2_llada_remask_humaneval_t0.9.jsonl.timing_summary.json",
        "mistral_dream_humaneval_t0.9":
            OUTPUTS / "mistral_dream_remask_humaneval_t0.9.jsonl.timing_summary.json",
        "mistral_llada_humaneval_t0.9":
            OUTPUTS / "mistral_llada_remask_humaneval_t0.9.jsonl.timing_summary.json",
        "codellama_dream_humaneval_t0.9":
            OUTPUTS / "codellama_dream_remask_humaneval_t0.9.jsonl.timing_summary.json",
        "codellama_llada_humaneval_t0.9":
            OUTPUTS / "codellama_llada_remask_humaneval_t0.9.jsonl.timing_summary.json",
        "seed_coder_instruct_dream_humaneval_t0.9":
            OUTPUTS / "seed-coder-instruct_dream_remask_humaneval_t0.9.jsonl.timing_summary.json",
        "seed_coder_instruct_llada_humaneval_t0.9":
            OUTPUTS / "seed-coder-instruct_llada_remask_humaneval_t0.9.jsonl.timing_summary.json",
        # MBPP
        "deepseek_dream_mbpp_t0.9":
            OUTPUTS / "deepseek_dream_remask_mbpp_t0.9_timed.jsonl.timing_summary.json",
        "qwen_dream_mbpp_t0.9":
            OUTPUTS / "qwen_dream_remask_mbpp_t0.9.jsonl.timing_summary.json",
        "llama31_dream_mbpp_t0.9":
            OUTPUTS / "llama31_dream_remask_mbpp_t0.9.jsonl.timing_summary.json",
        "starcoder2_dream_mbpp_t0.9":
            OUTPUTS / "starcoder2_dream_remask_mbpp_t0.9.jsonl.timing_summary.json",
        "deepseek_llada_mbpp_t0.9":
            OUTPUTS / "deepseek_llada_remask_mbpp_t0.9.jsonl.timing_summary.json",
        "qwen_llada_mbpp_t0.9":
            OUTPUTS / "qwen_llada_remask_mbpp_t0.9.jsonl.timing_summary.json",
        "llama31_llada_mbpp_t0.9":
            OUTPUTS / "llama31_llada_remask_mbpp_t0.9.jsonl.timing_summary.json",
        "starcoder2_llada_mbpp_t0.9":
            OUTPUTS / "starcoder2_llada_remask_mbpp_t0.9.jsonl.timing_summary.json",
        "mistral_dream_mbpp_t0.9":
            OUTPUTS / "mistral_dream_remask_mbpp_t0.9.jsonl.timing_summary.json",
        "mistral_llada_mbpp_t0.9":
            OUTPUTS / "mistral_llada_remask_mbpp_t0.9.jsonl.timing_summary.json",
        "codellama_dream_mbpp_t0.9":
            OUTPUTS / "codellama_dream_remask_mbpp_t0.9.jsonl.timing_summary.json",
        "codellama_llada_mbpp_t0.9":
            OUTPUTS / "codellama_llada_remask_mbpp_t0.9.jsonl.timing_summary.json",
        "seed_coder_instruct_dream_mbpp_t0.9":
            OUTPUTS / "seed-coder-instruct_dream_remask_mbpp_t0.9.jsonl.timing_summary.json",
        "seed_coder_instruct_llada_mbpp_t0.9":
            OUTPUTS / "seed-coder-instruct_llada_remask_mbpp_t0.9.jsonl.timing_summary.json",
    }
    _PAIR_EXPECTED_N = {
        "humaneval": 164,
        "mbpp": 378,
    }
    _PAIR_RESULT_OVERRIDES: dict[str, dict[str, Any]] = {}

    # Build AR timing lookup: label → {"humaneval": t_he, "mbpp": t_mbpp}
    _n_he, _n_mb = 164, 378
    _ar_timing: dict[str, dict[str, float | None]] = {}
    for (label, he_f, mbpp_f, _lcb_f, _bcb_f, t_he_f, t_mb_f) in _STANDALONE_ENTRIES:
        t_he = _load_timing(OUTPUTS / t_he_f if t_he_f else None, _n_he)
        t_mb = _load_timing(OUTPUTS / t_mb_f if t_mb_f else None, _n_mb)
        _ar_timing[label] = {"humaneval": t_he, "mbpp": t_mb}

    prev_dataset = None
    for row in data["rows"]:
        dataset = row.get("dataset", "?")
        ar = row.get("ar_drafter", "?")
        dllm = row.get("dllm_refiner", "?")
        slug = row.get("slug", "")
        override = _PAIR_RESULT_OVERRIDES.get(slug, {})
        ar_pct = override.get("ar_only_pass_at_1_pct", row.get("ar_only_pass_at_1_pct"))
        collab_pct = override.get("collab_pass_at_1_pct", row.get("collab_pass_at_1_pct"))
        wrong_to_right = override.get("wrong_to_right", row.get("wrong_to_right"))
        right_to_wrong = override.get("right_to_wrong", row.get("right_to_wrong"))
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

        # timing: AR draft + remask/denoising
        t_path = _PAIR_TIMING.get(slug)
        exp_n = _PAIR_EXPECTED_N.get(dataset)
        t_remask = _load_timing(t_path, exp_n)
        t_ar = _ar_timing.get(ar, {}).get(dataset)
        if t_remask is not None and t_ar is not None:
            t_total: float | None = round(t_remask + t_ar, 1)
        elif t_remask is not None:
            t_total = t_remask  # AR timing missing, show remask only
        else:
            t_total = None

        # separator between datasets
        if prev_dataset and dataset != prev_dataset:
            out.append(_fmt_row(*[""] * len(headers)))
        prev_dataset = dataset

        w2r = str(wrong_to_right) if wrong_to_right is not None else "—"
        r2w = str(right_to_wrong) if right_to_wrong is not None else "—"
        out.append(_fmt_row(
            dataset, ar, dllm, dllm,
            _pct(ar_pct), _pct(collab_pct), _delta(ar_pct, collab_pct),
            w2r, r2w,
            _sps(t_total), status,
        ))

    out.append("")
    out.append(f"> 产物：`{pairs_path.relative_to(REPO_ROOT)}`"
               f"  —  更新命令：`python -m coder.scripts.model_pairs_evalplus`\n")
    out.append("> s/sample = AR 草稿生成 + remask + dLLM denoising 的全流程平均每题耗时。\n")
    out.append("> Locator / Rewriter 拆分：当前实验均以同一 dLLM 同时充当两角色；后续 math 实验中可独立配置。\n")


# ---------------------------------------------------------------------------
# Section: Table 4 — Baselines
# ---------------------------------------------------------------------------

# (label, he_summary, mbpp_summary, he_timing_file, mbpp_timing_file)
_BASELINE_ENTRIES = [
    (
        "DeepSeek baseline",
        "deepseek_humaneval_summary.json",
        "deepseek_mbpp_summary.json",
        "deepseek_humaneval_timed.jsonl.timing_summary.json",
        "deepseek_mbpp_timed.jsonl.timing_summary.json",
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


def section_table4_all_baselines(out: list[str]) -> None:
    out.append("## Table 4 — AR Model Baselines（pass@1 plus%）\n")
    out.append("> 本表使用已有 EvalPlus sanitized 评测产物。\n")
    headers = [
        "AR 模型", "方法",
        "HE+ plus%", "HE+ base%",
        "MBPP+ plus%", "MBPP+ base%",
        "s/sample (HE)", "s/sample (MBPP)",
    ]
    out.append(_fmt_row(*headers))
    out.append(_hr(len(headers)))

    all_groups = [
        ("DeepSeek-Coder 6.7B", _BASELINE_ENTRIES),
        ("Qwen2.5-Coder 7B",    _QWEN_BASELINE_ENTRIES),
        ("Llama-3.1 8B",        _LLAMA31_BASELINE_ENTRIES),
        ("StarCoder2 7B",       _STARCODER2_BASELINE_ENTRIES),
    ]

    first_group = True
    for ar_label, entries in all_groups:
        if not first_group:
            out.append(_fmt_row(*[""] * len(headers)))
        first_group = False
        for i, (label, he_f, mbpp_f, t_he_f, t_mb_f) in enumerate(entries):
            he = _load_evalplus_summary(_resolve(he_f))
            mb = _load_evalplus_summary(_resolve(mbpp_f))
            n_he = he["n_tasks"] if he else None
            n_mb = mb["n_tasks"] if mb else None
            t_he = _load_timing(OUTPUTS / t_he_f if t_he_f else None, n_he)
            t_mb = _load_timing(OUTPUTS / t_mb_f if t_mb_f else None, n_mb)
            out.append(_fmt_row(
                ar_label if i == 0 else "",
                label,
                _pct(he["plus_pct"] if he else None),
                _pct(he["base_pct"] if he else None),
                _pct(mb["plus_pct"] if mb else None),
                _pct(mb["base_pct"] if mb else None),
                _sps(t_he), _sps(t_mb),
            ))
    out.append("")
    out.append("> s/sample = 方法总耗时 / 题目数。DeepSeek baseline timing 来自 `_timed` 重跑产物。\n")
    out.append("> 若 EvalPlus 结果中同一 task 出现重复样本，本表按 pass@1 口径只计每个 task 的第一条样本；这修正了 Llama-3.1/StarCoder2 Locate-AR-Rewrite HumanEval 的 merge duplicate artifact。\n")


# ---------------------------------------------------------------------------
# Section: Locator Ablation
# ---------------------------------------------------------------------------

_LOCATOR_ABLATION_ENTRIES = [
    (
        "dLLM locator (ours)",
        REMASK_KODAI / "remask_humaneval_t0.9_summary.json",
        REMASK_KODAI / "remask_mbpp_t0.9_summary.json",
        OUTPUTS / "deepseek_dream_remask_humaneval_t0.9_timed.jsonl.timing_summary.json",
        OUTPUTS / "deepseek_dream_remask_mbpp_t0.9_timed.jsonl.timing_summary.json",
    ),
    (
        "AR logprob locator",
        ABLATION_LOCATOR / "deepseek_dream_humaneval_t0.9_loc_ar_summary.json",
        ABLATION_LOCATOR / "deepseek_dream_mbpp_t0.9_loc_ar_summary.json",
        ABLATION_LOCATOR / "deepseek_dream_humaneval_t0.9_loc_ar.jsonl.timing_summary.json",
        ABLATION_LOCATOR / "deepseek_dream_mbpp_t0.9_loc_ar.jsonl.timing_summary.json",
    ),
    (
        "CodeBERT locator",
        ABLATION_LOCATOR / "deepseek_dream_humaneval_t0.9_loc_bert_summary.json",
        ABLATION_LOCATOR / "deepseek_dream_mbpp_t0.9_loc_bert_summary.json",
        ABLATION_LOCATOR / "deepseek_dream_humaneval_t0.9_loc_bert.jsonl.timing_summary.json",
        ABLATION_LOCATOR / "deepseek_dream_mbpp_t0.9_loc_bert.jsonl.timing_summary.json",
    ),
]


def section_locator_ablation(out: list[str]) -> None:
    out.append("## Locator Ablation（DeepSeek-Coder + Dream refine）\n")
    headers = [
        "Locator",
        "HE+ plus%", "HE+ base%",
        "MBPP+ plus%", "MBPP+ base%",
        "s/sample (HE)", "s/sample (MBPP)",
    ]
    out.append(_fmt_row(*headers))
    out.append(_hr(len(headers)))

    for label, he_path, mbpp_path, he_timing_path, mbpp_timing_path in _LOCATOR_ABLATION_ENTRIES:
        he = _load_evalplus_summary(he_path)
        mb = _load_evalplus_summary(mbpp_path)
        n_he = he["n_tasks"] if he else None
        n_mb = mb["n_tasks"] if mb else None
        t_he = _load_timing(he_timing_path, n_he)
        t_mb = _load_timing(mbpp_timing_path, n_mb)
        out.append(_fmt_row(
            label,
            _pct(he["plus_pct"] if he else None),
            _pct(he["base_pct"] if he else None),
            _pct(mb["plus_pct"] if mb else None),
            _pct(mb["base_pct"] if mb else None),
            _sps(t_he), _sps(t_mb),
        ))

    out.append("")
    out.append("> AR / CodeBERT locator rows use `confidence_threshold=0.9`; refine model remains Dream-Coder 7B.\n")


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
        "qwen_dream_remask_humaneval_t0.9_timed.jsonl.timing_summary.json",
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
# Section: Table 4c — Llama-3.1 8B Baselines
# ---------------------------------------------------------------------------

_LLAMA31_BASELINE_ENTRIES = [
    (
        "Llama-3.1 baseline",
        "llama31_humaneval_summary.json",
        "llama31_mbpp_summary.json",
        "llama31_humaneval_timed.jsonl.timing_summary.json",
        "llama31_mbpp_timed.jsonl.timing_summary.json",
    ),
    (
        "+ Self-Refine",
        "llama31_humaneval_selfrefine_r1_summary.json",
        "llama31_mbpp_selfrefine_r1_summary.json",
        "llama31_humaneval_selfrefine_r1.jsonl.timing_summary.json",
        "llama31_mbpp_selfrefine_r1.jsonl.timing_summary.json",
    ),
    (
        "+ Reflexion (w/ feedback)",
        "llama31_humaneval_reflexion_feedback_r1_summary.json",
        "llama31_mbpp_reflexion_feedback_r1_summary.json",
        "llama31_humaneval_reflexion_feedback_r1.jsonl.timing_summary.json",
        "llama31_mbpp_reflexion_feedback_r1.jsonl.timing_summary.json",
    ),
    (
        "+ Rerank logprob k=8",
        "llama31_humaneval_rerank_logprob_k8_summary.json",
        "llama31_mbpp_rerank_logprob_k8_summary.json",
        "llama31_humaneval_rerank_logprob_k8.jsonl.timing_summary.json",
        "llama31_mbpp_rerank_logprob_k8.jsonl.timing_summary.json",
    ),
    (
        "+ Locate-AR-Rewrite",
        "llama31_humaneval_locate_ar_rewrite_t0.9_summary.json",
        "llama31_mbpp_locate_ar_rewrite_t0.9_summary.json",
        "llama31_humaneval_locate_ar_rewrite_t0.9.jsonl.timing_summary.json",
        "llama31_mbpp_locate_ar_rewrite_t0.9.jsonl.timing_summary.json",
    ),
    (
        "+ LLaDA remask τ=0.9",
        "llama31_llada_remask_humaneval_t0.9_summary.json",
        "llama31_llada_remask_mbpp_t0.9_summary.json",
        "llama31_llada_remask_humaneval_t0.9.jsonl.timing_summary.json",
        "llama31_llada_remask_mbpp_t0.9.jsonl.timing_summary.json",
    ),
    (
        "+ Dream remask τ=0.9 (ours)",
        "llama31_dream_remask_humaneval_t0.9_summary.json",
        "llama31_dream_remask_mbpp_t0.9_summary.json",
        "llama31_dream_remask_humaneval_t0.9.jsonl.timing_summary.json",
        "llama31_dream_remask_mbpp_t0.9.jsonl.timing_summary.json",
    ),
]


def section_table4_llama31_baselines(out: list[str]) -> None:
    out.append("## Table 4c — Llama-3.1 8B Baselines（pass@1 plus%）\n")
    out.append("> 使用已有 EvalPlus sanitized 评测产物；Locate-AR-Rewrite HumanEval 行按每个 task 第一条样本去重后汇总。\n")
    headers = [
        "方法",
        "HE+ plus%", "HE+ base%",
        "MBPP+ plus%", "MBPP+ base%",
        "s/sample (HE)", "s/sample (MBPP)",
    ]
    out.append(_fmt_row(*headers))
    out.append(_hr(len(headers)))

    for (label, he_f, mbpp_f, t_he_f, t_mb_f) in _LLAMA31_BASELINE_ENTRIES:
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
# Section: Table 4d — StarCoder2 7B Baselines
# ---------------------------------------------------------------------------

_STARCODER2_BASELINE_ENTRIES = [
    (
        "StarCoder2 baseline",
        "starcoder2_humaneval_summary.json",
        "starcoder2_mbpp_summary.json",
        "starcoder2_humaneval_timed.jsonl.timing_summary.json",
        "starcoder2_mbpp_timed.jsonl.timing_summary.json",
    ),
    (
        "+ Self-Refine",
        "starcoder2_humaneval_selfrefine_r1_summary.json",
        "starcoder2_mbpp_selfrefine_r1_summary.json",
        "starcoder2_humaneval_selfrefine_r1.jsonl.timing_summary.json",
        "starcoder2_mbpp_selfrefine_r1.jsonl.timing_summary.json",
    ),
    (
        "+ Reflexion (w/ feedback)",
        "starcoder2_humaneval_reflexion_feedback_r1_summary.json",
        "starcoder2_mbpp_reflexion_feedback_r1_summary.json",
        "starcoder2_humaneval_reflexion_feedback_r1.jsonl.timing_summary.json",
        "starcoder2_mbpp_reflexion_feedback_r1.jsonl.timing_summary.json",
    ),
    (
        "+ Rerank logprob k=8",
        "starcoder2_humaneval_rerank_logprob_k8_summary.json",
        "starcoder2_mbpp_rerank_logprob_k8_summary.json",
        "starcoder2_humaneval_rerank_logprob_k8.jsonl.timing_summary.json",
        "starcoder2_mbpp_rerank_logprob_k8.jsonl.timing_summary.json",
    ),
    (
        "+ Locate-AR-Rewrite",
        "starcoder2_humaneval_locate_ar_rewrite_t0.9_summary.json",
        "starcoder2_mbpp_locate_ar_rewrite_t0.9_summary.json",
        "starcoder2_humaneval_locate_ar_rewrite_t0.9.jsonl.timing_summary.json",
        "starcoder2_mbpp_locate_ar_rewrite_t0.9.jsonl.timing_summary.json",
    ),
    (
        "+ LLaDA remask τ=0.9",
        "starcoder2_llada_remask_humaneval_t0.9_summary.json",
        "starcoder2_llada_remask_mbpp_t0.9_summary.json",
        "starcoder2_llada_remask_humaneval_t0.9.jsonl.timing_summary.json",
        "starcoder2_llada_remask_mbpp_t0.9.jsonl.timing_summary.json",
    ),
    (
        "+ Dream remask τ=0.9 (ours)",
        "starcoder2_dream_remask_humaneval_t0.9_summary.json",
        "starcoder2_dream_remask_mbpp_t0.9_summary.json",
        "starcoder2_dream_remask_humaneval_t0.9_timed.jsonl.timing_summary.json",
        "starcoder2_dream_remask_mbpp_t0.9.jsonl.timing_summary.json",
    ),
]


def section_table4_starcoder2_baselines(out: list[str]) -> None:
    out.append("## Table 4d — StarCoder2 7B Baselines（pass@1 plus%）\n")
    out.append("> 使用已有 EvalPlus sanitized 评测产物；Locate-AR-Rewrite HumanEval 行按每个 task 第一条样本去重后汇总。\n")
    headers = [
        "方法",
        "HE+ plus%", "HE+ base%",
        "MBPP+ plus%", "MBPP+ base%",
        "s/sample (HE)", "s/sample (MBPP)",
    ]
    out.append(_fmt_row(*headers))
    out.append(_hr(len(headers)))

    for (label, he_f, mbpp_f, t_he_f, t_mb_f) in _STARCODER2_BASELINE_ENTRIES:
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
# Section: Locator Fault-Detection Analysis
# ---------------------------------------------------------------------------

def section_locator_fault_detection(out: list[str]) -> None:
    """P(fault token) vs P(non-fault token) ratio — intrinsic locator quality."""
    data = _load_json(ABLATION_LOCATOR / "locator_fault_detection_summary.json")
    out.append("## Locator Fault-Detection Analysis\n")
    out.append(
        "> \"Surgical fault pairs\": 草稿失败 → remask 后通过，且改动 ≤10 字符的样本。"
        " 对每个 fault token 和 non-fault token 计算模型置信度，ratio = P(non-fault) / P(fault)，"
        "越高说明 locator 对错误位置的感知越敏锐。\n"
    )
    out.append("> AR 草稿：DeepSeek-Coder 6.7B，τ=0.9，dedupe_task=True。\n")

    headers = [
        "Locator",
        "HE P(fault)", "HE P(non-fault)", "HE ratio",
        "MBPP P(fault)", "MBPP P(non-fault)", "MBPP ratio",
    ]
    out.append(_fmt_row(*headers))
    out.append(_hr(len(headers)))

    if data is None:
        out.append(_fmt_row("*（locator_fault_detection_summary.json 不存在）*", *[""] * 6))
        out.append("")
        return

    for row in data.get("rows", []):
        he = row.get("humaneval", {})
        mb = row.get("mbpp", {})

        def _p(d: dict, key: str) -> str:
            v = d.get(key)
            return f"{v:.3f}" if v is not None else "—"

        def _ratio(d: dict) -> str:
            v = d.get("ratio")
            return f"**{v:.2f}x**" if v is not None else "—"

        out.append(_fmt_row(
            row.get("locator", "?"),
            _p(he, "p_fault_mean"), _p(he, "p_nonfault_mean"), _ratio(he),
            _p(mb, "p_fault_mean"), _p(mb, "p_nonfault_mean"), _ratio(mb),
        ))

    out.append("")
    out.append(
        f"> 产物：`outputs/ablation_locator/locator_fault_detection_summary.json`"
        f"  —  源 log：`outputs/ablation_locator/locator_scoring_clean_t09_deepseek.log`\n"
    )
    out.append(
        "> dLLM locator 对 fault token 置信度极低（HE P≈0.04，MBPP P≈0.008），"
        "与 non-fault token 差距悬殊；AR 和 MLM locator 几乎无区分（ratio≈1x）。\n"
    )


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

# (label, gsm8k_summary, math500_summary)  — AR + Dream-Coder remask collab
_MATH_COLLAB_ENTRIES = [
    ("DeepSeek-Coder + Dream", "deepseek_dream_remask_gsm8k_summary.json", "deepseek_dream_remask_math500_summary.json"),
    ("Qwen2.5-Coder + Dream",  "qwen_dream_remask_gsm8k_summary.json",     None),  # MATH500 run incomplete
    ("Llama-3.1 + Dream",      "llama31_dream_remask_gsm8k_summary.json",  "llama31_dream_remask_math500_summary.json"),
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

    # AR-only separator, then CoCoder collab rows
    out.append(_fmt_row(*[""] * 5))
    for label, gsm_f, math_f in _MATH_COLLAB_ENTRIES:
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
    out.append("> 上半部分：AR 模型独立推理 baseline；下半部分：CoCoder（AR草稿 + Dream-Coder remask τ=0.9）协作结果，整体不提升。\n")

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
# Section: Math Benchmarks — Code-Execution Mode（GSM8K / MATH-500 / AIME）
# ---------------------------------------------------------------------------

# (label, gsm8k_f, math500_f, aime_f, aime2025_f)
_MATH_CODE_ENTRIES = [
    ("DeepSeek-Coder 6.7B",
     "deepseek_gsm8k_code_eval.json",   "deepseek_math500_code_eval.json",
     "deepseek_aime_code_eval.json",     "deepseek_aime2025_code_eval.json"),
    ("Qwen2.5-Coder 7B",
     "qwen_gsm8k_code_eval.json",        "qwen_math500_code_eval.json",
     "qwen_aime_code_eval.json",         "qwen_aime2025_code_eval.json"),
    ("Llama-3.1 8B",
     "llama31_gsm8k_code_eval.json",     "llama31_math500_code_eval.json",
     "llama31_aime_code_eval.json",      "llama31_aime2025_code_eval.json"),
]

_MATH_CODE_COLLAB_ENTRIES = [
    ("DeepSeek-Coder + Dream",
     "deepseek_gsm8k_code_dream_t0.9_eval.json",   "deepseek_math500_code_dream_t0.9_eval.json",
     "deepseek_aime_code_dream_t0.9_eval.json",     "deepseek_aime2025_code_dream_t0.9_eval.json"),
    ("Qwen2.5-Coder + Dream",
     "qwen_gsm8k_code_dream_t0.9_eval.json",        "qwen_math500_code_dream_t0.9_eval.json",
     "qwen_aime_code_dream_t0.9_eval.json",         "qwen_aime2025_code_dream_t0.9_eval.json"),
    ("Llama-3.1 + Dream",
     "llama31_gsm8k_code_dream_t0.9_eval.json",     "llama31_math500_code_dream_t0.9_eval.json",
     "llama31_aime_code_dream_t0.9_eval.json",      "llama31_aime2025_code_dream_t0.9_eval.json"),
]


def _load_math_code_eval(path: Path | None) -> dict[str, Any] | None:
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


def section_math_code(out: list[str]) -> None:
    out.append("## Math Benchmarks — Code-Execution Mode\n")
    out.append("> AR model generates a Python `solution()` function; answer extracted by exec()."
               " CoCoder = AR code draft + Dream-Coder remask τ=0.9 on the code.\n")
    out.append("> GSM8K n=1319 (grade school). MATH-500 n=500 (competition). AIME n=90 (2022-2024). AIME-2025 n=30.\n")

    headers = ["模型", "GSM8K acc%", "MATH500 acc%", "AIME acc%", "AIME-2025 acc%"]
    out.append(_fmt_row(*headers))
    out.append(_hr(len(headers)))

    for label, gsm_f, math_f, aime_f, aime25_f in _MATH_CODE_ENTRIES:
        gsm   = _load_math_code_eval(MATH_CODE_OUT / gsm_f   if gsm_f   else None)
        math  = _load_math_code_eval(MATH_CODE_OUT / math_f  if math_f  else None)
        aime  = _load_math_code_eval(MATH_CODE_OUT / aime_f  if aime_f  else None)
        aime25 = _load_math_code_eval(MATH_CODE_OUT / aime25_f if aime25_f else None)
        out.append(_fmt_row(
            label,
            _pct(gsm["acc_pct"]    if gsm   else None),
            _pct(math["acc_pct"]   if math  else None),
            _pct(aime["acc_pct"]   if aime  else None),
            _pct(aime25["acc_pct"] if aime25 else None),
        ))

    out.append(_fmt_row(*[""] * 5))
    for label, gsm_f, math_f, aime_f, aime25_f in _MATH_CODE_COLLAB_ENTRIES:
        gsm   = _load_math_code_eval(MATH_CODE_OUT / gsm_f   if gsm_f   else None)
        math  = _load_math_code_eval(MATH_CODE_OUT / math_f  if math_f  else None)
        aime  = _load_math_code_eval(MATH_CODE_OUT / aime_f  if aime_f  else None)
        aime25 = _load_math_code_eval(MATH_CODE_OUT / aime25_f if aime25_f else None)
        out.append(_fmt_row(
            label,
            _pct(gsm["acc_pct"]    if gsm   else None),
            _pct(math["acc_pct"]   if math  else None),
            _pct(aime["acc_pct"]   if aime  else None),
            _pct(aime25["acc_pct"] if aime25 else None),
        ))

    out.append("")
    out.append("> 上半部分：AR code-only baseline；下半部分：CoCoder 协作结果。\n")
    out.append("> 核心 insight：math→code 使 CoCoder 的代码级 locator 可复用于数学推理。\n")

    out.append("### MATH500 Subject Breakdown (Code Mode)\n")
    subj_headers = ["模型"] + [s.replace("Counting & Probability", "C&P") for s in _MATH500_SUBJECTS]
    out.append(_fmt_row(*subj_headers))
    out.append(_hr(len(subj_headers)))
    for label, _gsm_f, math_f, _aime_f, _aime25_f in _MATH_CODE_ENTRIES:
        math = _load_math_code_eval(MATH_CODE_OUT / math_f if math_f else None)
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
# Section: General Domain Benchmarks（research QA + writing）
# ---------------------------------------------------------------------------

def _load_qa_eval(path: Path | None) -> dict[str, Any] | None:
    data = _load_json(path) if path else None
    if data is None:
        return None
    return {
        "n_total": data.get("n_total"),
        "em": data.get("exact_match"),
        "f1": data.get("token_f1"),
    }


def section_general_domain(out: list[str]) -> None:
    out.append("## General Domain Benchmarks（closed-book research QA）\n")
    out.append("> Dream-General = Dream-v0-Instruct-7B (text dLLM, not Dream-Coder)."
               " CoCoder = Llama-3.1 draft + Dream-General remask τ=0.9."
               " Closed-book: no retrieval.\n")

    # --- FRAMES ---
    out.append("### FRAMES（multi-hop research QA, n=824）\n")
    frames_entries = [
        ("Llama-3.1 8B (AR)",          RESEARCH_OUT / "frames_llama31_eval.json"),
        ("Dream-General 7B (dLLM)",     RESEARCH_OUT / "frames_dream_general_eval.json"),
        ("CoCoder τ=0.9 (Llama+Dream)", RESEARCH_OUT / "frames_cocoder_eval.json"),
    ]
    headers = ["模型", "n", "EM%", "Token F1%"]
    out.append(_fmt_row(*headers))
    out.append(_hr(len(headers)))
    for label, path in frames_entries:
        r = _load_qa_eval(path)
        if r is None:
            out.append(_fmt_row(label, "—", "—", "—"))
        else:
            out.append(_fmt_row(
                label,
                str(r["n_total"]) if r["n_total"] else "—",
                _pct(r["em"] * 100 if r["em"] is not None else None),
                _pct(r["f1"] * 100 if r["f1"] is not None else None),
            ))
    out.append("")

    # --- HotpotQA ---
    out.append("### HotpotQA（multi-hop QA distractor val, n=1000）\n")
    hotpot_entries = [
        ("Llama-3.1 8B (AR)",          RESEARCH_OUT / "hotpotqa_llama31_eval.json"),
        ("Dream-General 7B (dLLM)",     RESEARCH_OUT / "hotpotqa_dream_general_eval.json"),
        ("CoCoder τ=0.9 (Llama+Dream)", RESEARCH_OUT / "hotpotqa_llama31_dream_general_t0.9_eval.json"),
    ]
    headers = ["模型", "n", "EM%", "Token F1%"]
    out.append(_fmt_row(*headers))
    out.append(_hr(len(headers)))
    for label, path in hotpot_entries:
        r = _load_qa_eval(path)
        if r is None:
            out.append(_fmt_row(label, "—", "—", "—"))
        else:
            out.append(_fmt_row(
                label,
                str(r["n_total"]) if r["n_total"] else "—",
                _pct(r["em"] * 100 if r["em"] is not None else None),
                _pct(r["f1"] * 100 if r["f1"] is not None else None),
            ))
    out.append("")
    out.append("> WildBench Writing (n=146) 生成已完成（llama31 / dream_general），eval 需 LLM judge（API key），CoCoder run 仍在进行（37/146）。\n")


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
        ("Dream-Coder 7B",       "dream_livecodebench_pass1_sharded_summary.json"),
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
        ("DeepSeek-Coder 6.7B",
         "deepseek_bigcodebench_instruct_full_pass1_clean_pass_at_k.json"),
        ("Qwen2.5-Coder 7B",
         "qwen_bigcodebench_instruct_full_pass1_clean_pass_at_k.json"),
        ("Llama-3.1 8B",
         "llama31_bigcodebench_instruct_full_pass1_clean_pass_at_k.json"),
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
    out.append("> 以上为 pass1_clean 结果（strip markdown fencing）。raw 版本均 0.0%（见 pitfalls.md）。\n")

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
    section_table4_all_baselines(lines)
    section_locator_ablation(lines)
    section_locator_fault_detection(lines)
    section_math(lines)
    section_math_code(lines)
    section_general_domain(lines)
    section_tau_sweep(lines)
    section_table2_extended(lines)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
