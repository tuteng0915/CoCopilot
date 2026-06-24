#!/usr/bin/env python3
"""Print a clean results table from tau_rerun outputs."""
from __future__ import annotations
import json, sys
from pathlib import Path

OUT = Path("outputs/tau_rerun")

def load(fname):
    p = OUT / fname
    if not p.exists():
        return None
    return json.loads(p.read_text())

def pct(n, d):
    return f"{n/d*100:.1f}" if d else "N/A"

def row(d, n_tasks=None):
    if d is None:
        return ("N/A", "N/A")
    s = d.get("summary", {})
    n = s.get("n_tasks", n_tasks or 1)
    return (pct(s.get("n_base_pass",0), n), pct(s.get("n_plus_pass",0), n))

# ── AR baselines ──────────────────────────────────────────────────────────────
print("=" * 75)
print("AR BASELINES")
print("=" * 75)
for model, he_f, mb_f in [
    ("DeepSeek-7B",  "ar_deepseek_humaneval_reeval_summary.json", "ar_deepseek_mbpp_reeval_summary.json"),
    ("CodeLlama-7B", "ar_codellama_humaneval_reeval_summary.json","ar_codellama_mbpp_reeval_summary.json"),
    ("Llama-3.1-8B", "ar_llama31_humaneval_reeval_summary.json",  "ar_llama31_mbpp_reeval_summary.json"),
    ("Qwen2.5-7B",   "ar_qwen_humaneval_reeval_summary.json",     "ar_qwen_mbpp_reeval_summary.json"),
]:
    hb, hp = row(load(he_f))
    mb, mp = row(load(mb_f))
    print(f"  {model:18s}  HE base={hb:>5}  plus={hp:>5}  |  MBPP base={mb:>5}  plus={mp:>5}")

# ── CoCoder (tau=0.9) OLD vs FIXED ───────────────────────────────────────────
print()
print("=" * 75)
print("CoCoder τ=0.9  (OLD → FIXED after build_evalplus import fix)")
print("=" * 75)

configs = [
    ("DeepSeek-7B",  "remask_humaneval_t0.9",           "remask_mbpp_t0.9"),
    ("CodeLlama-7B", "codellama_remask_humaneval_t0.9", "codellama_remask_mbpp_t0.9"),
    ("Llama-3.1-8B", "llama31_remask_humaneval_t0.9",   "llama31_remask_mbpp_t0.9"),
    ("Qwen2.5-7B",   "qwen_remask_humaneval_t0.9",      "qwen_remask_mbpp_t0.9"),
]

for model, he_stem, mb_stem in configs:
    hb_old, hp_old = row(load(he_stem + "_summary.json"))
    mb_old, mp_old = row(load(mb_stem + "_summary.json"))
    hb_fix, hp_fix = row(load(he_stem + "_fixed_summary.json"))
    mb_fix, mp_fix = row(load(mb_stem + "_fixed_summary.json"))

    def delta(old, new):
        if old == "N/A" or new == "N/A": return ""
        d = float(new) - float(old)
        return f"({d:+.1f})"

    d_he = delta(hb_old, hb_fix)
    d_mb = delta(mb_old, mb_fix)
    fix_tag = "[FIXED]" if hb_fix != "N/A" else "[pending]"
    print(f"  {model:18s}  HE: {hb_old}→{hb_fix}{d_he:>7}  MBPP: {mb_old}→{mb_fix}{d_mb:>7}  {fix_tag}")

# ── Tau curves ────────────────────────────────────────────────────────────────
print()
print("=" * 75)
print("TAU CURVES — base pass@1  (* = old, no asterisk = fixed)")
print("=" * 75)
taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
header = "model/τ         " + "  ".join(f"{t:.1f}" for t in taus)
print(header)

curve_configs = [
    ("Qwen HE",      "qwen_remask_humaneval_t",      "humaneval", 164),
    ("Qwen MBPP",    "qwen_remask_mbpp_t",            "mbpp",      378),
    ("Llama31 HE",   "llama31_remask_humaneval_t",    "humaneval", 164),
    ("Llama31 MBPP", "llama31_remask_mbpp_t",         "mbpp",      378),
    ("DeepSeek HE",  "remask_humaneval_t",            "humaneval", 164),
    ("DeepSeek MBPP","remask_mbpp_t",                 "mbpp",      378),
    ("CodeLlama HE", "codellama_remask_humaneval_t",  "humaneval", 164),
    ("CodeLlama MBPP","codellama_remask_mbpp_t",      "mbpp",      378),
]

for model, prefix, dataset, n in curve_configs:
    scores = []
    for t in taus:
        # prefer fixed
        d = load(f"{prefix}{t:.1f}_fixed_summary.json")
        if d:
            s = d["summary"]
            scores.append(f"{s['n_base_pass']/n*100:.1f}")
        else:
            d2 = load(f"{prefix}{t:.1f}_summary.json")
            if d2:
                s = d2["summary"]
                scores.append(f"{s['n_base_pass']/n*100:.1f}*")
            else:
                scores.append(" ---")
    print(f"  {model:16s}  " + "  ".join(f"{sc:>5}" for sc in scores))

# ── Multi-round ───────────────────────────────────────────────────────────────
print()
print("=" * 75)
print("MULTI-ROUND REFINEMENT (DeepSeek, HumanEval+MBPP)")
print("=" * 75)
for rnd, he_f, mb_f in [
    ("r1", "remask_humaneval_t0.9_summary.json",    "remask_mbpp_t0.9_summary.json"),
    ("r2", "remask_humaneval_t0.9_r2_summary.json", "remask_mbpp_t0.9_r2_summary.json"),
    ("r3", "remask_humaneval_t0.9_r3_summary.json", "remask_mbpp_t0.9_r3_summary.json"),
]:
    hb, hp = row(load(he_f))
    mb, mp = row(load(mb_f))
    print(f"  {rnd}  HE base={hb} plus={hp}  |  MBPP base={mb} plus={mp}")

# ── Summary: how many solutions changed ──────────────────────────────────────
print()
print("=" * 75)
print("IMPORT FIX IMPACT — solutions changed by normalize")
print("=" * 75)
for fname_pattern, label in [
    ("qwen_remask_humaneval_t0.9_fixed.jsonl",      "Qwen HE τ=0.9"),
    ("qwen_remask_mbpp_t0.9_fixed.jsonl",           "Qwen MBPP τ=0.9"),
    ("llama31_remask_humaneval_t0.9_fixed.jsonl",   "Llama31 HE τ=0.9"),
    ("llama31_remask_mbpp_t0.9_fixed.jsonl",        "Llama31 MBPP τ=0.9"),
    ("codellama_remask_humaneval_t0.9_fixed.jsonl", "CodeLlama HE τ=0.9"),
    ("codellama_remask_mbpp_t0.9_fixed.jsonl",      "CodeLlama MBPP τ=0.9"),
    ("remask_humaneval_t0.9_fixed.jsonl",           "DeepSeek HE τ=0.9"),
    ("remask_mbpp_t0.9_fixed.jsonl",                "DeepSeek MBPP τ=0.9"),
]:
    p = OUT / fname_pattern
    if not p.exists():
        print(f"  {label:30s}  pending")
        continue
    n_changed = 0
    n_total = 0
    with p.open() as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                n_total += 1
                if (d.get("gen") or {}).get("packaging_solution_changed"):
                    n_changed += 1
    print(f"  {label:30s}  changed: {n_changed}/{n_total} ({n_changed/n_total*100:.1f}%)")
