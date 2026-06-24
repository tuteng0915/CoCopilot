#!/usr/bin/env python3
"""Print tau ablation results table from tau_rerun/ summaries."""
import json, os, sys

ROOT = os.path.join(os.path.dirname(__file__), "..", "outputs", "tau_rerun")

def read_summary(path):
    if not os.path.exists(path):
        return None, None
    d = json.load(open(path))
    s = d.get("summary", d)
    n = s.get("n_tasks", 0)
    if n == 0:
        return None, None
    return s.get("n_base_pass", 0) / n * 100, s.get("n_plus_pass", 0) / n * 100

def fmt(base, plus):
    if base is None:
        return "  --- /  ---"
    return f"{base:5.1f} / {plus:5.1f}"

print(f"\n{'tau':<6} {'HumanEval / HumanEval+':^24} {'MBPP / MBPP+':^24}")
print("-" * 60)

ar_he = read_summary(os.path.join(ROOT, "ar_humaneval_summary.json"))
ar_mb = read_summary(os.path.join(ROOT, "ar_mbpp_summary.json"))
print(f"{'AR':<6} {fmt(*ar_he):^24} {fmt(*ar_mb):^24}")

for t in ["0.5", "0.7", "0.8", "0.9", "0.93", "0.95", "0.97", "0.99"]:
    he = read_summary(os.path.join(ROOT, f"remask_humaneval_t{t}_summary.json"))
    mb = read_summary(os.path.join(ROOT, f"remask_mbpp_t{t}_summary.json"))
    print(f"{t:<6} {fmt(*he):^24} {fmt(*mb):^24}")

print()
