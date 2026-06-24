"""
Quick validation: protect_last_n_tokens=3 should prevent trailing truncation.
Tests 6 actual D cases from llama31 + LLaDA MBPP (τ=0.9).
"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import json, torch
from coder.models.llada_coder import LLaDACoder
from coder.utils.schema import ModelRequest

BASE = "/home/wjzhang/tt_workspace/model/CoCoder/CoCoder/outputs/base_tuteng"

def load_eval(path):
    with open(path) as f: d = json.load(f)
    return d.get("eval", d)

def passed(rec):
    if isinstance(rec, list): rec = rec[0] if rec else {}
    return rec.get("plus_status") == "pass"

ar_eval = load_eval(f"{BASE}/llama31_mbpp-sanitized_eval_results.json")
co_eval = load_eval(f"{BASE}/llama31_llada_remask_mbpp_t0.9-sanitized_eval_results.json")

co_data = {}
with open(f"{BASE}/llama31_llada_remask_mbpp_t0.9.jsonl") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        d = json.loads(line)
        co_data[d.get("task_id","")] = d

D_tasks = [tid for tid in ar_eval if passed(ar_eval.get(tid)) and not passed(co_eval.get(tid))]
short_D = sorted(D_tasks, key=lambda t: len(co_data.get(t,{}).get("draft_completion","")))[:6]

print("Loading LLaDA...")
llada = LLaDACoder(device="cuda")

print(f"\nTesting {len(short_D)} D cases with protect_last_n_tokens=3 vs baseline\n")

for tid in short_D:
    rec = co_data[tid]
    draft = rec.get("draft_completion", "")
    prompt = rec.get("prompt", "")
    original_bad = rec.get("raw_completion", "")
    req = ModelRequest(prompt=prompt, max_new_tokens=256)

    # Without protection (should truncate)
    out_baseline = llada.generate_with_remask(
        req, draft,
        confidence_threshold=0.9,
        protect_last_n_tokens=0,
    )
    # With protection
    out_protected = llada.generate_with_remask(
        req, draft,
        confidence_threshold=0.9,
        protect_last_n_tokens=3,
    )

    base_ok = draft.rstrip().startswith(out_baseline.rstrip()) and len(out_baseline.rstrip()) < len(draft.rstrip())
    prot_ok = out_protected.rstrip() == draft.rstrip() or out_protected == draft

    print(f"{tid}")
    print(f"  Draft:     {repr(draft[:80])}")
    print(f"  Baseline:  {repr(out_baseline[:80])}  {'TRUNC' if base_ok else 'ok'}")
    print(f"  Protected: {repr(out_protected[:80])}  {'FIXED' if prot_ok else 'CHANGED'}")
    print()
