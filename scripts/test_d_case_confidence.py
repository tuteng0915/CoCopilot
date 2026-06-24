"""
Analyze confidence scores for actual D-case drafts from llama31+LLaDA MBPP.
Shows what gets masked at τ=0.9 and why truncation happens.
"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import json, torch
from coder.models.llada_coder import LLaDACoder
from coder.utils.schema import ModelRequest

BASE = "/home/wjzhang/tt_workspace/model/CoCoder/CoCoder/outputs/base_tuteng"
REFINER_PATH = f"{BASE}/llama31_llada_remask_mbpp_t0.9.jsonl"
AR_EVAL = f"{BASE}/llama31_mbpp-sanitized_eval_results.json"
CO_EVAL = f"{BASE}/llama31_llada_remask_mbpp_t0.9-sanitized_eval_results.json"

def load_eval(path):
    with open(path) as f:
        d = json.load(f)
    return d.get("eval", d)

def passed(rec):
    if isinstance(rec, list): rec = rec[0] if rec else {}
    return rec.get("plus_status") == "pass"

ar_eval = load_eval(AR_EVAL)
co_eval = load_eval(CO_EVAL)

co_data = {}
with open(REFINER_PATH) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        d = json.loads(line)
        co_data[d.get("task_id","")] = d

D_tasks = [tid for tid in ar_eval if passed(ar_eval.get(tid)) and not passed(co_eval.get(tid))]
print(f"D cases: {len(D_tasks)}")

# Pick a few short D cases
short_D = sorted(D_tasks, key=lambda t: len(co_data.get(t,{}).get("draft_completion","")))[:6]

print("\nLoading LLaDA...")
llada = LLaDACoder(device="cuda")

TAU = 0.9

for tid in short_D:
    rec = co_data[tid]
    draft = rec.get("draft_completion","")
    prompt = rec.get("prompt","")
    refined = rec.get("raw_completion","")

    # Score
    messages = [{"role": "user", "content": prompt}]
    prompt_text = llada.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    enc = llada.tok(prompt_text, add_special_tokens=False, padding=True, return_tensors="pt")
    prompt_ids = enc["input_ids"].to("cuda")
    attn = enc["attention_mask"].to("cuda")

    comp_enc = llada.tok(draft, return_tensors="pt", add_special_tokens=False)
    comp_ids = comp_enc.input_ids.to("cuda")
    conf = llada.score_tokens(prompt_ids, comp_ids, attn)

    tokens = [llada.tok.decode([t]) for t in comp_ids[0].tolist()]
    n_masked = (conf < TAU).sum().item()

    print(f"\n{'='*60}")
    print(f"Task: {tid}")
    print(f"Draft: {repr(draft)}")
    print(f"Refined (actual): {repr(refined)}")
    print(f"Tokens: {len(tokens)}, Masked @τ={TAU}: {n_masked}")
    for i, (tok, c) in enumerate(zip(tokens, conf.tolist())):
        flag = " << MASKED" if c < TAU else ""
        print(f"  [{i:2d}] {repr(tok):20s} conf={c:.5f}{flag}")
