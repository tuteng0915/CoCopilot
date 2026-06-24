"""
Quick experiment: sweep τ and mask_ratio on MBPP/476 to see if truncation disappears.

The truncation only happens when tokens get masked. Since for correct AR output
Dream-Coder gives ~0.999 confidence, we also test with forced mask_ratio to
deliberately trigger the LLaDA truncation path.
"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import torch
from coder.models.dream_coder import DreamCoder
from coder.models.llada_coder import LLaDACoder
from coder.utils.schema import ModelRequest

PROMPT = (
    "Write a python function to find the sum of maximum and minimum element "
    "of the given array.\nYour code should pass these tests:\n"
    "assert big_sum([1,2,3]) == 4\n"
    "assert big_sum([5,6,7,8]) == 13\n"
    "assert big_sum([-1,-2,-3]) == -4\n"
)
DRAFT = "def big_sum(arr):\n    return max(arr) + min(arr)\n"
req = ModelRequest(prompt=PROMPT, max_new_tokens=128)

print("=" * 60)
print("Loading Dream-Coder on GPU 7...")
dream = DreamCoder(device="cuda")

print("Loading LLaDA on GPU 7...")
llada = LLaDACoder(device="cuda")

# ── Step 1: compare tokenizations ─────────────────────────────────────────────
dream_comp = dream.tok(DRAFT, return_tensors="pt", add_special_tokens=False)
llada_comp = llada.tok(DRAFT, return_tensors="pt", add_special_tokens=False)
dream_ids = dream_comp.input_ids[0].tolist()
llada_ids = llada_comp.input_ids[0].tolist()
dream_tokens = [dream.tok.decode([t]) for t in dream_ids]
llada_tokens = [llada.tok.decode([t]) for t in llada_ids]

print(f"\n── Tokenization comparison ───────────────────────────────────────")
print(f"  Dream-Coder: {len(dream_tokens)} tokens  {dream_tokens}")
print(f"  LLaDA:       {len(llada_tokens)} tokens  {llada_tokens}")
same_vocab = (dream.tok.vocab_size == llada.tok.vocab_size)
print(f"  Same vocab size: {same_vocab} ({dream.tok.vocab_size} vs {llada.tok.vocab_size})")
print(f"  Same ids: {dream_ids == llada_ids}")

# ── Step 2: Dream-Coder per-token confidence ──────────────────────────────────
messages = [{"role": "user", "content": PROMPT}]
enc = dream.tok.apply_chat_template(messages, return_tensors="pt",
                                     return_dict=True, add_generation_prompt=True)
prompt_ids_dream = enc.input_ids.to("cuda")
comp_ids_dream = dream_comp.input_ids.to("cuda")
confidence_dream = dream.score_tokens(prompt_ids_dream, comp_ids_dream)

print("\n── Dream-Coder per-token confidence ─────────────────────────────")
for i, (tok, conf) in enumerate(zip(dream_tokens, confidence_dream.tolist())):
    tag = " << MASK @τ=0.9" if conf < 0.9 else ""
    print(f"  [{i:2d}] {repr(tok):20s}  conf={conf:.4f}{tag}")

# ── Step 3: LLaDA per-token confidence ────────────────────────────────────────
messages2 = [{"role": "user", "content": PROMPT}]
prompt_text = llada.tok.apply_chat_template(messages2, add_generation_prompt=True, tokenize=False)
enc2 = llada.tok(prompt_text, add_special_tokens=False, padding=True, return_tensors="pt")
prompt_ids_llada = enc2["input_ids"].to("cuda")
attn_llada = enc2["attention_mask"].to("cuda")
comp_ids_llada = llada_comp.input_ids.to("cuda")
confidence_llada = llada.score_tokens(prompt_ids_llada, comp_ids_llada, attn_llada)

print("\n── LLaDA per-token confidence (in LLaDA token space) ────────────")
for i, (tok, conf) in enumerate(zip(llada_tokens, confidence_llada.tolist())):
    tag = " << MASK @τ=0.9" if conf < 0.9 else ""
    print(f"  [{i:2d}] {repr(tok):20s}  conf={conf:.4f}{tag}")

# ── Step 4: sweep with LLaDA's own scoring (correct token space) ──────────────
print("\n── LLaDA output sweep (LLaDA scores its own token space) ────────")
configs = [
    {"label": "τ=0.9",         "ct": 0.9,  "mr": None},
    {"label": "τ=0.7",         "ct": 0.7,  "mr": None},
    {"label": "τ=0.5",         "ct": 0.5,  "mr": None},
    {"label": "τ=0.3",         "ct": 0.3,  "mr": None},
    {"label": "mask_ratio=0.5", "ct": 0.9,  "mr": 0.50},
    {"label": "mask_ratio=0.3", "ct": 0.9,  "mr": 0.30},
    {"label": "mask_ratio=0.2", "ct": 0.9,  "mr": 0.20},
    {"label": "mask_ratio=0.1", "ct": 0.9,  "mr": 0.10},
]
for cfg in configs:
    # Use LLaDA's own confidence (correct token space), no external_confidence
    out = llada.generate_with_remask(
        req, DRAFT,
        confidence_threshold=cfg["ct"],
        mask_ratio=cfg["mr"],
    )
    ok = "max(arr) + min(arr)" in out and "return" in out
    status = "OK   " if ok else "TRUNC"
    print(f"  [{cfg['label']:18s}] [{status}]  {repr(out)}")
