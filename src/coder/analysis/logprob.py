from __future__ import annotations

from typing import Dict, Any

import torch


def _build_prompt_ids(model_wrapper: Any, prompt: str):
    """
    Build prompt token ids using the wrapper tokenizer/chat template when available.
    """
    tok = getattr(model_wrapper, "tok", None)
    device = getattr(model_wrapper, "device", "cuda")
    if tok is None:
        raise ValueError("model wrapper has no tokenizer `tok`")

    if hasattr(tok, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        enc = tok(prompt_text, return_tensors="pt")
        return enc["input_ids"].to(device)

    enc = tok(prompt, return_tensors="pt")
    return enc["input_ids"].to(device)


@torch.inference_mode()
def score_completion_logprob(model_wrapper: Any, prompt: str, completion: str) -> Dict[str, float]:
    """
    Teacher-forcing logprob score for completion given prompt.

    Returns:
      {
        "sum_logprob": float,
        "avg_logprob": float,
        "n_tokens": int,
      }
    """
    model = getattr(model_wrapper, "model", None)
    tok = getattr(model_wrapper, "tok", None)
    device = getattr(model_wrapper, "device", "cuda")
    if model is None or tok is None:
        raise ValueError("model wrapper must expose `.model` and `.tok` for logprob scoring")

    prompt_ids = _build_prompt_ids(model_wrapper, prompt)  # [1, Lp]
    comp_enc = tok(completion or "", return_tensors="pt", add_special_tokens=False)
    comp_ids = comp_enc["input_ids"].to(device)  # [1, Lc]
    n_tokens = int(comp_ids.shape[1])
    if n_tokens == 0:
        return {"sum_logprob": -1e9, "avg_logprob": -1e9, "n_tokens": 0}

    full_ids = torch.cat([prompt_ids, comp_ids], dim=1)  # [1, L]
    logits = model(full_ids).logits  # [1, L, V]
    log_probs = torch.log_softmax(logits, dim=-1)

    lp = prompt_ids.shape[1]
    # token at position t is predicted by logits at t-1
    target_positions = torch.arange(lp, lp + n_tokens, device=full_ids.device)
    pred_positions = target_positions - 1
    token_ids = full_ids[0, target_positions]
    tok_lp = log_probs[0, pred_positions, token_ids]  # [Lc]

    sum_lp = float(tok_lp.sum().item())
    avg_lp = float(sum_lp / max(1, n_tokens))
    return {"sum_logprob": sum_lp, "avg_logprob": avg_lp, "n_tokens": n_tokens}

