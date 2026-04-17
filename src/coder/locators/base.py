from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch


class TokenLocator(ABC):
    """
    Abstract base for locators that score which draft tokens are low-confidence.

    A locator operates in its own token space and returns per-token confidence
    scores together with character-level spans so callers can align them to any
    target tokenizer.
    """

    @torch.inference_mode()
    @abstractmethod
    def score(
        self,
        prompt_text: str,
        draft_text: str,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """
        Score each token in draft_text.

        Returns:
            confidence: float32 array of shape [N], per-token confidence in
                [0, 1].  Higher = model is more confident this token is correct.
            char_spans: list of (start, end) character offsets in draft_text,
                one entry per token (same length as confidence).
        """
        ...


# ---------------------------------------------------------------------------
# Character-span utilities
# ---------------------------------------------------------------------------

def get_token_char_spans(tokenizer, text: str) -> list[tuple[int, int]]:
    """
    Return (start, end) character spans for each token when tokenizing `text`
    with add_special_tokens=False.

    Tries tokenizer's offset_mapping first (fast tokenizers); falls back to
    per-token greedy character matching.
    """
    try:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        spans = enc.get("offset_mapping") or enc.get("offset_mapping")
        if spans is not None and len(spans) > 0 and spans[0] is not None:
            return [(int(s), int(e)) for s, e in spans]
    except Exception:
        pass

    # Fallback: decode each token and greedily match character positions
    enc = tokenizer(text, add_special_tokens=False)
    token_ids = enc["input_ids"]
    spans: list[tuple[int, int]] = []
    pos = 0
    for tid in token_ids:
        tok_str = tokenizer.decode([tid], skip_special_tokens=False)
        if not tok_str:
            spans.append((pos, pos))
            continue
        idx = text.find(tok_str, pos)
        if idx == -1:
            spans.append((pos, pos + len(tok_str)))
            pos += len(tok_str)
        else:
            spans.append((idx, idx + len(tok_str)))
            pos = idx + len(tok_str)
    return spans


def align_confidence_to_spans(
    conf: np.ndarray,
    src_spans: list[tuple[int, int]],
    tgt_spans: list[tuple[int, int]],
) -> np.ndarray:
    """
    Map per-token confidence from src token space to tgt token space via
    character overlap.

    For each target token, the result is the mean confidence of all source
    tokens whose char span overlaps with it.  Defaults to 1.0 (confident /
    do-not-mask) when no source token overlaps.

    Args:
        conf:      confidence scores in src space, shape [N_src].
        src_spans: char spans for each src token, len == N_src.
        tgt_spans: char spans for each tgt token.

    Returns:
        aligned confidence in tgt space, shape [N_tgt], dtype float32.
    """
    N_tgt = len(tgt_spans)
    result = np.ones(N_tgt, dtype=np.float32)
    for ti, (ts, te) in enumerate(tgt_spans):
        if te <= ts:
            continue
        overlapping = [
            float(conf[si])
            for si, (ss, se) in enumerate(src_spans)
            if si < len(conf) and se > ts and ss < te
        ]
        if overlapping:
            result[ti] = float(np.mean(overlapping))
    return result


# ---------------------------------------------------------------------------
# Shared masking-policy logic (used by DreamCoder, LLaDACoder, and gen_remask)
# ---------------------------------------------------------------------------

def apply_masking_policy(
    confidence: torch.Tensor,          # [M]
    confidence_threshold: float | None,
    mask_ratio: float | None,
    mask_granularity: str,
    span_merge_gap: int,
    comp_ids: torch.Tensor,            # [1, M] — needed for line-level granularity
    tokenizer,                         # refiner tokenizer — needed for line-level
) -> torch.BoolTensor:
    """
    Convert per-token confidence scores to a boolean mask of positions to remask.

    Extracted here so DreamCoder, LLaDACoder, and external-locator paths all
    apply exactly the same policy.
    """
    M = confidence.shape[0]
    device = confidence.device

    if mask_ratio is not None:
        k = max(1, int(M * mask_ratio))
        threshold_val = torch.kthvalue(confidence, k).values.item()
        mask_pos = confidence <= threshold_val
    else:
        thresh = confidence_threshold if confidence_threshold is not None else 0.5
        mask_pos = confidence < thresh

    gran = (mask_granularity or "token").lower().strip()
    if gran not in ("token", "span", "line"):
        raise ValueError(f"Unsupported mask_granularity: {mask_granularity!r}")

    if gran == "span":
        gap = max(int(span_merge_gap), 0)
        if gap > 0:
            mp = mask_pos.clone()
            idx = torch.nonzero(mp, as_tuple=False).view(-1)
            if idx.numel() > 0:
                for a, b in zip(idx[:-1], idx[1:]):
                    d = int(b.item() - a.item())
                    if 1 < d <= gap + 1:
                        mp[a + 1 : b] = True
            mask_pos = mp

    elif gran == "line":
        line_ids: list[int] = []
        cur_line = 0
        for tid in comp_ids[0].tolist():
            s = tokenizer.decode([tid], skip_special_tokens=False)
            line_ids.append(cur_line)
            cur_line += s.count("\n")
        masked_lines = {line_ids[i] for i, m in enumerate(mask_pos.tolist()) if m}
        if masked_lines:
            mask_pos = torch.tensor(
                [ln in masked_lines for ln in line_ids],
                device=device,
                dtype=torch.bool,
            )

    return mask_pos
