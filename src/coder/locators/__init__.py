from __future__ import annotations

from coder.locators.base import (
    TokenLocator,
    align_confidence_to_spans,
    apply_masking_policy,
    get_token_char_spans,
)
from coder.locators.ar_locator import ARLocator
from coder.locators.bert_locator import BERTLocator

__all__ = [
    "TokenLocator",
    "ARLocator",
    "BERTLocator",
    "get_token_char_spans",
    "align_confidence_to_spans",
    "apply_masking_policy",
    "build_locator",
]

# Default model IDs used when --locator_model_id is not specified.
_DEFAULT_AR_MODEL   = "deepseek-ai/deepseek-coder-6.7b-instruct"
_DEFAULT_BERT_MODEL = "microsoft/codebert-base-mlm"


def build_locator(
    name: str,
    model_id: str | None,
    device: str,
) -> TokenLocator | None:
    """
    Instantiate a TokenLocator by name.

    Returns None for name='dream': in that case the refiner (DreamCoder /
    LLaDACoder) acts as its own locator via its built-in score_tokens().

    Args:
        name:     one of 'dream', 'ar', 'bert'.
        model_id: HuggingFace model ID; falls back to a sensible default when
                  None.
        device:   torch device string.
    """
    if name == "dream":
        return None
    if name == "ar":
        mid = model_id or _DEFAULT_AR_MODEL
        print(f"[locator] loading AR locator: {mid}")
        return ARLocator(model_id=mid, device=device)
    if name == "bert":
        mid = model_id or _DEFAULT_BERT_MODEL
        print(f"[locator] loading BERT locator: {mid}")
        return BERTLocator(model_id=mid, device=device)
    raise ValueError(f"Unknown locator: {name!r}. Choose from: dream, ar, bert")
