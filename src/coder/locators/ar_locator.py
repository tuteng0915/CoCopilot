from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from coder.locators.base import TokenLocator, get_token_char_spans


class ARLocator(TokenLocator):
    """
    Locator using an autoregressive model's teacher-forced log-probabilities.

    For each completion token t_i, confidence is:

        P(t_i | formatted_prompt, t_0, ..., t_{i-1})

    computed via a single forward pass (teacher forcing).

    The prompt is formatted with the model's chat template (when available) to
    match the conditions under which the AR draft was originally generated.

    Note on left-context bias
    -------------------------
    AR logprobs are inherently causal: a token that looks fine given all
    preceding tokens may still be wrong given what follows.  This is the key
    difference from dLLM scoring (which is bidirectional) and what this
    ablation is designed to measure.
    """

    def __init__(self, model_id: str, device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device).eval()

    @torch.inference_mode()
    def score(
        self,
        prompt_text: str,
        draft_text: str,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """
        Single forward pass teacher-forced scoring in the AR model's token space.

        Returns:
            confidence: float32 array of shape [M_ar].
            char_spans: character spans in draft_text for each AR token.
        """
        # Format prompt with chat template to match generation conditions.
        try:
            messages = [{"role": "user", "content": prompt_text}]
            formatted_prompt = self.tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )
        except Exception:
            formatted_prompt = prompt_text

        prompt_enc = self.tok(
            formatted_prompt, return_tensors="pt", add_special_tokens=False,
        )
        prompt_ids = prompt_enc.input_ids.to(self.device)  # [1, P]
        P = prompt_ids.shape[1]

        comp_enc = self.tok(
            draft_text, return_tensors="pt", add_special_tokens=False,
        )
        comp_ids = comp_enc.input_ids.to(self.device)  # [1, M]
        M = comp_ids.shape[1]
        if M == 0:
            return np.array([], dtype=np.float32), []

        full_ids = torch.cat([prompt_ids, comp_ids], dim=1)  # [1, P+M]
        logits = self.model(full_ids).logits.float()          # [1, P+M, V]

        # logits[:, t, :] predicts token at position t+1 (standard causal LM).
        # For completion token i at position P+i, the predicting logit is at
        # position P+i-1.  The first completion token (i=0) is predicted by
        # the last prompt logit at position P-1.
        comp_logits = logits[0, P - 1 : P + M - 1, :]  # [M, V]
        probs = torch.softmax(comp_logits, dim=-1)        # [M, V]
        confidence = probs[
            torch.arange(M, device=self.device), comp_ids[0]
        ]  # [M]

        char_spans = get_token_char_spans(self.tok, draft_text)
        return confidence.cpu().numpy().astype(np.float32), char_spans
