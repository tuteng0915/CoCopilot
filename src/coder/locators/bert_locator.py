from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from coder.locators.base import TokenLocator, get_token_char_spans


class BERTLocator(TokenLocator):
    """
    Locator using a masked language model (e.g. CodeBERT) for per-token confidence.

    For each draft token t_i, confidence is approximated by a single forward
    pass (no masking):

        P_approx(t_i | all_other_tokens) ≈ softmax(MLM_head(h_i))[t_i]

    This is an approximation of the true leave-one-out MLM probability, but
    runs in O(1) forward passes instead of O(M).  The bidirectionality is
    preserved: h_i attends to both left and right context.

    Input format:
        [CLS] <truncated_prompt> [SEP] <draft> [SEP]

    The prompt is truncated from the left if needed so that the draft always
    fits within max_length.

    Default model: microsoft/codebert-base-mlm
        (pre-trained on code with MLM objective; 125 M params)
    """

    def __init__(
        self,
        model_id: str = "microsoft/codebert-base-mlm",
        device: str = "cuda",
        max_length: int = 512,
    ):
        self.model_id = model_id
        self.device = device
        self.max_length = max_length
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForMaskedLM.from_pretrained(model_id).to(device).eval()

    @torch.inference_mode()
    def score(
        self,
        prompt_text: str,
        draft_text: str,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """
        Single-pass MLM scoring.

        Returns:
            confidence: float32 array of shape [M_bert] (draft tokens only).
            char_spans: character spans in draft_text for each BERT draft token.
        """
        cls_id = self.tok.cls_token_id
        sep_id = self.tok.sep_token_id

        # Tokenize draft without special tokens to know its exact length and
        # char spans.  Long drafts are scored in chunks because RoBERTa/CodeBERT
        # has a hard 512-position limit.
        draft_enc = self.tok(
            draft_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        draft_ids = draft_enc["input_ids"]
        all_spans = [(int(s), int(e)) for s, e in draft_enc["offset_mapping"]]
        M = len(draft_ids)
        if M == 0:
            return np.array([], dtype=np.float32), []

        confidences: list[np.ndarray] = []
        max_draft_tokens = self.max_length - 3
        for start in range(0, M, max_draft_tokens):
            chunk_ids = draft_ids[start:start + max_draft_tokens]
            chunk_len = len(chunk_ids)

            # Budget: [CLS] prompt [SEP] draft_chunk [SEP] <= max_length.
            budget_prompt = self.max_length - chunk_len - 3
            if budget_prompt > 0:
                prompt_ids = self.tok(
                    prompt_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=budget_prompt,
                )["input_ids"]
            else:
                prompt_ids = []

            input_ids_list = [cls_id] + prompt_ids + [sep_id] + chunk_ids + [sep_id]
            draft_start = 1 + len(prompt_ids) + 1
            draft_end = draft_start + chunk_len

            input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=self.device)
            attention_mask = torch.ones_like(input_ids)

            logits = self.model(
                input_ids=input_ids, attention_mask=attention_mask,
            ).logits.float()

            draft_logits = logits[0, draft_start:draft_end, :]
            draft_token_ids = input_ids[0, draft_start:draft_end]
            probs = torch.softmax(draft_logits, dim=-1)
            confidence = probs[
                torch.arange(chunk_len, device=self.device), draft_token_ids
            ]
            confidences.append(confidence.cpu().numpy().astype(np.float32))

        return np.concatenate(confidences), all_spans[:M]
