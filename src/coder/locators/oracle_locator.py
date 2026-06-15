from __future__ import annotations

import json

import numpy as np
import torch

from coder.locators.base import TokenLocator


class OracleLocator(TokenLocator):
    """
    Precomputed oracle locator backed by gen_oracle_mask.py JSONL output.

    The task-aware API returns character-level confidence: 0.0 for characters
    inside oracle_mask_spans, 1.0 elsewhere. gen_remask uses score_for_task()
    so tasks without oracle spans stay all-confident and are not remasked.
    """

    def __init__(self, oracle_jsonl: str) -> None:
        self.model_id = oracle_jsonl
        self._spans: dict[str, list[tuple[int, int]] | None] = {}
        with open(oracle_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                task_id = rec["task_id"]
                spans = rec.get("oracle_mask_spans")
                self._spans[task_id] = [(int(s), int(e)) for s, e in spans] if spans else None

        n_with_spans = sum(spans is not None for spans in self._spans.values())
        print(
            f"[oracle_locator] loaded {len(self._spans)} tasks, "
            f"{n_with_spans} with oracle spans"
        )

    def score_for_task(
        self,
        task_id: str,
        draft_text: str,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        spans = self._spans.get(task_id)
        n_chars = len(draft_text)
        confidence = np.ones(n_chars, dtype=np.float32)
        if spans:
            for start, end in spans:
                start = max(0, min(start, n_chars))
                end = max(start, min(end, n_chars))
                confidence[start:end] = 0.0
        char_spans = [(idx, idx + 1) for idx in range(n_chars)]
        return confidence, char_spans

    @torch.inference_mode()
    def score(
        self,
        prompt_text: str,
        draft_text: str,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        n_chars = len(draft_text)
        return np.ones(n_chars, dtype=np.float32), [(idx, idx + 1) for idx in range(n_chars)]
