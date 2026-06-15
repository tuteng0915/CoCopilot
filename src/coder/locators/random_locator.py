from __future__ import annotations

import re

import numpy as np
import torch

from coder.locators.base import TokenLocator


class RandomLocator(TokenLocator):
    """
    Assign uniformly random confidence to coarse draft spans.

    This is a null locator baseline used with --mask_ratio, so the rewrite masks
    a fixed fraction of draft tokens without using model confidence.
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    @torch.inference_mode()
    def score(
        self,
        prompt_text: str,
        draft_text: str,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        parts = re.findall(r"\S+|\s+", draft_text)
        confidence = self.rng.uniform(0.0, 1.0, size=len(parts)).astype(np.float32)

        spans: list[tuple[int, int]] = []
        pos = 0
        for part in parts:
            spans.append((pos, pos + len(part)))
            pos += len(part)

        return confidence, spans
