from __future__ import annotations

from .dream_coder import DreamCoder
from coder.utils.schema import ModelRequest


class DiffuCoder(DreamCoder):
    """
    Wrapper for Apple's DiffuCoder 7B (apple/DiffuCoder-7B-Instruct).

    DiffuCoder uses the same Dream architecture and diffusion_generate API as
    Dream-Coder-v0, so we reuse DreamCoder entirely — only the default model_id
    and the name property differ.
    """

    def __init__(
        self,
        model_id: str = "apple/DiffuCoder-7B-Instruct",
        device: str = "cuda",
    ):
        super().__init__(model_id=model_id, device=device)

    @property
    def name(self) -> str:
        return f"diffucoder::{self.model_id}"
