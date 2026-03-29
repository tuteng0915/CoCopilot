from __future__ import annotations

from coder.models.qwen_coder import QwenCoder


class Qwen35Coder(QwenCoder):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3.5-4B",
        device: str = "cuda",
    ):
        super().__init__(model_id=model_id, device=device)

    @property
    def name(self) -> str:
        return f"qwen35_coder::{self.model_id}"

