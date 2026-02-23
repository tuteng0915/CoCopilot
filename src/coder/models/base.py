from __future__ import annotations
from abc import ABC, abstractmethod
from coder.utils.schema import ModelRequest

class CoderModel(ABC):
    """Unified interface for different code generation models."""

    @property
    @abstractmethod
    def name(self) -> str:
        return f"dream-coder::{self.model_id}"

    @abstractmethod
    def generate(self, req: ModelRequest) -> str:
        ...
