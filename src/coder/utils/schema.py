from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

@dataclass
class ModelRequest:
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    seed: Optional[int] = 3407

@dataclass
class SampleRecord:
    task_id: str
    solution: str  # EvalPlus: solution or completion; we use solution for simplicity
    model: str
    gen: Dict[str, Any]

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        # EvalPlus expects task_id + (solution|completion). keep extra fields ok.
        return d
