from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from coder.models.base import CoderModel
from coder.utils.schema import ModelRequest


class SeedCoder(CoderModel):
    def __init__(
        self,
        model_id: str | None = None,
        device: str = "cuda",
    ):
        if not model_id:
            raise ValueError(
                "Seed-Coder requires --model_id. "
                "Please provide a valid HuggingFace model id for Seed-Coder 8B."
            )
        self.model_id = model_id
        self.device = device

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device).eval()
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tok.pad_token is None and self.tok.eos_token is not None:
            self.tok.pad_token = self.tok.eos_token

    @property
    def name(self) -> str:
        return f"seed_coder::{self.model_id}"

    @torch.inference_mode()
    def generate(self, req: ModelRequest) -> str:
        if hasattr(self.tok, "apply_chat_template"):
            messages = [{"role": "user", "content": req.prompt}]
            try:
                prompt = self.tok.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except ValueError:
                prompt = req.prompt
            inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        else:
            inputs = self.tok(req.prompt, return_tensors="pt").to(self.device)
        if "token_type_ids" in inputs:
            inputs.pop("token_type_ids")

        if req.seed is not None:
            torch.manual_seed(req.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(req.seed)

        do_sample = req.temperature is not None and req.temperature > 0
        out = self.model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=do_sample,
            temperature=max(req.temperature or 0.0, 1e-6),
            top_p=req.top_p,
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.pad_token_id,
        )
        gen_ids = out[0][inputs["input_ids"].shape[1] :]
        gen = self.tok.decode(gen_ids, skip_special_tokens=True)
        return gen.strip()

