from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from coder.models.base import CoderModel
from coder.utils.schema import ModelRequest


class StarCoder2Coder(CoderModel):
    def __init__(
        self,
        model_id: str = "bigcode/starcoder2-7b",
        device: str = "cuda",
    ):
        self.model_id = model_id
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        ).to(device).eval()
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token is None and self.tok.eos_token is not None:
            self.tok.pad_token = self.tok.eos_token

    @property
    def name(self) -> str:
        return f"starcoder2_coder::{self.model_id}"

    @torch.inference_mode()
    def generate(self, req: ModelRequest) -> str:
        prompt = req.prompt
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)

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

