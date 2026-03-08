from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from coder.models.base import CoderModel
from coder.utils.schema import ModelRequest


class QwenCoder(CoderModel):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        device: str = "cuda",
    ):
        self.model_id = model_id
        self.device = device

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=None,
        ).to(device).eval()

        self.tok = AutoTokenizer.from_pretrained(model_id)

    @property
    def name(self) -> str:
        return f"qwen_coder::{self.model_id}"

    @torch.inference_mode()
    def generate(self, req: ModelRequest) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful coding assistant. "
                "Only output valid Python code for the requested function (no explanations).",
            },
            {"role": "user", "content": req.prompt},
        ]

        prompt = self.tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.tok(
            prompt,
            return_tensors="pt",
        ).to(self.device)

        if req.seed is not None:
            torch.manual_seed(req.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(req.seed)

        do_sample = req.temperature is not None and req.temperature > 0
        temperature = max(req.temperature or 0.0, 1e-6)

        out = self.model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=req.top_p,
            eos_token_id=self.tok.eos_token_id,
        )

        gen_ids = out[0][inputs["input_ids"].shape[1] :]
        gen = self.tok.decode(gen_ids, skip_special_tokens=True)
        return gen.strip()

