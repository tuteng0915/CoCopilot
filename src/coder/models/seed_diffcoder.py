from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from coder.models.base import CoderModel
from coder.utils.schema import ModelRequest


class SeedDiffCoder(CoderModel):
    def __init__(
        self,
        model_id: str | None = None,
        device: str = "cuda",
    ):
        if not model_id:
            raise ValueError(
                "Seed-DiffCoder requires --model_id. "
                "Please provide a valid HuggingFace model id for Seed-DiffCoder."
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
        return f"seed_diffcoder::{self.model_id}"

    @torch.inference_mode()
    def generate(self, req: ModelRequest) -> str:
        if hasattr(self.tok, "apply_chat_template"):
            messages = [{"role": "user", "content": req.prompt}]
            try:
                enc = self.tok.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=True,
                )
                input_ids = enc.input_ids.to(self.device)
                attention_mask = enc.attention_mask.to(self.device)
            except ValueError:
                enc = self.tok(req.prompt, return_tensors="pt")
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
        else:
            enc = self.tok(req.prompt, return_tensors="pt")
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

        if req.seed is not None:
            torch.manual_seed(req.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(req.seed)

        if hasattr(self.model, "diffusion_generate"):
            out = self.model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=req.max_new_tokens,
                output_history=False,
                return_dict_in_generate=True,
                steps=max(req.max_new_tokens, 256),
                temperature=req.temperature if (req.temperature and req.temperature > 0) else 0.1,
                top_p=req.top_p if (req.top_p and 0.0 < req.top_p <= 1.0) else 0.95,
            )
            seq = out.sequences[0]
            gen_ids = seq[len(input_ids[0]) :]
            gen = self.tok.decode(gen_ids.tolist(), skip_special_tokens=True)
            return gen.strip()

        # Fallback to AR generation when diffusion API is not exposed.
        # Stable-DiffCoder's trust_remote_code implementation expects diffusion-style
        # kwargs in generate_block and does not accept attention_mask/top_p/do_sample.
        if hasattr(self.model, "generate_block"):
            out = self.model.generate(
                input_ids=input_ids,
                gen_length=req.max_new_tokens,
                steps=max(req.max_new_tokens, 256),
                block_length=4,
                temperature=max(req.temperature or 0.0, 1e-6),
                remasking="low_confidence",
                eos_id=self.tok.eos_token_id,
                tokenizer=self.tok,
                shift=False,
                threshold=None,
            )
        else:
            inputs = {"input_ids": input_ids}
            if attention_mask is not None:
                inputs["attention_mask"] = attention_mask
            out = self.model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                do_sample=(req.temperature is not None and req.temperature > 0),
                temperature=max(req.temperature or 0.0, 1e-6),
                top_p=req.top_p,
                eos_token_id=self.tok.eos_token_id,
                pad_token_id=self.tok.pad_token_id,
            )
        gen_ids = out[0][input_ids.shape[1] :]
        gen = self.tok.decode(gen_ids, skip_special_tokens=True)
        return gen.strip()

