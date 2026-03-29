from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from coder.models.base import CoderModel
from coder.utils.schema import ModelRequest


class DiffuLLaMACoder(CoderModel):
    def __init__(
        self,
        model_id: str | None = None,
        device: str = "cuda",
    ):
        if not model_id:
            raise ValueError(
                "DiffuLLaMA requires --model_id. "
                "Please provide a valid HuggingFace model id for DiffuLLaMA."
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
        return f"diffullama_coder::{self.model_id}"

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
            # DiffuLLaMA official quick-start (HKUNLP/DiffuLLaMA/inf_diffullama.py):
            #   diffusion_steps=64, logits_temp=0.9, topp_temp=0.9, shift=True (do not change)
            #
            # The HF model exposes a convenience `diffusion_generate` API; different versions may
            # accept different kwargs. We prefer official defaults unless the request overrides.
            #
            # Important: for *conditional* generation, diffusion models typically need a `src_mask`
            # to keep prefix tokens fixed. If supported by the model, pass it.
            # steps: denoising iterations (official default 64)
            steps = 64
            logits_temp = req.temperature if (req.temperature and req.temperature > 0) else 0.9
            # Treat top_p==1.0 as "unspecified"; prefer official default 0.9.
            topp_temp = req.top_p if (req.top_p and 0.0 < req.top_p < 1.0) else 0.9

            # Try to follow official prefix-generation recipe:
            # build a fixed-length sequence and provide src_mask to keep the prefix.
            prefix_len = int(input_ids.shape[1])
            gen_len = prefix_len + int(req.max_new_tokens or 0)
            pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
            if gen_len <= prefix_len:
                gen_len = prefix_len
            x0 = torch.full((1, gen_len), pad_id, dtype=input_ids.dtype, device=self.device)
            x0[:, :prefix_len] = input_ids
            src_mask = torch.zeros_like(x0, dtype=torch.bool)
            src_mask[:, :prefix_len] = True

            gen_kwargs = dict(
                attention_mask=attention_mask,
                max_new_tokens=req.max_new_tokens,
                output_history=False,
                return_dict_in_generate=True,
                steps=steps,
                temperature=logits_temp,
                top_p=topp_temp,
            )
            try:
                out = self.model.diffusion_generate(
                    x0,
                    src_mask=src_mask,
                    **gen_kwargs,
                )
            except TypeError:
                out = self.model.diffusion_generate(
                    x0,
                    **gen_kwargs,
                )
            seq = out.sequences[0]
            gen_ids = seq[prefix_len:]
            gen = self.tok.decode(gen_ids.tolist(), skip_special_tokens=True)
            return gen.strip()

        # Fallback to AR generation if diffusion API is unavailable.
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

