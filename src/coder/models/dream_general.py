# src/coder/models/dream_general.py
"""General-purpose Dream dLLM (Dream-v0-Instruct-7B, not code-specialized).

Architecture is identical to DreamCoder but:
  - Default model: Dream-org/Dream-v0-Instruct-7B
  - No code-specific output cleaning (just strip whitespace)
"""
from __future__ import annotations

import torch
from transformers import AutoModel, AutoTokenizer

from coder.locators.base import apply_masking_policy
from coder.models.base import CoderModel
from coder.utils.schema import ModelRequest


class DreamGeneral(CoderModel):
    def __init__(
        self,
        model_id: str = "Dream-org/Dream-v0-Instruct-7B",
        device: str = "cuda",
    ):
        self.model_id = model_id
        self.device = device

        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device).eval()

        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tok.pad_token is None and self.tok.eos_token is not None:
            self.tok.pad_token = self.tok.eos_token

    @property
    def name(self) -> str:
        return f"dream_general::{self.model_id}"

    def _clean_completion(self, text: str, prompt: str) -> str:
        """No code-specific cleaning — just strip surrounding whitespace."""
        return text.strip()

    @torch.inference_mode()
    def score_tokens(
        self,
        prompt_ids: torch.Tensor,
        comp_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Single forward pass → per-token confidence for completion tokens."""
        if comp_ids.shape[1] == 0:
            return torch.zeros(0, device=self.device)

        full_ids = torch.cat([prompt_ids, comp_ids], dim=1)
        logits = self.model(full_ids).logits
        logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
        comp_logits = logits[0, prompt_ids.shape[1]:, :].float()
        probs = torch.softmax(comp_logits, dim=-1)
        confidence = probs[torch.arange(comp_ids.shape[1]), comp_ids[0]]
        return confidence

    @torch.inference_mode()
    def generate_with_remask(
        self,
        req: ModelRequest,
        draft: str,
        confidence_threshold: float = 0.5,
        mask_ratio: float | None = None,
        mask_granularity: str = "token",
        span_merge_gap: int = 0,
        external_confidence: torch.Tensor | None = None,
    ) -> str:
        mask_token_id = self.model.config.mask_token_id

        messages = [{"role": "user", "content": req.prompt}]
        enc = self.tok.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        prompt_ids = enc.input_ids.to(self.device)
        attn_mask = enc.attention_mask.to(self.device)

        comp_enc = self.tok(draft, return_tensors="pt", add_special_tokens=False)
        comp_ids = comp_enc.input_ids.to(self.device)
        M = comp_ids.shape[1]
        if M == 0:
            return draft

        if external_confidence is not None:
            confidence = external_confidence.to(self.device)
        else:
            confidence = self.score_tokens(prompt_ids, comp_ids)

        mask_pos = apply_masking_policy(
            confidence, confidence_threshold, mask_ratio,
            mask_granularity, span_merge_gap, comp_ids, self.tok,
        )

        if not mask_pos.any():
            return draft

        init_comp = comp_ids.clone()
        init_comp[0, mask_pos] = mask_token_id

        L_p = prompt_ids.shape[1]

        def init_hook(step, x, logits):
            if step is None:
                x[0, L_p: L_p + M] = init_comp[0]
            return x

        if req.seed is not None:
            torch.manual_seed(req.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(req.seed)

        dream_temp = req.temperature if (req.temperature and req.temperature > 0) else 0.1
        dream_top_p = req.top_p if (req.top_p and 0.0 < req.top_p <= 1.0) else 0.95
        steps = max(M, 128)

        out = self.model.diffusion_generate(
            prompt_ids,
            attention_mask=attn_mask,
            max_new_tokens=M,
            output_history=False,
            return_dict_in_generate=True,
            steps=steps,
            temperature=dream_temp,
            top_p=dream_top_p,
            alg="entropy",
            alg_temp=0.0,
            generation_tokens_hook_func=init_hook,
        )

        seq = out.sequences[0]
        gen_ids = seq[L_p:]
        raw = self.tok.decode(gen_ids.tolist(), skip_special_tokens=False)
        eos = self.tok.eos_token
        if eos:
            raw = raw.split(eos)[0]
        return self._clean_completion(raw, req.prompt)

    @torch.inference_mode()
    def generate(self, req: ModelRequest) -> str:
        messages = [{"role": "user", "content": req.prompt}]
        inputs = self.tok.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        if req.seed is not None:
            torch.manual_seed(req.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(req.seed)

        dream_temp = req.temperature if (req.temperature is not None and req.temperature > 0) else 0.1
        dream_top_p = req.top_p if (req.top_p is not None and 0.0 < req.top_p <= 1.0) else 0.95
        steps = max(req.max_new_tokens, 512)

        out = self.model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=req.max_new_tokens,
            output_history=False,
            return_dict_in_generate=True,
            steps=steps,
            temperature=dream_temp,
            top_p=dream_top_p,
            alg="entropy",
            alg_temp=0.0,
        )

        seq = out.sequences[0]
        gen_ids = seq[len(input_ids[0]):]
        gen = self.tok.decode(gen_ids.tolist(), skip_special_tokens=False)
        eos = self.tok.eos_token
        if eos:
            gen = gen.split(eos)[0]
        return self._clean_completion(gen, req.prompt)
