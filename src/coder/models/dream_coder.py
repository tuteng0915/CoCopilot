# src/coder/models/dream_coder.py
from __future__ import annotations

import re
import inspect
from types import SimpleNamespace

import torch
from transformers.generation.configuration_utils import GenerationConfig
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from coder.locators.base import apply_masking_policy
from coder.models.base import CoderModel
from coder.utils.code_cleaning import clean_model_completion
from coder.utils.schema import ModelRequest


def _compute_default_rope_parameters(config=None, device=None, seq_len=None, layer_type=None):
    config.standardize_rope_params()
    params = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters
    base = params.get("rope_theta", getattr(config, "rope_theta", 10000.0))
    partial_rotary_factor = params.get("partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
    )
    return inv_freq, 1.0


ROPE_INIT_FUNCTIONS.setdefault("default", _compute_default_rope_parameters)

_GENERATION_CONFIG_UPDATE = GenerationConfig.update
_GENERATION_CONFIG_UPDATE_PARAMS = inspect.signature(_GENERATION_CONFIG_UPDATE).parameters
_GENERATION_CONFIG_UPDATE_SUPPORTS_FLAGS = (
    "defaults_only" in _GENERATION_CONFIG_UPDATE_PARAMS
    or "allow_custom_entries" in _GENERATION_CONFIG_UPDATE_PARAMS
)

for _name, _value in {
    "eps": 1e-3,
    "steps": 512,
    "alg": "origin",
    "alg_temp": None,
    "eos_penalty": 0.0,
    "output_history": False,
}.items():
    if not hasattr(GenerationConfig, _name):
        setattr(GenerationConfig, _name, _value)


def _update_generation_config_compat(self, defaults_only=False, allow_custom_entries=False, **kwargs):
    try:
        if _GENERATION_CONFIG_UPDATE_SUPPORTS_FLAGS:
            return _GENERATION_CONFIG_UPDATE(
                self,
                defaults_only=defaults_only,
                allow_custom_entries=allow_custom_entries,
                **kwargs,
            )
        return _GENERATION_CONFIG_UPDATE(self, **kwargs)
    except TypeError as exc:
        if "user_set_attributes" not in str(exc):
            raise

    to_remove = []
    for key, value in kwargs.items():
        if allow_custom_entries and not hasattr(self, key):
            setattr(self, key, value)
            to_remove.append(key)
        elif hasattr(self, key):
            if not defaults_only or getattr(self, key) is None:
                setattr(self, key, value)
                to_remove.append(key)
    self.validate()
    return {key: value for key, value in kwargs.items() if key not in to_remove}


GenerationConfig.update = _update_generation_config_compat


class DreamCoder(CoderModel):
    def __init__(
        self,
        model_id: str = "Dream-org/Dream-Coder-v0-Instruct-7B",
        device: str = "cuda",
    ):
        self.model_id = model_id
        self.device = device

        # Dream-Coder uses custom generation logic (diffusion_generate) via trust_remote_code.
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device).eval()
        gen_config = self.model.generation_config
        for name, value in {
            "eps": 1e-3,
            "steps": 512,
            "alg": "origin",
            "alg_temp": None,
            "eos_penalty": 0.0,
            "output_history": False,
            "mask_token_id": getattr(self.model.config, "mask_token_id", None),
            "pad_token_id": getattr(self.model.config, "pad_token_id", None),
            "bos_token_id": getattr(self.model.config, "bos_token_id", None),
            "eos_token_id": getattr(self.model.config, "eos_token_id", None),
        }.items():
            if not hasattr(gen_config, name) or getattr(gen_config, name) is None:
                setattr(gen_config, name, value)
        self._wrap_forward_logits_compat()

        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tok.pad_token is None and self.tok.eos_token is not None:
            self.tok.pad_token = self.tok.eos_token

    @property
    def name(self) -> str:
        return f"dream_coder::{self.model_id}"

    def _wrap_forward_logits_compat(self) -> None:
        original_forward = self.model.forward

        def forward_compat(*args, **kwargs):
            out = original_forward(*args, **kwargs)
            logits = getattr(out, "logits", None)
            if isinstance(logits, torch.Tensor) and logits.ndim == 4 and (
                logits.shape[1] == 1 or logits.shape[2] == 1
            ):
                squeezed = logits.squeeze(1) if logits.shape[1] == 1 else logits.squeeze(2)
                if hasattr(out, "logits"):
                    try:
                        out.logits = squeezed
                        return out
                    except Exception:
                        pass
                if isinstance(out, tuple):
                    return (squeezed, *out[1:])
                data = dict(out) if isinstance(out, dict) else {}
                data["logits"] = squeezed
                return SimpleNamespace(**data)
            return out

        self.model.forward = forward_compat

    def _clean_completion(self, text: str, prompt: str) -> str:
        return clean_model_completion(text, prompt=prompt)

    @torch.inference_mode()
    def score_tokens(
        self,
        prompt_ids: torch.Tensor,
        comp_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single forward pass → per-token confidence for the completion tokens.

        Returns confidence[i] = P(comp_ids[0, i] | context), shape [M].
        Uses the same left-shifted logit convention as _sample in generation_utils.py.
        """
        if comp_ids.shape[1] == 0:
            return torch.zeros(0, device=self.device)

        full_ids = torch.cat([prompt_ids, comp_ids], dim=1)           # [1, L_p+M]
        logits = self.model(full_ids).logits                           # [1, L_p+M, V]
        # Same left-shift used in _sample (generation_utils.py line 413)
        logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
        comp_logits = logits[0, prompt_ids.shape[1]:, :].float()       # [M, V]
        probs = torch.softmax(comp_logits, dim=-1)                     # [M, V]
        confidence = probs[torch.arange(comp_ids.shape[1]), comp_ids[0]]  # [M]
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
        """
        Remask low-confidence tokens in `draft` and regenerate via diffusion.

        Args:
            req: ModelRequest with the original prompt and generation params.
            draft: Raw completion string from the AR model.
            confidence_threshold: Mask tokens where P(token) < threshold.
            mask_ratio: Alternative — mask the bottom K% least-confident
                tokens (0–1).  Takes precedence over confidence_threshold.
            mask_granularity: token | span | line.
            span_merge_gap: For span-level masking, bridge gaps of ≤ this many
                tokens between masked positions.
            external_confidence: Optional pre-computed confidence tensor of
                shape [M] in the refiner's token space.  When provided,
                score_tokens() is skipped and this tensor is used directly.
                Masking policy and granularity are still applied here.
        """
        mask_token_id = self.model.config.mask_token_id

        # Build prompt via chat template (matches generate())
        messages = [{"role": "user", "content": req.prompt}]
        enc = self.tok.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        prompt_ids = enc.input_ids.to(self.device)      # [1, L_p]
        attn_mask = enc.attention_mask.to(self.device)

        # Tokenize draft as plain text (no chat template, no special tokens)
        comp_enc = self.tok(draft, return_tensors="pt", add_special_tokens=False)
        comp_ids = comp_enc.input_ids.to(self.device)   # [1, M]
        M = comp_ids.shape[1]
        if M == 0:
            return draft

        # Score each completion token (skip if caller provides scores)
        if external_confidence is not None:
            confidence = external_confidence.to(self.device)
        else:
            confidence = self.score_tokens(prompt_ids, comp_ids)  # [M]

        mask_pos = apply_masking_policy(
            confidence, confidence_threshold, mask_ratio,
            mask_granularity, span_merge_gap, comp_ids, self.tok,
        )

        if not mask_pos.any():
            return draft  # all tokens are confident enough — nothing to do

        # Build partial init: high-confidence tokens stay, low-confidence → MASK
        init_comp = comp_ids.clone()
        init_comp[0, mask_pos] = mask_token_id

        # Hook: at step=None (init), overwrite the generation slot with init_comp.
        # _sample pads input_ids with MASK tokens to max_length, then calls the hook.
        # Positions where x == mask_token_id are the only ones updated by diffusion,
        # so pre-filled (non-MASK) positions are preserved throughout generation.
        L_p = prompt_ids.shape[1]

        def init_hook(step, x, logits):
            if step is None:
                x[0, L_p : L_p + M] = init_comp[0]
            return x

        def logits_hook(step, x, logits):
            return torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

        if req.seed is not None:
            torch.manual_seed(req.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(req.seed)

        dream_temp = req.temperature if (req.temperature and req.temperature > 0) else 0.1
        dream_top_p = req.top_p if (req.top_p and 0.0 < req.top_p <= 1.0) else 0.95
        # Fewer steps than full generation since most tokens are pre-filled
        steps = max(M, 128)

        out = self.model.diffusion_generate(
            prompt_ids,
            attention_mask=attn_mask,
            generation_config=self.model.generation_config,
            max_new_tokens=M,           # must equal draft length exactly
            output_history=False,
            return_dict_in_generate=True,
            steps=steps,
            temperature=dream_temp,
            top_p=dream_top_p,
            alg="entropy",
            alg_temp=0.0,
            generation_tokens_hook_func=init_hook,
            generation_logits_hook_func=logits_hook,
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
        """
        Return completion text only (not the prompt).
        """
        # Instruct-style prompting
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

        # Dream-Coder usually behaves better with a small non-zero temp and top-p<1.
        dream_temp = req.temperature if (req.temperature is not None and req.temperature > 0) else 0.1
        dream_top_p = req.top_p if (req.top_p is not None and 0.0 < req.top_p <= 1.0) else 0.95

        # Dream diffusion steps are important; keep a sane floor.
        # Official quickstart uses 768/768.
        # For your course setup, this is a good compromise.
        steps = max(req.max_new_tokens, 512)

        out = self.model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            generation_config=self.model.generation_config,
            max_new_tokens=req.max_new_tokens,
            output_history=False,
            return_dict_in_generate=True,
            steps=steps,
            temperature=dream_temp,
            top_p=dream_top_p,
            alg="entropy",
            alg_temp=0.0,
        )

        # Decode only the generated suffix (same spirit as official quick start)
        seq = out.sequences[0]
        gen_ids = seq[len(input_ids[0]) :]
        gen = self.tok.decode(gen_ids.tolist(), skip_special_tokens=False)

        # Trim eos if present
        eos = self.tok.eos_token
        if eos:
            gen = gen.split(eos)[0]

        # Clean prompt-echo / prose / markdown fences
        gen = self._clean_completion(gen, req.prompt)
        return gen
