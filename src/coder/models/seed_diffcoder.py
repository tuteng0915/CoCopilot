from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from coder.locators.base import apply_masking_policy, import_line_token_mask
from coder.models.base import CoderModel
from coder.utils.schema import ModelRequest

# <[MASK_TOKEN]> is token ID 5 in Stable-DiffCoder's vocabulary.
_STABLE_DIFFCODER_MASK_ID = 5
_BLOCK_LENGTH = 4


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

        # Resolve mask token ID: prefer tokenizer attr, fall back to known constant.
        self.mask_id: int = (
            self.tok.mask_token_id
            if (getattr(self.tok, "mask_token_id", None) is not None)
            else _STABLE_DIFFCODER_MASK_ID
        )

    @property
    def name(self) -> str:
        return f"seed_diffcoder::{self.model_id}"

    def _encode_prompt(self, prompt: str):
        """Encode prompt with chat template; returns (input_ids, attn_mask)."""
        if hasattr(self.tok, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            try:
                enc = self.tok.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=True,
                )
                return enc.input_ids.to(self.device), enc.attention_mask.to(self.device)
            except (ValueError, Exception):
                pass
        enc = self.tok(prompt, return_tensors="pt")
        ids = enc["input_ids"].to(self.device)
        mask = enc.get("attention_mask")
        if mask is not None:
            mask = mask.to(self.device)
        return ids, mask

    @torch.inference_mode()
    def score_tokens(
        self,
        prompt_ids: torch.Tensor,
        comp_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single forward pass (full attention) → per-token confidence for completion.

        We do NOT use the block-causal mask here so that each completion token
        is scored given the full bidirectional context, giving the most reliable
        confidence signal for masking decisions.
        """
        if comp_ids.shape[1] == 0:
            return torch.zeros(0, device=self.device)

        full_ids = torch.cat([prompt_ids, comp_ids], dim=1)  # [1, L_p + M]
        logits = self.model(full_ids).logits                  # [1, L_p + M, V]
        comp_logits = logits[0, prompt_ids.shape[1]:, :].float()  # [M, V]
        probs = torch.softmax(comp_logits, dim=-1)
        confidence = probs[
            torch.arange(comp_ids.shape[1], device=self.device),
            comp_ids[0],
        ]
        return confidence  # [M]

    @torch.inference_mode()
    def _run_block_diffusion(
        self,
        x: torch.Tensor,
        prompt_len: int,
        orig_comp_len: int,
        steps: int,
        block_length: int,
        temperature: float,
    ) -> torch.Tensor:
        """
        Block-level masked diffusion starting from a pre-filled x = [prompt | init_comp].

        init_comp already has real tokens for high-confidence positions and
        self.mask_id for low-confidence positions (set by the caller).
        Returns the decoded completion ids: shape [1, orig_comp_len].
        """
        mask_id = self.mask_id
        total_comp_len = x.shape[1] - prompt_len  # includes any block-alignment padding

        # Build block schedule matching generate_block's logic.
        gen_block_list = [block_length] * (total_comp_len // block_length)
        remainder = prompt_len % block_length
        if remainder != 0:
            res_block = block_length - remainder
            gen_block_list = [res_block] + gen_block_list
            if gen_block_list:
                gen_block_list[-1] = block_length - res_block
                if gen_block_list[-1] == 0:
                    gen_block_list.pop()

        gen_blocks = len(gen_block_list)
        if gen_blocks == 0:
            return x[:, prompt_len:prompt_len + orig_comp_len]

        cum_block = [sum(gen_block_list[:i + 1]) for i in range(gen_blocks)]
        steps_per_block = max(1, steps // gen_blocks)

        total_len = x.shape[1]
        block_causal_mask = self.model._make_block_causal_mask(
            total_len, block_length, x.device, dtype=torch.bfloat16
        )

        past_key_values = DynamicCache()

        # Prefill prompt KV cache up to the aligned boundary.
        prefill_length = prompt_len // block_length * block_length
        if prefill_length > 0:
            cur_attn_mask = block_causal_mask[..., :prefill_length, :prefill_length]
            cache_pos = torch.arange(prefill_length, device=x.device)
            self.model(
                x[:, :prefill_length],
                past_key_values=past_key_values,
                attention_mask=cur_attn_mask,
                use_cache=True,
                cache_position=cache_pos,
            )

        for block_id, _block_size in enumerate(gen_block_list):
            block_start = (
                prompt_len + cum_block[block_id - 1] if block_id > 0 else prefill_length
            )
            block_end = prompt_len + cum_block[block_id]

            replace_pos = torch.zeros(1, total_len, dtype=torch.bool, device=x.device)
            replace_pos[:, block_start:block_end] = True
            cache_pos = replace_pos.nonzero(as_tuple=True)[1]

            block_mask_map = x[:, block_start:block_end] == mask_id

            if not block_mask_map.any():
                # All tokens pre-filled: just populate KV cache for this block.
                self.model(
                    x[:, block_start:block_end],
                    attention_mask=block_causal_mask[..., block_start:block_end, :block_end],
                    past_key_values=past_key_values,
                    use_cache=True,
                    cache_position=cache_pos,
                )
                continue

            num_transfer_tokens = self.model._get_num_transfer_tokens(
                block_mask_map, steps_per_block
            )

            for token_count in num_transfer_tokens:
                if int(token_count) > 0:
                    mask_map = x[:, block_start:block_end] == mask_id
                    attn_mask = block_causal_mask[..., block_start:block_end, :block_end]
                    output = self.model(
                        x[:, block_start:block_end],
                        attention_mask=attn_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        cache_position=cache_pos,
                    )
                    past_key_values.crop(block_start)

                    x0, transfer_map = self.model._get_transfer_index(
                        output.logits,
                        temperature,
                        "low_confidence",
                        mask_map,
                        x[:, block_start:block_end],
                        int(token_count),
                        threshold=None,
                        shift=False,
                    )
                    x[:, block_start:block_end][transfer_map] = x0[transfer_map]

                if (x[:, block_start:block_end] == mask_id).sum() == 0:
                    # Block fully decoded; update KV cache before next block.
                    self.model(
                        x[:, block_start:block_end],
                        attention_mask=block_causal_mask[
                            ..., block_start:block_end, :block_end
                        ],
                        past_key_values=past_key_values,
                        use_cache=True,
                        cache_position=cache_pos,
                    )
                    break

        return x[:, prompt_len:prompt_len + orig_comp_len]

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
        protect_imports: bool = False,
        protect_last_n_tokens: int = 0,
    ) -> str:
        """
        Remask low-confidence tokens in `draft` and regenerate via Stable-DiffCoder
        block-level masked diffusion.

        The caller can pass external_confidence (pre-computed by gen_remask.py's
        compute_refiner_mask_plan) or let this method score tokens internally.
        """
        prompt_ids, _ = self._encode_prompt(req.prompt)
        L_p = prompt_ids.shape[1]

        comp_enc = self.tok(draft, return_tensors="pt", add_special_tokens=False)
        comp_ids = comp_enc.input_ids.to(self.device)
        orig_len = comp_ids.shape[1]
        if orig_len == 0:
            return draft

        if external_confidence is not None:
            confidence = external_confidence.to(self.device)
        else:
            confidence = self.score_tokens(prompt_ids, comp_ids)

        if protect_imports:
            imp_mask = import_line_token_mask(draft, comp_ids, self.tok)
            confidence = confidence.clone()
            confidence[imp_mask] = 1.0

        if protect_last_n_tokens > 0 and orig_len > 0:
            n = min(int(protect_last_n_tokens), orig_len)
            confidence = confidence.clone()
            confidence[-n:] = 1.0

        mask_pos = apply_masking_policy(
            confidence, confidence_threshold, mask_ratio,
            mask_granularity, span_merge_gap, comp_ids, self.tok,
        )

        if not mask_pos.any():
            return draft

        # Initialise completion: keep high-confidence tokens, mask the rest.
        init_comp = comp_ids.clone()
        init_comp[0, mask_pos] = self.mask_id

        # Pad to the nearest block boundary so generate_block constraints hold.
        block_length = _BLOCK_LENGTH
        padded_len = orig_len
        if padded_len % block_length != 0:
            padded_len = (padded_len // block_length + 1) * block_length
        if padded_len != orig_len:
            pad = torch.full(
                (1, padded_len - orig_len),
                self.mask_id,
                dtype=init_comp.dtype,
                device=self.device,
            )
            init_comp = torch.cat([init_comp, pad], dim=1)

        x = torch.cat([prompt_ids, init_comp], dim=1)  # [1, L_p + padded_len]

        if req.seed is not None:
            torch.manual_seed(req.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(req.seed)

        n_blocks = max(1, padded_len // block_length)
        steps = max(256, n_blocks)
        if steps % n_blocks != 0:
            steps = (steps // n_blocks + 1) * n_blocks

        temperature = (
            max(float(req.temperature), 1e-6)
            if (req.temperature and req.temperature > 0)
            else 0.0
        )

        gen_ids = self._run_block_diffusion(x, L_p, orig_len, steps, block_length, temperature)
        gen = self.tok.decode(gen_ids[0].tolist(), skip_special_tokens=True)
        return gen.strip()

    @torch.inference_mode()
    def generate(self, req: ModelRequest) -> str:
        input_ids, attention_mask = self._encode_prompt(req.prompt)

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
            gen_ids = seq[len(input_ids[0]):]
            gen = self.tok.decode(gen_ids.tolist(), skip_special_tokens=True)
            return gen.strip()

        # Stable-DiffCoder's trust_remote_code implementation uses generate_block
        # and does not accept attention_mask/top_p/do_sample.
        if hasattr(self.model, "generate_block"):
            out = self.model.generate(
                input_ids=input_ids,
                gen_length=req.max_new_tokens,
                steps=max(req.max_new_tokens, 256),
                block_length=_BLOCK_LENGTH,
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
        gen_ids = out[0][input_ids.shape[1]:]
        gen = self.tok.decode(gen_ids, skip_special_tokens=True)
        return gen.strip()
