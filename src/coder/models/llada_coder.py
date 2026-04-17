from __future__ import annotations

import re

import torch
from transformers import AutoModel, AutoTokenizer

from coder.locators.base import apply_masking_policy
from coder.models.base import CoderModel
from coder.utils.schema import ModelRequest


class LLaDACoder(CoderModel):
    """
    Wrapper for LLaDA diffusion LM (e.g., GSAI-ML/LLaDA-8B-Instruct).

    This follows the official sampling script (generate.py) but exposes a simple
    `generate(ModelRequest)` interface compatible with the rest of this repo.
    """

    def __init__(
        self,
        model_id: str = "GSAI-ML/LLaDA-8B-Instruct",
        device: str = "cuda",
        steps: int = 128,
        gen_length: int = 128,
        block_length: int = 32,
        temperature: float = 0.0,
        cfg_scale: float = 0.0,
        mask_id: int = 126336,
    ):
        self.model_id = model_id
        self.device = device

        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(device).eval()
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        # LLaDA sampling hyperparameters (can be overridden per instance if needed)
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.temperature = temperature
        self.cfg_scale = cfg_scale
        self.mask_id = mask_id

        # Prefer left padding as in the official implementation
        if self.tok.padding_side != "left":
            self.tok.padding_side = "left"

        # If pad_token_id accidentally equals mask_id, users must adjust generate()
        # here we just trust the standard LLaDA tokenizer config.

    @property
    def name(self) -> str:
        return f"llada_coder::{self.model_id}"

    def _clean_completion(self, text: str, prompt: str) -> str:
        if not text:
            return ""

        s = text.strip()
        p = (prompt or "").strip()
        if p and s.startswith(p):
            s = s[len(p) :].lstrip()

        fence_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", s, flags=re.S | re.I)
        if fence_blocks:
            s = fence_blocks[-1].strip()

        s = s.replace("```python", "").replace("```", "").strip()
        s = re.sub(r"^\s*(assistant|response)\s*:\s*", "", s, flags=re.I)

        m = re.search(r"(?m)^(def|class|import|from|@)\s+", s)
        if m:
            s = s[m.start() :].lstrip()

        return s.strip()

    @torch.inference_mode()
    def _add_gumbel_noise(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature == 0:
            return logits
        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

    @torch.inference_mode()
    def _get_num_transfer_tokens(self, mask_index: torch.Tensor, steps: int) -> torch.Tensor:
        mask_num = mask_index.sum(dim=1, keepdim=True)

        base = mask_num // steps
        remainder = mask_num % steps

        num_transfer_tokens = torch.zeros(
            mask_num.size(0),
            steps,
            device=mask_index.device,
            dtype=torch.int64,
        ) + base

        for i in range(mask_num.size(0)):
            num_transfer_tokens[i, : remainder[i]] += 1

        return num_transfer_tokens

    @torch.inference_mode()
    def _diffusion_generate(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        steps: int,
        gen_length: int,
        block_length: int,
        temperature: float,
        cfg_scale: float,
        init_completion_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Port of the official LLaDA `generate` function, specialized to batch=1.
        """
        mask_id = self.mask_id
        device = self.model.device

        if init_completion_ids is not None:
            gen_length = init_completion_ids.shape[1]

        x = torch.full(
            (prompt_ids.shape[0], prompt_ids.shape[1] + gen_length),
            mask_id,
            dtype=torch.long,
            device=device,
        )
        x[:, : prompt_ids.shape[1]] = prompt_ids.clone()
        if init_completion_ids is not None:
            x[:, prompt_ids.shape[1] :] = init_completion_ids.to(device)

        if attention_mask is not None:
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (prompt_ids.shape[0], gen_length),
                        dtype=attention_mask.dtype,
                        device=device,
                    ),
                ],
                dim=-1,
            )

        prompt_index = x != mask_id

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks

        for num_block in range(num_blocks):
            block_mask_index = (
                x[
                    :,
                    prompt_ids.shape[1]
                    + num_block * block_length : prompt_ids.shape[1]
                    + (num_block + 1) * block_length,
                ]
                == mask_id
            )
            num_transfer_tokens = self._get_num_transfer_tokens(
                block_mask_index,
                steps_per_block,
            )

            for i in range(steps_per_block):
                mask_index = x == mask_id
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    if attention_mask is not None:
                        attention_mask_ = torch.cat(
                            [attention_mask, attention_mask],
                            dim=0,
                        )
                    else:
                        attention_mask_ = None
                    logits = self.model(
                        x_,
                        attention_mask=attention_mask_,
                    ).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(
                        x,
                        attention_mask=attention_mask,
                    ).logits

                logits_with_noise = self._add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                # Confidence-based remasking
                p = torch.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(
                        p,
                        dim=-1,
                        index=torch.unsqueeze(x0, -1),
                    ),
                    -1,
                )

                x0_p[
                    :,
                    prompt_ids.shape[1]
                    + (num_block + 1) * block_length :,
                ] = -float("inf")

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -float("inf"))

                transfer_index = torch.zeros_like(
                    x0,
                    dtype=torch.bool,
                    device=x0.device,
                )
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(
                        confidence[j],
                        k=num_transfer_tokens[j, i],
                    )
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]

        return x

    @torch.inference_mode()
    def score_tokens(
        self,
        prompt_ids: torch.Tensor,
        comp_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if comp_ids.shape[1] == 0:
            return torch.zeros(0, device=self.device)

        full_ids = torch.cat([prompt_ids, comp_ids], dim=1)
        full_attention_mask = None
        if attention_mask is not None:
            full_attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], comp_ids.shape[1]),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ],
                dim=-1,
            )

        logits = self.model(full_ids, attention_mask=full_attention_mask).logits
        comp_logits = logits[0, prompt_ids.shape[1] :, :].float()
        probs = torch.softmax(comp_logits, dim=-1)
        confidence = probs[
            torch.arange(comp_ids.shape[1], device=comp_ids.device),
            comp_ids[0],
        ]
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
        Remask low-confidence tokens in `draft` and regenerate via LLaDA diffusion.

        Args:
            external_confidence: Optional pre-computed confidence tensor of
                shape [M] in the refiner's token space.  When provided,
                score_tokens() is skipped.  Masking policy is still applied.
        """
        messages = [{"role": "user", "content": req.prompt}]
        prompt_text = self.tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        enc = self.tok(
            prompt_text,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        prompt_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        comp_enc = self.tok(draft, return_tensors="pt", add_special_tokens=False)
        comp_ids = comp_enc.input_ids.to(self.device)
        orig_len = comp_ids.shape[1]
        if orig_len == 0:
            return draft

        if external_confidence is not None:
            confidence = external_confidence.to(self.device)
        else:
            confidence = self.score_tokens(prompt_ids, comp_ids, attention_mask)

        mask_pos = apply_masking_policy(
            confidence, confidence_threshold, mask_ratio,
            mask_granularity, span_merge_gap, comp_ids, self.tok,
        )

        if not mask_pos.any():
            return draft

        init_comp = comp_ids.clone()
        init_comp[0, mask_pos] = self.mask_id

        block_length = min(self.block_length, max(1, orig_len))
        padded_len = orig_len
        if padded_len % block_length != 0:
            padded_len = (padded_len // block_length + 1) * block_length
        if padded_len != orig_len:
            pad = torch.full(
                (1, padded_len - orig_len),
                self.mask_id,
                dtype=init_comp.dtype,
                device=init_comp.device,
            )
            init_comp = torch.cat([init_comp, pad], dim=1)

        num_blocks = max(1, padded_len // block_length)
        steps = max(self.steps, num_blocks)
        if steps % num_blocks != 0:
            steps = (steps // num_blocks + 1) * num_blocks

        if req.seed is not None:
            torch.manual_seed(req.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(req.seed)

        out = self._diffusion_generate(
            prompt_ids=prompt_ids,
            attention_mask=attention_mask,
            steps=steps,
            gen_length=padded_len,
            block_length=block_length,
            temperature=self.temperature,
            cfg_scale=self.cfg_scale,
            init_completion_ids=init_comp,
        )
        gen_ids = out[:, prompt_ids.shape[1] : prompt_ids.shape[1] + orig_len]
        gen = self.tok.batch_decode(gen_ids, skip_special_tokens=True)[0]
        return self._clean_completion(gen, req.prompt)

    @torch.inference_mode()
    def generate(self, req: ModelRequest) -> str:
        messages = [{"role": "user", "content": req.prompt}]
        prompt_text = self.tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        enc = self.tok(
            prompt_text,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        if req.seed is not None:
            torch.manual_seed(req.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(req.seed)

        steps = max(self.steps, 1)
        gen_length = req.max_new_tokens or self.gen_length
        block_length = min(self.block_length, gen_length)
        # Ensure divisibility
        if gen_length % block_length != 0:
            gen_length = (gen_length // block_length + 1) * block_length

        out = self._diffusion_generate(
            prompt_ids=input_ids,
            attention_mask=attention_mask,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=self.temperature,
            cfg_scale=self.cfg_scale,
        )
        gen_ids = out[:, input_ids.shape[1] :]
        gen = self.tok.batch_decode(gen_ids, skip_special_tokens=True)[0]
        return gen.strip()
