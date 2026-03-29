# src/coder/models/dream_coder.py
from __future__ import annotations

import re

import torch
from transformers import AutoModel, AutoTokenizer

from coder.models.base import CoderModel
from coder.utils.schema import ModelRequest


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

        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tok.pad_token is None and self.tok.eos_token is not None:
            self.tok.pad_token = self.tok.eos_token

    @property
    def name(self) -> str:
        return f"dream_coder::{self.model_id}"

    def _clean_completion(self, text: str, prompt: str) -> str:
        """
        Best-effort cleanup for:
        - prompt echo
        - markdown code fences
        - assistant prose before code

        Keep this conservative: if the model returns body-only completion, don't over-trim.
        """
        if not text:
            return ""

        s = text.strip()

        # 1) Remove exact prompt echo (full prompt) if present
        p = (prompt or "").strip()
        if p and s.startswith(p):
            s = s[len(p) :].lstrip()

        # 2) Prefer fenced code block if present
        fence_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", s, flags=re.S | re.I)
        if fence_blocks:
            s = fence_blocks[-1].strip()

        # 3) Remove leftover fence markers
        s = s.replace("```python", "").replace("```", "").strip()

        # 4) Remove common assistant-style prefixes
        s = re.sub(r"^\s*(assistant|response)\s*:\s*", "", s, flags=re.I)

        # 5) If there is an obvious code start, cut to it.
        #    (Do not force this when output is body-only completion.)
        m = re.search(r"(?m)^(def|class|import|from|@)\s+", s)
        if m:
            s = s[m.start() :].lstrip()

        return s.strip()

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
    ) -> str:
        """
        Score the DeepSeek draft, remask low-confidence tokens, and regenerate
        only those positions via DreamCoder diffusion.

        Args:
            req: ModelRequest carrying the original prompt and generation params.
            draft: Raw completion string from DeepSeek-Coder.
            confidence_threshold: Mask tokens where P(token) < threshold.
            mask_ratio: Alternative — mask the bottom K% least-confident tokens.
                        If set, takes precedence over confidence_threshold.
            mask_granularity: token | span | line.
              - token: mask individual low-confidence tokens (default; old behavior).
              - span: merge low-confidence tokens into spans (optionally bridging small gaps).
              - line: if any token on a line is low-confidence, mask the entire line.
            span_merge_gap: for span-level masking, also mask gaps of <= this many tokens
              between masked tokens (helps reduce fragmented masks).
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

        # Score each completion token
        confidence = self.score_tokens(prompt_ids, comp_ids)  # [M]

        # Determine which positions to remask
        if mask_ratio is not None:
            k = max(1, int(M * mask_ratio))
            threshold_val = torch.kthvalue(confidence, k).values.item()
            mask_pos = confidence <= threshold_val
        else:
            mask_pos = confidence < confidence_threshold

        # Apply granularity transforms on mask_pos
        gran = (mask_granularity or "token").lower().strip()
        if gran not in ("token", "span", "line"):
            raise ValueError(f"Unsupported mask_granularity: {mask_granularity}")

        if gran == "span":
            gap = max(int(span_merge_gap), 0)
            if gap > 0:
                # Bridge small unmasked gaps between masked tokens.
                mp = mask_pos.clone()
                idx = torch.nonzero(mp, as_tuple=False).view(-1)
                if idx.numel() > 0:
                    # Fill gaps between consecutive masked indices if gap <= span_merge_gap.
                    for a, b in zip(idx[:-1], idx[1:]):
                        d = int(b.item() - a.item())
                        if 1 < d <= gap + 1:
                            mp[a + 1 : b] = True
                mask_pos = mp
        elif gran == "line":
            # Map token indices -> line index by decoding token-by-token.
            line_ids = []
            cur_line = 0
            for tid in comp_ids[0].tolist():
                s = self.tok.decode([tid], skip_special_tokens=False)
                line_ids.append(cur_line)
                cur_line += s.count("\n")
            # If any token in a line is masked, mask the full line.
            masked_lines = set()
            for i, m in enumerate(mask_pos.tolist()):
                if m:
                    masked_lines.add(line_ids[i])
            if masked_lines:
                mask_pos = torch.tensor(
                    [ln in masked_lines for ln in line_ids],
                    device=mask_pos.device,
                    dtype=torch.bool,
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
            max_new_tokens=M,           # must equal draft length exactly
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
