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
