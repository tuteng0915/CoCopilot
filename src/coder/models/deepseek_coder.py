import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
from coder.models.base import CoderModel
from coder.utils.schema import ModelRequest


def _decode_preserves_layout(tok) -> bool:
    probe = "from typing import List\n\ndef f(x):\n    return x + 1\n"
    ids = tok(probe, add_special_tokens=False).input_ids
    return tok.decode(ids, skip_special_tokens=True) == probe


def _load_deepseek_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if _decode_preserves_layout(tok):
        return tok

    tokenizer_file = hf_hub_download(
        repo_id=model_id,
        filename="tokenizer.json",
        local_files_only=True,
    )
    fast = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
    for attr in ("bos_token", "eos_token", "unk_token", "pad_token"):
        value = getattr(tok, attr, None)
        if value is not None:
            setattr(fast, attr, value)
    fast.chat_template = getattr(tok, "chat_template", None)
    fast.model_max_length = getattr(tok, "model_max_length", fast.model_max_length)
    return fast


class DeepSeekCoder(CoderModel):
    def __init__(self, model_id: str = "deepseek-ai/deepseek-coder-6.7b-instruct", device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device).eval()
        self.tok = _load_deepseek_tokenizer(model_id)

    @property
    def name(self) -> str:
        return f"deepseek_coder::{self.model_id}"

    @torch.inference_mode()
    def generate(self, req: ModelRequest) -> str:
        messages = [{"role": "user", "content": req.prompt}]
        prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)

        if req.seed is not None:
            torch.manual_seed(req.seed)

        out = self.model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=req.temperature > 0,
            temperature=max(req.temperature, 1e-6),
            top_p=req.top_p,
            eos_token_id=self.tok.eos_token_id,
        )
        gen = self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return gen.strip()
