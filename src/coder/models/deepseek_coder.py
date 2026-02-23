import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from coder.models.base import CoderModel
from coder.utils.schema import ModelRequest

class DeepSeekCoder(CoderModel):
    def __init__(self, model_id: str = "deepseek-ai/deepseek-coder-6.7b-instruct", device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device).eval()
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

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
