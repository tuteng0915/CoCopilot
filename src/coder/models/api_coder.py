from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from coder.models.base import CoderModel
from coder.utils.schema import ModelRequest


class ApiCoder(CoderModel):
    """
    Universal AR API wrapper for OpenAI-compatible chat completion APIs.

    Required env vars:
      - CODER_API_KEY
      - CODER_API_MODEL (or pass --model_id)

    Optional env vars:
      - CODER_API_BASE_URL (default: https://api.openai.com/v1)
      - CODER_API_TIMEOUT_SEC (default: 120)
      - CODER_API_SYSTEM_PROMPT (default: code-only prompt)
      - CODER_API_BEARER_PREFIX (default: "Bearer")
    """

    def __init__(self, model_id: str | None = None, device: str = "api"):
        self.device = device
        self.base_url = os.getenv("CODER_API_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.getenv("CODER_API_KEY")
        self.model_id = model_id or os.getenv("CODER_API_MODEL")
        self.timeout_sec = int(os.getenv("CODER_API_TIMEOUT_SEC", "120"))
        self.system_prompt = os.getenv(
            "CODER_API_SYSTEM_PROMPT",
            "You are a helpful coding assistant. Output only valid Python code.",
        )
        self.bearer_prefix = os.getenv("CODER_API_BEARER_PREFIX", "Bearer")

        if not self.api_key:
            raise ValueError("ApiCoder requires CODER_API_KEY environment variable.")
        if not self.model_id:
            raise ValueError("ApiCoder requires --model_id or CODER_API_MODEL environment variable.")

    @property
    def name(self) -> str:
        return f"api_coder::{self.model_id}"

    def _extract_text(self, resp: dict) -> str:
        choices = resp.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message", {})
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    t = item.get("text")
                    if isinstance(t, str):
                        parts.append(t)
            return "".join(parts)
        return str(content)

    def generate(self, req: ModelRequest) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": req.prompt},
            ],
            "max_tokens": req.max_new_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p,
        }
        if req.seed is not None:
            payload["seed"] = req.seed

        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"{self.bearer_prefix} {self.api_key}",
        }
        request = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                raw = response.read().decode("utf-8")
                data = json.loads(raw)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"API request failed: HTTP {e.code} {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"API request failed: {e.reason}") from e

        return self._extract_text(data).strip()

