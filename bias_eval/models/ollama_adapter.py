"""Ollama backend for Qwen and DeepSeek text models (runs locally on CPU or GPU)."""
from __future__ import annotations

import base64
import io
import os

import requests
from PIL import Image

from bias_eval.models.base import ModelAdapter, ModelError, ModelInfo, strip_reasoning

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


def list_local_models(timeout: float = 3) -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=timeout)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except (requests.RequestException, ValueError):
        return []


def family_of(name: str) -> str:
    n = name.lower()
    if "deepseek" in n:
        return "DeepSeek"
    if "qwen" in n and "vl" in n:
        return "Qwen2.5-VL"
    if "qwen" in n:
        return "Qwen"
    if "smolvlm" in n:
        return "SmolVLM"
    return "Other"


class OllamaAdapter(ModelAdapter):
    def __init__(self, model: str, vision: bool = False, timeout: float = 180):
        self.model = model
        self.timeout = timeout
        self.info = ModelInfo(f"ollama:{model}", model, family_of(model), "ollama", text=True, vision=vision)

    def _chat(self, messages: list[dict], max_tokens: int, temperature: float) -> str:
        # reasoning models need room to think before answering
        budget = max_tokens + (1024 if self.info.family == "DeepSeek" else 0)
        payload = {"model": self.model, "messages": messages, "stream": False,
                   "options": {"temperature": temperature, "num_predict": budget, "seed": 0}}
        try:
            r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=self.timeout)
        except requests.RequestException as exc:
            raise ModelError(f"Ollama unreachable at {OLLAMA_URL}: {exc}") from exc
        if r.status_code != 200:
            raise ModelError(f"Ollama error {r.status_code}: {r.text[:200]}")
        return strip_reasoning(r.json().get("message", {}).get("content", ""))

    def generate(self, prompt: str, max_tokens: int = 64, temperature: float = 0.0) -> str:
        return self._chat([{"role": "user", "content": prompt}], max_tokens, temperature)

    def describe_image(self, image: Image.Image, prompt: str, max_tokens: int = 32) -> str:
        if not self.info.vision:
            return super().describe_image(image, prompt, max_tokens)
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG")
        msg = {"role": "user", "content": prompt, "images": [base64.b64encode(buf.getvalue()).decode()]}
        return self._chat([msg], max_tokens, 0.0)

    def is_available(self) -> tuple[bool, str]:
        local = list_local_models()
        if not local:
            return False, f"Ollama not reachable at {OLLAMA_URL}"
        if self.model not in local:
            return False, f"Model not pulled. Run: ollama pull {self.model}"
        return True, "ok"
