"""Common interface for every model the framework can evaluate."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image


class ModelError(RuntimeError):
    """Raised when a model backend is unreachable or returns an error."""


@dataclass(frozen=True)
class ModelInfo:
    key: str           # unique id used in results, e.g. "ollama:qwen2.5:3b"
    display_name: str
    family: str        # Qwen, DeepSeek, Qwen2.5-VL, SmolVLM, Simulator
    backend: str       # ollama | huggingface | simulator
    text: bool
    vision: bool
    simulated: bool = False


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """DeepSeek-R1 style models emit <think>…</think>; evaluate only the final answer."""
    text = _THINK_RE.sub("", text or "")
    if "<think>" in text.lower():  # unterminated reasoning block (hit token limit)
        text = text.split("</think>")[-1] if "</think>" in text else ""
    return text.strip()


class ModelAdapter(ABC):
    info: ModelInfo

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 64, temperature: float = 0.0) -> str:
        """Return the model's text completion for a prompt."""

    def describe_image(self, image: Image.Image, prompt: str, max_tokens: int = 32) -> str:
        raise ModelError(f"{self.info.display_name} does not support images")

    def is_available(self) -> tuple[bool, str]:
        return True, "ok"
