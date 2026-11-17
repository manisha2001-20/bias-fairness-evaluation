"""Discovers which models are available and builds adapters by key."""
from __future__ import annotations

from bias_eval.models.base import ModelAdapter, ModelInfo
from bias_eval.models.hf_vision import HF_VISION_MODELS, HFVisionAdapter, transformers_available
from bias_eval.models.ollama_adapter import OllamaAdapter, family_of, list_local_models
from bias_eval.models.simulator import SimulatedModel

# Ollama tags that support images
OLLAMA_VISION_HINTS = ("vl", "vision", "llava", "smolvlm", "moondream")
RECOMMENDED_OLLAMA = ["qwen2.5:3b", "deepseek-r1:1.5b", "qwen2.5vl:3b"]


def available_models(include_simulators: bool = True) -> list[ModelInfo]:
    infos: list[ModelInfo] = []
    for name in list_local_models():
        vision = any(h in name.lower() for h in OLLAMA_VISION_HINTS)
        infos.append(OllamaAdapter(name, vision=vision).info)
    if transformers_available():
        infos += [HFVisionAdapter(m).info for m in HF_VISION_MODELS]
    if include_simulators:
        infos += [SimulatedModel(0.0).info, SimulatedModel(0.6).info]
    return infos


def build(key: str) -> ModelAdapter:
    backend, _, rest = key.partition(":")
    if backend == "ollama":
        return OllamaAdapter(rest, vision=any(h in rest.lower() for h in OLLAMA_VISION_HINTS))
    if backend == "hf":
        return HFVisionAdapter(rest)
    if backend == "sim":
        return SimulatedModel(float(rest))
    raise ValueError(f"Unknown model key: {key}")


__all__ = ["available_models", "build", "family_of", "RECOMMENDED_OLLAMA"]
