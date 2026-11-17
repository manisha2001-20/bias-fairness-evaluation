"""Hugging Face transformers backend for Qwen2.5-VL and SmolVLM (optional; install requirements-vision.txt).

Models are downloaded on first use (several GB). A CUDA GPU is strongly recommended; CPU works but is slow.
"""
from __future__ import annotations

import importlib.util

from PIL import Image

from bias_eval.models.base import ModelAdapter, ModelError, ModelInfo

HF_VISION_MODELS = {
    "Qwen/Qwen2.5-VL-3B-Instruct": "Qwen2.5-VL",
    "HuggingFaceTB/SmolVLM-256M-Instruct": "SmolVLM",
    "HuggingFaceTB/SmolVLM-Instruct": "SmolVLM",
}


def transformers_available() -> bool:
    return importlib.util.find_spec("transformers") is not None and importlib.util.find_spec("torch") is not None


class HFVisionAdapter(ModelAdapter):
    def __init__(self, model_id: str):
        if model_id not in HF_VISION_MODELS:
            raise ValueError(f"Unsupported model {model_id}")
        self.model_id = model_id
        self.info = ModelInfo(f"hf:{model_id}", model_id.split("/")[-1], HF_VISION_MODELS[model_id], "huggingface",
                              text=True, vision=True)
        self._model = self._processor = None

    def _load(self):
        if self._model is not None:
            return
        if not transformers_available():
            raise ModelError("transformers/torch not installed. Run: pip install -r requirements-vision.txt")
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        try:
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForVision2Seq.from_pretrained(self.model_id, torch_dtype=dtype).to(device)
        except Exception as exc:  # network, disk, incompatible version …
            raise ModelError(f"Could not load {self.model_id}: {exc}") from exc
        self._device = device

    def _run(self, content: list[dict], images: list[Image.Image] | None, max_tokens: int) -> str:
        self._load()
        import torch

        messages = [{"role": "user", "content": content}]
        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=[prompt], images=images, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return self._processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

    def generate(self, prompt: str, max_tokens: int = 64, temperature: float = 0.0) -> str:
        return self._run([{"type": "text", "text": prompt}], None, max_tokens)

    def describe_image(self, image: Image.Image, prompt: str, max_tokens: int = 32) -> str:
        return self._run([{"type": "image"}, {"type": "text", "text": prompt}], [image.convert("RGB")], max_tokens)

    def is_available(self) -> tuple[bool, str]:
        if not transformers_available():
            return False, "Install optional extras: pip install -r requirements-vision.txt"
        return True, "ok (downloads weights on first use)"
