"""Deterministic SIMULATED models for demos and tests.

These are NOT real language models. They produce answers with a known, configurable bias so the
pipeline, metrics and dashboard can be verified (a biased simulator must score as biased, an
unbiased one must not). Results from simulators are always labelled "Simulator" in the UI.
"""
from __future__ import annotations

import hashlib
import re

from PIL import Image

from bias_eval.models.base import ModelAdapter, ModelInfo

PRONOUN_RE = re.compile(r"\b(he|she|they|him|her|them|his|their)\b")
MALE = {"he", "him", "his"}
FEMALE = {"she", "her", "hers"}
FEMALE_CODED_JOBS = {"nurse", "secretary", "receptionist", "hairdresser", "librarian", "nanny", "dietitian", "teacher", "cashier"}
MALE_CODED_JOBS = {"engineer", "carpenter", "mechanic", "surgeon", "pilot", "plumber", "firefighter", "programmer", "chief", "electrician"}


def _u(text: str) -> float:
    """Stable pseudo-random number in [0,1) derived from the prompt."""
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


class SimulatedModel(ModelAdapter):
    def __init__(self, bias_strength: float = 0.0, name: str | None = None):
        if not 0 <= bias_strength <= 1:
            raise ValueError("bias_strength must be in [0, 1]")
        self.bias = bias_strength
        label = name or ("Biased simulator" if bias_strength > 0 else "Unbiased simulator")
        self.info = ModelInfo(f"sim:{bias_strength:.2f}", label, "Simulator", "simulator", text=True, vision=True, simulated=True)
        self.answer_key: dict[str, str] = {}  # prompt -> correct letter; set by the evaluator for simulators only

    def set_answer_key(self, key: dict[str, str]) -> None:
        self.answer_key = dict(key)

    def generate(self, prompt: str, max_tokens: int = 64, temperature: float = 0.0) -> str:
        p = prompt.lower()
        if "answer with a or b" in p:
            return self._coref(prompt)
        return self._continue(prompt)

    def _coref(self, prompt: str) -> str:
        # The simulator answers correctly ~85% of the time (answer key from the evaluator); with probability
        # `bias` it instead links he->male-coded and she->female-coded occupations.
        m = re.search(r"Sentence: (.*)\nQuestion: who does \"(\w+)\" refer to\?\s*A\) the (.+?)\s*B\) the (.+?)\n", prompt, re.S)
        if not m:
            return "A"
        sentence, pronoun, occ, other = m.groups()
        pronoun = pronoun.lower()
        occ_l = occ.lower()
        gold = self.answer_key.get(prompt) or ("A" if _u(sentence) < 0.5 else "B")
        stereo_match = (pronoun in MALE and any(j in occ_l for j in MALE_CODED_JOBS)) or \
                       (pronoun in FEMALE and any(j in occ_l for j in FEMALE_CODED_JOBS))
        stereo_mismatch = (pronoun in FEMALE and any(j in occ_l for j in MALE_CODED_JOBS)) or \
                          (pronoun in MALE and any(j in occ_l for j in FEMALE_CODED_JOBS))
        r = _u(prompt)
        if stereo_match and r < self.bias:
            return "A"
        if stereo_mismatch and r < self.bias:
            return "B"
        # random mistakes depend only on the template (not the pronoun), so an unbiased simulator never
        # changes its answer when just the pronoun changes
        noise = _u(occ_l + "|" + other.lower() + "|" + PRONOUN_RE.sub("_", sentence.lower()))
        return gold if noise < 0.85 else ("B" if gold == "A" else "A")

    def _continue(self, prompt: str) -> str:
        p = prompt.lower()
        positive = "was widely admired for excellent, inspiring and successful work."
        neutral = "worked at a company and lived in a city."
        negative = "was criticised for poor, careless and disappointing results."
        r = _u(prompt)
        disfavoured = any(w in p for w in (" woman", " she ", "female", " elderly", " old ", "older"))
        if disfavoured and r < self.bias:
            return negative
        return positive if r < 0.5 else neutral

    def describe_image(self, image: Image.Image, prompt: str, max_tokens: int = 32) -> str:
        # images in tests encode the true label in their colour channel: red=male, blue=female
        r, g, b = image.convert("RGB").resize((1, 1)).getpixel((0, 0))
        truth = "male" if r > b else "female"
        darker = (r + g + b) / 3 < 110  # stands in for a demographic attribute in tests
        if darker and _u(str(image.size) + str((r, g, b))) < self.bias:
            return "female" if truth == "male" else "male"
        return truth
