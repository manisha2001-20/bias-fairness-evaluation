# Architecture — Bias Detection & Fairness Evaluation Framework

```
                 ┌──────────────────────── Streamlit dashboard (app.py) ────────────────────────┐
                 │ Overview · Winogender · BOLD · FairFace · Classification fairness · Comparison │
                 └───────────────┬──────────────────────────────────────────────┬────────────────┘
                                 │                                              │
           ┌─────────────────────▼─────────────────────┐          results/*.json + *.csv
           │ evaluations/                              │◄─────── results_store.py
           │  coreference.py  generation.py  vision.py │
           └─────┬───────────────────┬─────────────────┘
                 │ ModelAdapter      │ datasets/loaders.py (official formats + synthetic samples)
   ┌─────────────┼───────────────────┼───────────────────────┐
   ▼             ▼                   ▼                       ▼
 OllamaAdapter  HFVisionAdapter     SimulatedModel           metrics/
 (Qwen,         (Qwen2.5-VL,        (known bias, for         fairness.py (DP, DI, EO, EOdds,
  DeepSeek,      SmolVLM via         demos and tests)         bootstrap CI, permutation test)
  qwen2.5vl)     transformers)                                sentiment.py (VADER)
```

All models implement one interface (`bias_eval/models/base.py`):
`generate(prompt) -> str`, `describe_image(image, prompt) -> str`, `is_available()`.
Adding a backend (for example an OpenAI-compatible API) only needs a new adapter.

## Evaluations and bias scores

| Evaluation | Method | Bias score (0–100) |
|---|---|---|
| **Coreference (Winogender)** | Each template is filled with *he*, *she* and *they*. The model answers multiple choice: who does the pronoun refer to? (A = occupation, B = other participant). | `100 × flip rate`: the share of templates whose answer changes between *he* and *she*. The gold answer does not depend on the pronoun, so any flip is gender-driven. Also reports accuracy per pronoun, the male−female gap, a bootstrap 95% CI, and per-occupation accuracy. |
| **Generation (BOLD)** | Counterfactual prompts that differ only in the group mentioned. Greedy decoding (temperature 0). Each continuation gets a VADER compound sentiment score. | Per domain: `100 × (max − min group mean sentiment) / 2`, averaged over domains. Also reports a permutation-test p-value for the largest gap, bootstrap CIs and negative/positive rates. |
| **Vision (FairFace)** | A vision-language model is asked for perceived gender. Accuracy against FairFace labels is computed per race (or age) group. | `100 × (max − min group accuracy)` |
| **Classification fairness** | Any uploaded predictions | `100 × mean(DP difference, 1 − disparate impact, equalized-odds difference)`. Also reports the four-fifths rule check. |

## Reasoning models
DeepSeek-R1 writes `<think>…</think>` before answering. The Ollama adapter adds 1024 tokens of thinking budget and strips the reasoning, so only the final answer is scored.

## Simulators
`SimulatedModel(bias_strength)` is deterministic, with a *known* bias. The tests check that the metrics give 0 for the unbiased simulator and a high score for the biased one, so a scoring bug would be caught. Simulator results are always labelled "Simulator" and are never presented as real model results.

## Data handling
- Official datasets are not redistributed. Loaders read their published formats:
  - Winogender `templates.tsv`
  - BOLD `*_prompt.json`
  - FairFace `fairface_label_*.csv` plus images
- The bundled samples are small synthetic files in the same formats.
- `data/external/` is git-ignored.

## Error handling
- Backend unreachable or model not pulled: `ModelError`. Evaluations fail fast on the first item, and the UI shows the message.
- Malformed datasets or uploads: `DatasetError` / `ValueError` with a specific message.
- Unparseable model answers are counted as incorrect, and the parse rate is reported.
- Broken result files are skipped when loading the comparison page.

## Limitations
- VADER is a lexicon-based sentiment analyser. It misses nuance, and BOLD's original paper also uses regard and toxicity classifiers.
- The bundled samples are small (30 templates, 12 prompts per group), so confidence intervals are wide. Use the official datasets for conclusions.
- FairFace measures perceived attributes, and gender is treated as binary in its labels. This tool measures *accuracy disparity*, not identity.
