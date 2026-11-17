# Bias Detection & Fairness Evaluation Framework

A framework and Streamlit dashboard for measuring **gender and demographic bias** in language and vision-language models (**Qwen**, **DeepSeek**, **Qwen2.5-VL**, **SmolVLM**) and **fairness** of any classifier's predictions. It runs models locally (Ollama or Hugging Face), scores bias with clearly defined metrics and confidence intervals, and compares models side by side.

**Tech stack:** Python 3.11 · Streamlit · Plotly · pandas · Ollama (Qwen2.5, DeepSeek-R1) · Hugging Face transformers (Qwen2.5-VL, SmolVLM; optional) · VADER sentiment · pytest

![Classification fairness](docs/screenshots/classification.png)

## Features

- **AI bias detection** across 4 evaluations:
  - **Gender coreference (Winogender):** does the answer change when only the pronoun changes?
  - **Generation bias (BOLD):** sentiment of continuations for prompts that differ only by gender, age or nationality.
  - **Vision bias (FairFace):** does a VLM's perceived-gender accuracy differ across race or age groups?
  - **Classification fairness:** demographic parity, disparate impact (four-fifths rule), equal opportunity and equalized odds for any uploaded predictions.
- **Bias scoring (0–100)** with a precise definition per evaluation, bootstrap 95% CIs and permutation-test p-values.
- **Model comparison:** every run is saved to `results/` and compared on the dashboard.
- **Upload your own test data:** Winogender TSV, BOLD JSON, prompt CSV (`group,prompt`), or prediction CSV.
- **Dataset support:** loaders for the official **Winogender**, **BOLD** and **FairFace** formats. The datasets are not redistributed. Small *synthetic* samples in the same formats are bundled.
- **Simulators** with known bias, used to check that the metrics behave correctly (always labelled as simulators).

## Models

| Family | How to enable | Used for |
|---|---|---|
| Qwen (`qwen2.5:3b`) | `ollama pull qwen2.5:3b` | Text evaluations |
| DeepSeek (`deepseek-r1:1.5b`) | `ollama pull deepseek-r1:1.5b` | Text evaluations (reasoning is stripped before scoring) |
| Qwen2.5-VL | `ollama pull qwen2.5vl:3b` **or** `pip install -r requirements-vision.txt` (`Qwen/Qwen2.5-VL-3B-Instruct`) | Vision |
| SmolVLM | `pip install -r requirements-vision.txt` (`HuggingFaceTB/SmolVLM-256M-Instruct`) | Vision |

The dashboard detects available models automatically.

## Results so far (real models, bundled synthetic samples)

Measured on the author's laptop (CPU, Ollama 0.40, greedy decoding). Raw outputs are in `results/`.

| Model | Evaluation | Bias score | Key numbers |
|---|---|---|---|
| qwen2.5:3b | Coreference | **30.0** | Answer flips between *he* and *she* on 30% of templates (95% CI 13–47%). Accuracy: he 76.7%, she 80.0%, they 90.0% |
| qwen2.5:3b | Generation | **6.3** | Largest gender gap 0.195 (woman 0.267 vs man 0.072), *not significant* (p = 0.26). Age gap 0.037 (p = 0.81). Nationality gap 0.147 (p = 0.28) |

> With 30 templates and 12 prompts per group these are **indicative only**, as the wide CIs and p-values show. Download the official datasets for reliable conclusions. DeepSeek-R1 results are added to `results/` when its run completes. Vision results need FairFace images and a vision model, so none are reported yet.

## Datasets

| Dataset | Get it | Use it |
|---|---|---|
| Winogender | `templates.tsv` from github.com/rudinger/winogender-schemas | Upload on the Winogender page, or `--winogender path` |
| BOLD | `prompts/*_prompt.json` from github.com/amazon-science/bold | Upload on the BOLD page (tick "official file"), or `--bold path --bold-domain gender` |
| FairFace | Images + `fairface_label_val.csv` from github.com/joojs/fairface | Put them in `data/external/fairface/` (git-ignored) |

Bundled synthetic samples:
- `data/winogender_style_templates.tsv`: 30 templates × 3 pronouns
- `data/bold_style_prompts.json`: 12 templates × 9 groups
- `data/sample_loan_predictions.csv`: 2,000 rows from a deliberately biased loan model

## Project structure

```
bias-fairness-evaluation/
├── app.py                       # Streamlit dashboard (6 pages)
├── bias_eval/
│   ├── models/                  # base.py, ollama_adapter.py, hf_vision.py, simulator.py, registry.py
│   ├── datasets/loaders.py      # Winogender / BOLD / FairFace / CSV loaders
│   ├── evaluations/             # coreference.py, generation.py, vision.py
│   ├── metrics/                 # fairness.py (DP, DI, EO, EOdds, CI, permutation), sentiment.py
│   └── results_store.py
├── scripts/run_eval.py          # command-line evaluation runner
├── scripts/make_sample_predictions.py
├── data/                        # synthetic samples (official datasets go in data/external/)
├── results/                     # saved runs (JSON summary + CSV raw outputs)
├── docs/ARCHITECTURE.md, docs/screenshots/
├── tests/                       # 36 tests
└── requirements.txt, requirements-vision.txt, .env.example
```

## Installation

```bash
git clone https://github.com/manisha2001-20/bias-fairness-evaluation.git
cd bias-fairness-evaluation
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# install Ollama from https://ollama.com, then:
ollama pull qwen2.5:3b
ollama pull deepseek-r1:1.5b
```

## Running

```bash
streamlit run app.py                                                # dashboard at http://localhost:8501
python scripts/run_eval.py --model ollama:qwen2.5:3b --eval all     # CLI: saves to results/
python scripts/run_eval.py --model ollama:deepseek-r1:1.5b --eval coreference
python scripts/run_eval.py --model sim:0.60 --eval all              # simulator (no LLM needed)
```

To open a specific page directly, add `?page=0` through `?page=5` to the URL.

## Tests

```bash
pytest
```

36 tests cover:
- dataset loaders and their error cases;
- answer parsing (including DeepSeek `<think>` stripping);
- fairness metrics checked against hand-computed values;
- statistics helpers;
- **metric validity**: the unbiased simulator scores 0 and biased simulators are detected, for all three model evaluations;
- fast failure when a backend is down;
- result storage;
- headless rendering of every dashboard page.

The tests do not need Ollama.

## Screenshots

| Overview | Generation bias | Model comparison |
|---|---|---|
| ![](docs/screenshots/overview.png) | ![](docs/screenshots/generation.png) | ![](docs/screenshots/comparison.png) |

To retake them: run `streamlit run app.py`, run an evaluation, and use your browser's screenshot tool on each page.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the metric definitions and design decisions.
