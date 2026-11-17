import io
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from bias_eval.datasets.loaders import DatasetError, load_bold, load_fairface, load_prompt_csv, load_winogender
from bias_eval.evaluations import coreference, generation, vision
from bias_eval.metrics.fairness import bootstrap_ci, classification_fairness, permutation_pvalue
from bias_eval.metrics.sentiment import label, sentiment
from bias_eval.models.base import ModelError, strip_reasoning
from bias_eval.models.registry import build
from bias_eval.models.simulator import SimulatedModel
from bias_eval.results_store import comparison_table, load_runs, save_run


# ------------------------------------------------------------------ datasets
def test_winogender_sample_expands_three_pronouns():
    items = load_winogender()
    assert len(items) == 90
    she = [i for i in items if i.gender == "female"][0]
    assert " she " in f" {she.sentence} " and "$" not in she.sentence
    assert {i.gold for i in items} == {"A", "B"}


def test_winogender_singular_they_agreement():
    items = [i for i in load_winogender() if i.gender == "neutral"]
    assert not any("they was" in i.sentence for i in items)


def test_winogender_rejects_bad_file():
    with pytest.raises(DatasetError):
        load_winogender(io.StringIO("a\tb\tc\td\nx\ty\t7\tThe $OCCUPATION saw $NOM_PRONOUN."))


def test_bold_sample_and_official_format():
    df = load_bold(max_per_group=3)
    assert set(df["domain"]) == {"gender", "age", "nationality"}
    assert df.groupby("group").size().max() == 3
    official = {"American_actors": {"Tom_Hanks": ["Tom Hanks is an American actor"]},
                "American_actresses": {"Meryl_Streep": ["Meryl Streep is an American actress"]}}
    o = load_bold(official, domain="gender")
    assert list(o["group"]) == ["American_actors", "American_actresses"] and set(o["domain"]) == {"gender"}


def test_prompt_csv_validation():
    ok = load_prompt_csv(io.StringIO("group,prompt\na,The a person\nb,The b person\n"))
    assert len(ok) == 2 and set(ok["domain"]) == {"custom"}
    with pytest.raises(DatasetError):
        load_prompt_csv(io.StringIO("group,prompt\na,x\n"))
    with pytest.raises(DatasetError):
        load_prompt_csv(io.StringIO("foo,bar\n1,2\n"))


def test_fairface_loader(tmp_path):
    (tmp_path / "val").mkdir()
    rows = []
    for i, race in enumerate(["East Asian", "Black", "White", "Indian"] * 2):
        Image.new("RGB", (8, 8), (200, 50, 50)).save(tmp_path / "val" / f"{i}.jpg")
        rows.append({"file": f"val/{i}.jpg", "age": "20-29", "gender": "Male", "race": race})
    rows.append({"file": "val/missing.jpg", "age": "20-29", "gender": "Female", "race": "White"})
    pd.DataFrame(rows).to_csv(tmp_path / "labels.csv", index=False)
    df = load_fairface(tmp_path / "labels.csv", per_group=1)
    assert len(df) == 4 and set(df["gender"]) == {"male"}
    with pytest.raises(DatasetError):
        load_fairface(tmp_path / "nope.csv")


# ------------------------------------------------------------------ parsing
@pytest.mark.parametrize("text,expected", [("A", "A"), ("B)", "B"), ("The answer is B.", "B"), ("the nurse", "A"),
                                           ("the patient", "B"), ("I cannot tell", None)])
def test_parse_choice(text, expected):
    assert coreference.parse_choice(text, "nurse", "patient") == expected


def test_strip_reasoning():
    assert strip_reasoning("<think>hmm he or she</think>\nA") == "A"
    assert strip_reasoning("<think>never finished") == ""


@pytest.mark.parametrize("text,expected", [("Female.", "female"), ("male", "male"), ("a man", "male"), ("unclear", None),
                                           ("male or female", None)])
def test_parse_gender(text, expected):
    assert vision.parse_gender(text) == expected


# ------------------------------------------------------------------ evaluations (simulators have KNOWN bias)
def test_coreference_detects_simulated_bias():
    items = load_winogender()
    fair = coreference.summarise(coreference.run(SimulatedModel(0.0), items))
    biased = coreference.summarise(coreference.run(SimulatedModel(0.8), items))
    assert fair["bias_score"] == 0.0          # answers never depend on the pronoun
    assert biased["bias_score"] > 25
    assert fair["parse_rate"] == 1.0


def test_generation_detects_simulated_bias():
    prompts = load_bold(max_per_group=12)
    fair = generation.summarise(generation.run(SimulatedModel(0.0), prompts))
    biased = generation.summarise(generation.run(SimulatedModel(0.9), prompts))
    assert fair["domains"]["gender"]["max_gap"] < 0.1
    assert biased["domains"]["gender"]["max_gap"] > 0.5
    assert biased["domains"]["gender"]["least_positive"] == "woman"
    assert biased["domains"]["gender"]["p_value_max_gap"] < 0.05


def test_vision_detects_simulated_bias():
    rows = []
    for i in range(40):
        male = i % 2 == 0
        dark = i % 4 >= 2
        base = 60 if dark else 200
        colour = (base + 40, base - 20, base - 40) if male else (base - 40, base - 20, base + 40)
        rows.append({"image": Image.new("RGB", (16 + i, 16), colour), "gender": "male" if male else "female",
                     "race": "group_dark" if dark else "group_light"})
    items = pd.DataFrame(rows)
    fair = vision.summarise(vision.run(SimulatedModel(0.0), items))
    biased = vision.summarise(vision.run(SimulatedModel(1.0), items))
    assert fair["bias_score"] == 0.0
    assert biased["groups"]["group_dark"]["accuracy"] < biased["groups"]["group_light"]["accuracy"]


class _Down(SimulatedModel):
    def generate(self, *a, **k):
        raise ModelError("backend down")


def test_backend_failure_raises_fast():
    with pytest.raises(ModelError):
        coreference.run(_Down(), load_winogender()[:3])
    with pytest.raises(ModelError):
        generation.run(_Down(), load_bold(max_per_group=1))


def test_text_only_model_rejected_for_vision():
    from bias_eval.models.ollama_adapter import OllamaAdapter
    with pytest.raises(ModelError):
        vision.run(OllamaAdapter("qwen2.5:3b"), pd.DataFrame([{"image": Image.new("RGB", (4, 4)), "gender": "male", "race": "x"}]))


# ------------------------------------------------------------------ metrics
def test_classification_fairness_hand_computed():
    df = pd.DataFrame({"group": ["a"] * 4 + ["b"] * 4, "y_true": [1, 1, 0, 0, 1, 1, 0, 0], "y_pred": [1, 1, 1, 0, 1, 0, 0, 0]})
    f = classification_fairness(df)
    assert f.demographic_parity_difference == pytest.approx(0.75 - 0.25)
    assert f.disparate_impact_ratio == pytest.approx(0.25 / 0.75)
    assert f.equal_opportunity_difference == pytest.approx(0.5)
    assert f.equalized_odds_difference == pytest.approx(0.5)
    assert not f.passes_four_fifths() and f.privileged_group == "a"


def test_classification_fairness_perfectly_fair_and_errors():
    df = pd.DataFrame({"group": ["a", "a", "b", "b"], "y_pred": [1, 0, 1, 0]})
    f = classification_fairness(df, true_col=None)
    assert f.bias_score == 0 and f.equal_opportunity_difference is None
    with pytest.raises(ValueError):
        classification_fairness(df.assign(group="a"))
    with pytest.raises(ValueError):
        classification_fairness(df, pred_col="missing")


def test_stats_helpers():
    lo, hi = bootstrap_ci(np.array([0, 1] * 50))
    assert lo < 0.5 < hi
    assert permutation_pvalue(np.ones(30), np.zeros(30), n=500) < 0.01
    assert permutation_pvalue(np.array([0, 1] * 15), np.array([1, 0] * 15), n=500) > 0.5


def test_sentiment():
    assert sentiment("excellent and inspiring work") > 0.5
    assert sentiment("terrible, careless failure") < -0.5
    assert label(0.0) == "neutral"


# ------------------------------------------------------------------ storage + registry
def test_results_roundtrip(tmp_path):
    m = SimulatedModel(0.5)
    df = coreference.run(m, load_winogender()[:6])
    save_run(m.info, "coreference", coreference.summarise(df), df, "test", results_dir=tmp_path)
    runs = load_runs(tmp_path)
    assert len(runs) == 1 and runs[0]["model"]["simulated"] is True
    t = comparison_table(runs)
    assert t.loc[0, "evaluation"] == "coreference"
    (tmp_path / "broken.json").write_text("{not json")
    assert len(load_runs(tmp_path)) == 1


def test_registry_build():
    assert build("sim:0.30").info.simulated
    assert build("ollama:qwen2.5:3b").info.family == "Qwen"
    assert build("ollama:deepseek-r1:1.5b").info.family == "DeepSeek"
    with pytest.raises(ValueError):
        build("unknown:x")
