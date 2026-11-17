"""Headless rendering of each dashboard page (Streamlit AppTest)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.mark.parametrize("idx", range(6))
def test_page_renders(idx):
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value(at.sidebar.radio[0].options[idx]).run()
    assert not at.exception, at.exception
    assert at.title[0].value


def test_run_coreference_with_simulator(monkeypatch, tmp_path):
    monkeypatch.setattr("bias_eval.results_store.RESULTS_DIR", tmp_path)
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value("Gender coreference (Winogender)").run()
    sb = at.selectbox(key="wino_model")
    sb.set_value(next(o for o in sb.options if "Biased simulator" in o)).run()
    next(b for b in at.button if "Run evaluation" in b.label).click().run()
    assert not at.exception, at.exception
    assert any("Bias score" in m.label for m in at.metric)
