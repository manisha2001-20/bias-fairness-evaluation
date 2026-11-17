"""Saves evaluation runs as JSON (+ CSV of raw outputs) for later comparison."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()


def save_run(model_info, evaluation: str, summary: dict, details: pd.DataFrame, dataset: str,
             results_dir: Path | None = None) -> Path:
    results_dir = Path(results_dir or RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = results_dir / f"{stamp}_{evaluation}_{_slug(model_info.key)}"
    record = {"timestamp": datetime.now().isoformat(timespec="seconds"), "evaluation": evaluation, "dataset": dataset,
              "model": {"key": model_info.key, "name": model_info.display_name, "family": model_info.family,
                        "backend": model_info.backend, "simulated": model_info.simulated},
              "summary": summary}
    base.with_suffix(".json").write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")
    details.to_csv(base.with_suffix(".csv"), index=False)
    return base.with_suffix(".json")


def load_runs(results_dir: Path | None = None) -> list[dict]:
    results_dir = Path(results_dir or RESULTS_DIR)
    runs = []
    for p in sorted(results_dir.glob("*.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
            r["_file"] = p.name
            runs.append(r)
        except (json.JSONDecodeError, OSError):
            continue
    return runs


def comparison_table(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        s = r["summary"]
        rows.append({"timestamp": r["timestamp"], "evaluation": r["evaluation"], "dataset": r.get("dataset", ""),
                     "model": r["model"]["name"], "family": r["model"]["family"], "simulated": r["model"]["simulated"],
                     "bias_score": s.get("bias_score"), "n_items": s.get("n_items"),
                     "accuracy": s.get("accuracy_overall"), "file": r["_file"]})
    return pd.DataFrame(rows)
