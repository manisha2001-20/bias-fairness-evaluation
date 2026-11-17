"""Group-fairness metrics for classification outputs + statistics helpers (bootstrap CI, permutation test)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def bootstrap_ci(values: np.ndarray, stat=np.mean, n: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = [stat(rng.choice(values, size=len(values), replace=True)) for _ in range(n)]
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def permutation_pvalue(a: np.ndarray, b: np.ndarray, n: int = 5000, seed: int = 0) -> float:
    """Two-sided permutation test for a difference in means."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    observed = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    count = 0
    for _ in range(n):
        rng.shuffle(pooled)
        if abs(pooled[: len(a)].mean() - pooled[len(a):].mean()) >= observed - 1e-12:
            count += 1
    return (count + 1) / (n + 1)


@dataclass
class ClassificationFairness:
    per_group: pd.DataFrame
    demographic_parity_difference: float
    disparate_impact_ratio: float
    equal_opportunity_difference: float | None
    equalized_odds_difference: float | None
    accuracy_difference: float | None
    privileged_group: str
    bias_score: float  # 0 (fair) … 100 (maximally unfair)

    def passes_four_fifths(self) -> bool:
        return self.disparate_impact_ratio >= 0.8


def classification_fairness(df: pd.DataFrame, group_col: str = "group", pred_col: str = "y_pred",
                            true_col: str | None = "y_true", positive=1) -> ClassificationFairness:
    if group_col not in df or pred_col not in df:
        raise ValueError(f"Data must contain '{group_col}' and '{pred_col}' columns")
    d = df.dropna(subset=[group_col, pred_col]).copy()
    if d[group_col].nunique() < 2:
        raise ValueError("Need at least two groups")
    d["_pred"] = (d[pred_col] == positive).astype(int)
    has_truth = true_col is not None and true_col in d.columns
    if has_truth:
        d["_true"] = (d[true_col] == positive).astype(int)

    rows = []
    for g, sub in d.groupby(group_col):
        row = {"group": str(g), "n": len(sub), "selection_rate": sub["_pred"].mean()}
        if has_truth:
            pos, neg = sub[sub["_true"] == 1], sub[sub["_true"] == 0]
            row["tpr"] = pos["_pred"].mean() if len(pos) else np.nan
            row["fpr"] = neg["_pred"].mean() if len(neg) else np.nan
            row["accuracy"] = (sub["_pred"] == sub["_true"]).mean()
            row["precision"] = sub.loc[sub["_pred"] == 1, "_true"].mean() if sub["_pred"].sum() else np.nan
        rows.append(row)
    pg = pd.DataFrame(rows).set_index("group")

    sr = pg["selection_rate"]
    privileged = str(sr.idxmax())
    dpd = float(sr.max() - sr.min())
    dir_ = float(sr.min() / sr.max()) if sr.max() > 0 else 1.0
    eod = eqodds = accd = None
    if has_truth:
        eod = float(pg["tpr"].max() - pg["tpr"].min())
        eqodds = float(max(eod, pg["fpr"].max() - pg["fpr"].min()))
        accd = float(pg["accuracy"].max() - pg["accuracy"].min())
    components = [dpd, 1 - dir_] + ([eqodds] if eqodds is not None else [])
    return ClassificationFairness(pg, dpd, dir_, eod, eqodds, accd, privileged, round(100 * float(np.mean(components)), 1))
