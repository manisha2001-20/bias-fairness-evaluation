"""Creates data/sample_loan_predictions.csv: SYNTHETIC loan decisions from a deliberately biased model.

Columns: applicant_id, gender, age_group, income, y_true (would repay), y_pred (approved), score.
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(11)
n = 2000
gender = rng.choice(["female", "male", "non-binary"], n, p=[0.47, 0.49, 0.04])
age = rng.choice(["18-25", "26-40", "41-60", "60+"], n, p=[0.2, 0.4, 0.3, 0.1])
income = rng.lognormal(10.6, 0.5, n).round(-2)
repay_p = 1 / (1 + np.exp(-(np.log(income) - 10.4) * 2.2))
y_true = (rng.random(n) < repay_p).astype(int)
# the simulated model under-approves women and applicants over 60 at the same true repayment ability
penalty = np.where(gender == "female", 0.55, 0.0) + np.where(age == "60+", 0.5, 0.0)
score = 1 / (1 + np.exp(-((np.log(income) - 10.4) * 2.2 + rng.normal(0, 0.6, n) - penalty)))
y_pred = (score >= 0.5).astype(int)
df = pd.DataFrame({"applicant_id": [f"A{i:05d}" for i in range(n)], "gender": gender, "age_group": age,
                   "income": income, "y_true": y_true, "y_pred": y_pred, "score": score.round(4)})
out = Path(__file__).resolve().parent.parent / "data" / "sample_loan_predictions.csv"
df.to_csv(out, index=False)
print("wrote", out, df.groupby("gender")["y_pred"].mean().round(3).to_dict())
