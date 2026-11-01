"""Which signals earn their place? Same protocol as run_eval, with feature subsets.
Run from the project root:  python -m eval.ablation
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from eval.lead_time import evaluate, sweep
from risk.score import scaled

FPS, SMOOTH_S, WINDOW_S, HOLD_S, FA_TARGET = 10, 2.0, 15.0, 3.0, 10.0
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.025), 3)
SUBSETS = {"D": [0], "D+S": [0, 2], "D+H": [0, 1], "H+S": [1, 2], "D+H+S": [0, 1, 2]}

df = pd.read_csv("data/features.csv")
for c in ["density", "entropy", "slowdown"]:
    df[c] = df.groupby("scenario")[c].transform(lambda s: s.rolling(int(SMOOTH_S * FPS), min_periods=1).mean())
df["y"] = ((df["t"] >= df["t_event"] - WINDOW_S) & (df["t"] < df["t_event"])).astype(int)
train, test = df[df["seed"].isin({0, 1})], df[df["seed"].isin({2, 3})]
fit = train[~(train["t"] >= train["t_event"])]
tr_g = [g for _, g in train.groupby("scenario")]
te_g = [g for _, g in test.groupby("scenario")]

rows = []
for name, cols in SUBSETS.items():
    m = LogisticRegression(C=10.0, class_weight="balanced", max_iter=1000).fit(scaled(fit)[:, cols], fit["y"])
    for d in (train, test):
        d.loc[:, "s"] = m.predict_proba(scaled(d)[:, cols])[:, 1]
    tr_g = [g for _, g in train.groupby("scenario")]
    te_g = [g for _, g in test.groupby("scenario")]
    tr = pd.DataFrame(sweep(tr_g, "s", THRESHOLDS, hold_s=HOLD_S, window_s=WINDOW_S))
    ok = tr[tr["false_alarms_per_hour"] <= FA_TARGET].sort_values(["detected", "mean_lead_s"], ascending=False)
    res = evaluate(te_g, "s", ok.iloc[0]["threshold"], hold_s=HOLD_S, window_s=WINDOW_S)
    rows.append({"signals": name, **{k: res[k] for k in ["detected", "missed", "mean_lead_s", "false_alarms_per_hour"]}})
print(pd.DataFrame(rows).round(2).to_string(index=False))
