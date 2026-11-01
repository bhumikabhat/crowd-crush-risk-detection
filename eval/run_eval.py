"""Compare the flow-aware index with the density-only baseline on held-out simulations.

Train: seeds 0,1 (fit weights, pick thresholds). Test: seeds 2,3 (report).
Run from the project root:  python -m eval.run_eval
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval.lead_time import evaluate, sweep
from risk.score import DensityBaseline, FlowAwareRisk

FPS, SMOOTH_S, WINDOW_S, HOLD_S = 10, 2.0, 15.0, 3.0
FA_TARGET = 10.0                              # false alarms per hour allowed when picking thresholds
TRAIN_SEEDS, TEST_SEEDS = {0, 1}, {2, 3}
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.025), 3)

df = pd.read_csv("data/features.csv")
for c in ["density", "entropy", "slowdown"]:                       # trailing 2 s mean (causal)
    df[c] = df.groupby("scenario")[c].transform(lambda s: s.rolling(int(SMOOTH_S * FPS), min_periods=1).mean())
df["y"] = ((df["t"] >= df["t_event"] - WINDOW_S) & (df["t"] < df["t_event"])).astype(int)

train, test = df[df["seed"].isin(TRAIN_SEEDS)], df[df["seed"].isin(TEST_SEEDS)]
fit_rows = train[~(train["t"] >= train["t_event"])]               # drop frames after the event began

baseline, risk = DensityBaseline(), FlowAwareRisk().fit(fit_rows, fit_rows["y"].to_numpy())
print("learned weights:", risk.weights)
for m in (baseline, risk):
    df[m.name] = m.score(df)
train, test = df[df["seed"].isin(TRAIN_SEEDS)], df[df["seed"].isin(TEST_SEEDS)]
groups = lambda d: [g for _, g in d.groupby("scenario")]
tr_g, te_g = groups(train), groups(test)
print(f"train: {len(tr_g)} runs, {sum(g.t_event.notna().iloc[0] for g in tr_g)} with danger events;"
      f" test: {len(te_g)} runs, {sum(g.t_event.notna().iloc[0] for g in te_g)} with danger events")

rows, curves = [], {}
for m in (baseline, risk):
    tr = pd.DataFrame(sweep(tr_g, m.name, THRESHOLDS, hold_s=HOLD_S, window_s=WINDOW_S))
    ok = tr[tr["false_alarms_per_hour"] <= FA_TARGET]
    # best operating point on train: most events detected, then longest mean lead
    best = ok.sort_values(["detected", "mean_lead_s"], ascending=False).iloc[0]
    res = evaluate(te_g, m.name, best["threshold"], hold_s=HOLD_S, window_s=WINDOW_S)
    rows.append({"method": m.name, **res})
    curves[m.name] = pd.DataFrame(sweep(te_g, m.name, THRESHOLDS, hold_s=HOLD_S, window_s=WINDOW_S))

out = pd.DataFrame(rows)[["method", "threshold", "events", "detected", "missed",
                          "mean_lead_s", "false_alarms", "false_alarms_per_hour", "hours"]]
print(f"\nTest results (threshold chosen on train, FA/h <= {FA_TARGET}):")
print(out.round(2).to_string(index=False))
out.round(3).to_csv("data/results.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for name, c in curves.items():
    ax[0].plot(c["false_alarms_per_hour"], c["mean_lead_s"], marker="o", ms=3, label=name)
    ax[1].plot(c["false_alarms_per_hour"], c["missed"], marker="o", ms=3, label=name)
ax[0].set(xlabel="false alarms per hour", ylabel="mean lead time (s)", title="Lead time vs false alarms (test)")
ax[1].set(xlabel="false alarms per hour", ylabel="missed events", title="Missed events (test)")
for a in ax:
    a.legend(); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig("data/operating_curves.png", dpi=150)
print("saved data/results.csv and data/operating_curves.png")
