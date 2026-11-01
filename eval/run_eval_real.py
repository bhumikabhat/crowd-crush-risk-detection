"""Evaluate the existing crowd-crush-risk models on the real Wuppertal 2018 data.

The models (DensityBaseline, FlowAwareRisk) were trained on simulated data
(data/features.csv, seeds 0-1).  Here we apply the *already-trained* models
to the 24 real experimental runs and report:

    detected / total events  |  mean lead time (s)  |  false alarms per hour

Split strategy: every run is its own group (no further train/test split
because we have only 24 runs and just 5 danger events).  We report at two
operating points:
  A) The threshold chosen on simulation training data (FA/h <= 10)
  B) A sweep over all thresholds to show the operating curve.

Run from the project root:
    python -m eval.run_eval_real
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval.lead_time import evaluate, sweep
from risk.score import DensityBaseline, FlowAwareRisk

# ── Parameters (match run_eval.py) ───────────────────────────────────────────
FPS        = 25       # real data fps
SIM_FPS    = 10       # simulation fps used during training
SMOOTH_S   = 2.0
WINDOW_S   = 15.0
HOLD_S     = 3.0
FA_TARGET  = 10.0
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.025), 3)

TRAIN_SEEDS = {0, 1}

# ── 1. Load simulation data and re-train models ───────────────────────────────
print("Loading simulation data...")
sim = pd.read_csv("data/features.csv")
for c in ["density", "entropy", "slowdown"]:
    sim[c] = sim.groupby("scenario")[c].transform(
        lambda s: s.rolling(int(SMOOTH_S * SIM_FPS), min_periods=1).mean()
    )
sim["y"] = (
    (sim["t"] >= sim["t_event"] - WINDOW_S) & (sim["t"] < sim["t_event"])
).astype(int)

train_sim = sim[sim["seed"].isin(TRAIN_SEEDS)]
fit_rows  = train_sim[~(train_sim["t"] >= train_sim["t_event"])]

baseline = DensityBaseline()
risk     = FlowAwareRisk().fit(fit_rows, fit_rows["y"].to_numpy())
print("Learned weights:", risk.weights)

# Choose thresholds on simulation training data
for m in (baseline, risk):
    sim[m.name] = m.score(sim)

tr_g = [g for _, g in sim[sim["seed"].isin(TRAIN_SEEDS)].groupby("scenario")]

sim_thresholds = {}
for m in (baseline, risk):
    tr = pd.DataFrame(sweep(tr_g, m.name, THRESHOLDS, hold_s=HOLD_S, window_s=WINDOW_S))
    ok = tr[tr["false_alarms_per_hour"] <= FA_TARGET]
    best = ok.sort_values(["detected", "mean_lead_s"], ascending=False).iloc[0]
    sim_thresholds[m.name] = float(best["threshold"])
    print(f"  {m.name}: sim-optimal threshold = {sim_thresholds[m.name]:.3f}")

# ── 2. Load real data ─────────────────────────────────────────────────────────
print("\nLoading real data...")
from data.real.loader import load_all_runs

real = load_all_runs()           # uses cache if available
print(f"  {real.scenario.nunique()} runs, {len(real):,} frames")
n_events_real = real.groupby("scenario")["t_event"].first().notna().sum()
print(f"  Danger events: {n_events_real} / {real.scenario.nunique()} runs")

# Smooth signals (causal rolling mean, same as in run_eval.py)
for c in ["density", "entropy", "slowdown"]:
    real[c] = real.groupby("scenario")[c].transform(
        lambda s: s.rolling(int(SMOOTH_S * FPS), min_periods=1).mean()
    )
real["y"] = (
    (real["t"] >= real["t_event"] - WINDOW_S) & (real["t"] < real["t_event"])
).astype(int)

# Score each run
for m in (baseline, risk):
    real[m.name] = m.score(real)

# ── 3. Evaluate at sim-optimal threshold ─────────────────────────────────────
real_groups = [g for _, g in real.groupby("scenario")]

print(f"\n{'='*70}")
print(f"Real-data results (threshold from simulation train, FA/h <= {FA_TARGET})")
print(f"{'='*70}")

rows, curves = [], {}
for m in (baseline, risk):
    th = sim_thresholds[m.name]
    res = evaluate(real_groups, m.name, th, hold_s=HOLD_S, window_s=WINDOW_S)
    rows.append({"method": m.name, **res})
    curves[m.name] = pd.DataFrame(
        sweep(real_groups, m.name, THRESHOLDS, hold_s=HOLD_S, window_s=WINDOW_S)
    )

out = pd.DataFrame(rows)[
    ["method", "threshold", "events", "detected", "missed",
     "mean_lead_s", "false_alarms", "false_alarms_per_hour", "hours"]
]
print(out.round(2).to_string(index=False))
out.round(3).to_csv("data/real/results_real.csv", index=False)

# ── 4. Operating curves ───────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for name, c in curves.items():
    ax[0].plot(c["false_alarms_per_hour"], c["mean_lead_s"],
               marker="o", ms=3, label=name)
    ax[1].plot(c["false_alarms_per_hour"], c["missed"],
               marker="o", ms=3, label=name)

for a in ax:
    a.legend()
    a.grid(alpha=0.3)

ax[0].set(xlabel="false alarms per hour", ylabel="mean lead time (s)",
          title="Real data: lead time vs false alarms")
ax[1].set(xlabel="false alarms per hour", ylabel="missed events",
          title="Real data: missed events")

plt.suptitle("Wuppertal 2018 bottleneck experiments (24 runs, 5 danger events)",
             fontsize=10)
plt.tight_layout()
plt.savefig("data/real/operating_curves_real.png", dpi=150)
print("\nSaved data/real/results_real.csv and data/real/operating_curves_real.png")

# ── 5. Per-door-width breakdown ───────────────────────────────────────────────
print("\nPer corridor-width breakdown (density-only | flow-aware):")
summary_rows = []
for door_w in sorted(real["door_w"].unique()):
    sub_g = [g for _, g in real[real["door_w"] == door_w].groupby("scenario")]
    n_ev = sum(1 for g in sub_g if not np.isnan(g["t_event"].iloc[0]))
    for m in (baseline, risk):
        th = sim_thresholds[m.name]
        res = evaluate(sub_g, m.name, th, hold_s=HOLD_S, window_s=WINDOW_S)
        summary_rows.append({
            "door_w": door_w, "n_runs": len(sub_g), "n_events": n_ev,
            "method": m.name,
            "detected": res["detected"],
            "mean_lead_s": round(res["mean_lead_s"], 2) if res["mean_lead_s"] is not np.nan else float("nan"),
            "fa_per_h": round(res["false_alarms_per_hour"], 2),
        })

summary = pd.DataFrame(summary_rows)
print(summary.to_string(index=False))
summary.to_csv("data/real/results_real_by_width.csv", index=False)
print("Saved data/real/results_real_by_width.csv")
