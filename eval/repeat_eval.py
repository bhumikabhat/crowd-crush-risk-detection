"""Repeat the density-only vs flow-aware comparison over many train/test splits.

The original eval uses one split (train seeds 0,1 / test seeds 2,3). This script reports how stable
the result is:
  - "seed" splits: all 6 ways of choosing 2 of the 4 seeds for training (the rest test),
  - "random" splits: random 50/50 splits of the runs.
Same protocol as run_eval.py: weights and threshold chosen on train (false alarms <= FA_TARGET/h),
reported on test. Writes data/repeat_results.csv (one row per split and method) and
data/repeat_summary.csv (mean and 95% interval over splits, plus paired flow-aware minus density-only).

Run from the project root:  python -m eval.repeat_eval --reps 100
"""
import argparse
import itertools
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from eval.lead_time import evaluate, sweep
from risk.score import DensityBaseline, FlowAwareRisk

FPS, SMOOTH_S, WINDOW_S, HOLD_S = 10, 2.0, 15.0, 3.0
FA_TARGET = 10.0
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.025), 3)

_DF = None  # set in each worker


def load():
    df = pd.read_csv("data/features.csv")
    for c in ["density", "entropy", "slowdown"]:
        df[c] = df.groupby("scenario")[c].transform(lambda s: s.rolling(int(SMOOTH_S * FPS), min_periods=1).mean())
    df["y"] = ((df["t"] >= df["t_event"] - WINDOW_S) & (df["t"] < df["t_event"])).astype(int)
    return df


def _init():
    global _DF
    _DF = load()


def one_split(args):
    kind, split_id, train_ids = args
    df = _DF
    is_train = df["scenario"].isin(train_ids)
    train, test = df[is_train].copy(), df[~is_train].copy()
    fit_rows = train[~(train["t"] >= train["t_event"])]
    baseline, risk = DensityBaseline(), FlowAwareRisk().fit(fit_rows, fit_rows["y"].to_numpy())
    for m in (baseline, risk):
        train[m.name] = m.score(train)
        test[m.name] = m.score(test)
    tr_g = [g for _, g in train.groupby("scenario")]
    te_g = [g for _, g in test.groupby("scenario")]
    rows = []
    for m in (baseline, risk):
        tr = pd.DataFrame(sweep(tr_g, m.name, THRESHOLDS, hold_s=HOLD_S, window_s=WINDOW_S))
        ok = tr[tr["false_alarms_per_hour"] <= FA_TARGET]
        if ok.empty:
            continue
        best = ok.sort_values(["detected", "mean_lead_s"], ascending=False).iloc[0]
        res = evaluate(te_g, m.name, best["threshold"], hold_s=HOLD_S, window_s=WINDOW_S)
        res.update(kind=kind, split=split_id, method=m.name,
                   detect_rate=res["detected"] / res["events"] if res["events"] else np.nan)
        if m.name == "flow-aware":
            res.update({f"w_{k}": float(v) for k, v in risk.weights.items()})
        rows.append(res)
    return rows


def ci(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return x.mean(), np.percentile(x, 2.5), np.percentile(x, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100, help="number of random 50/50 splits")
    ap.add_argument("--rng", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv("data/features.csv", usecols=["scenario", "seed"]).drop_duplicates()
    scen, seed_of = df["scenario"].to_numpy(), dict(zip(df["scenario"], df["seed"]))
    seeds = sorted(df["seed"].unique())
    jobs = []
    for i, tr in enumerate(itertools.combinations(seeds, 2)):
        jobs.append(("seed", i, {s for s in scen if seed_of[s] in tr}))
    rng = np.random.default_rng(args.rng)
    for i in range(args.reps):
        perm = rng.permutation(scen)
        jobs.append(("random", i, set(perm[: len(perm) // 2])))
    print(f"{len(jobs)} splits on {os.cpu_count()} cores", flush=True)

    with Pool(os.cpu_count(), initializer=_init) as pool:
        out = []
        for k, rows in enumerate(pool.imap_unordered(one_split, jobs), 1):
            out.extend(rows)
            if k % 10 == 0:
                print(f"  {k}/{len(jobs)} splits done", flush=True)
    res = pd.DataFrame(out)
    res.to_csv("data/repeat_results.csv", index=False)

    summ = []
    for kind in ("seed", "random"):
        sub = res[res["kind"] == kind]
        for method, g in sub.groupby("method"):
            for col in ("detect_rate", "missed", "mean_lead_s", "false_alarms_per_hour"):
                m, lo, hi = ci(g[col])
                summ.append({"splits": kind, "method": method, "metric": col, "mean": m, "lo95": lo, "hi95": hi})
        wide = sub.pivot_table(index="split", columns="method",
                               values=["detect_rate", "missed", "mean_lead_s", "false_alarms_per_hour"])
        for col in ("detect_rate", "missed", "mean_lead_s", "false_alarms_per_hour"):
            d = (wide[(col, "flow-aware")] - wide[(col, "density-only")]).dropna()
            m, lo, hi = ci(d)
            summ.append({"splits": kind, "method": "flow-aware minus density-only", "metric": col,
                         "mean": m, "lo95": lo, "hi95": hi})
            if col in ("detect_rate", "mean_lead_s"):
                summ.append({"splits": kind, "method": "share of splits flow-aware better", "metric": col,
                             "mean": float((d > 0).mean()), "lo95": np.nan, "hi95": np.nan})
    s = pd.DataFrame(summ)
    s.round(3).to_csv("data/repeat_summary.csv", index=False)
    print("\n" + s.round(3).to_string(index=False))
    print("\nsaved data/repeat_results.csv and data/repeat_summary.csv")


if __name__ == "__main__":
    main()
