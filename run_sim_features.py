"""Run bottleneck scenarios, extract video features per frame, save data/features.csv."""
import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from features.extract import FeatureExtractor
from sim.bottleneck import PX_PER_M, ZONE_PX, Scenario, run

DANGER_DENSITY = 3.5      # people / m^2 in the watched zone (true positions)
DANGER_HOLD_S = 2.0       # must be sustained this long


def event_time(t, d_true, fps):
    n = int(DANGER_HOLD_S * fps)
    over = d_true >= DANGER_DENSITY
    for i in range(len(over) - n + 1):
        if over[i:i + n].all():
            return t[i]
    return np.nan


def one(args):
    sid, sc = args
    ext = FeatureExtractor(ZONE_PX, PX_PER_M, sc.fps)
    rows = []
    for t, frame, d_true in run(sc):
        f = ext.update(frame)
        rows.append({"scenario": sid, "door_w": sc.door_w, "n_agents": sc.n_agents,
                     "seed": sc.seed, "t": t, "density_true": d_true, **f})
    df = pd.DataFrame(rows)
    df["t_event"] = event_time(df["t"].to_numpy(), df["density_true"].to_numpy(), sc.fps)
    return df


if __name__ == "__main__":
    widths = [0.8, 1.0, 1.2, 1.6, 2.0, 2.5]
    counts = [60, 100, 140, 180, 200]
    seeds = [int(s) for s in (sys.argv[1].split(",") if len(sys.argv) > 1 else "0,1,2,3")]
    scs = [Scenario(door_w=w, n_agents=n, seed=s, duration=80.0)
           for w, n, s in itertools.product(widths, counts, seeds)]
    jobs = list(enumerate(scs))
    print(f"{len(jobs)} scenarios on {os.cpu_count()} cores", flush=True)
    with Pool(os.cpu_count()) as pool:
        frames = []
        for i, df in enumerate(pool.imap_unordered(one, jobs), 1):
            frames.append(df)
            if i % 10 == 0:
                print(f"  {i}/{len(jobs)} done", flush=True)
    os.makedirs("data", exist_ok=True)
    pd.concat(frames).sort_values(["scenario", "t"]).to_csv("data/features.csv", index=False)
    print("saved data/features.csv")
