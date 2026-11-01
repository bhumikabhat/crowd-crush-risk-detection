"""Benchmark the existing per-frame pipeline on simulated video frames.

Pipeline under test (unchanged):
  features/extract.py  FeatureExtractor.update()
  risk/score.py        FlowAwareRisk.score() + DensityBaseline.score()
  alert/gate.py        alert_times() with 3 s hold

Measures at three resolutions:
  - Native sim size  (720 x 270  at PX_PER_M=30, 24 m x 9 m)
  - 640 x 480
  - 1280 x 720

For each resolution reports:
  - mean and 95th-percentile per-frame latency (ms)
  - throughput (fps)
  - event-to-alert latency: wall-clock time from the first frame whose score
    crosses the threshold to the frame where the 3 s hold expires and the
    alert fires (broken out: score_latency and gate_hold_latency)
  - CPU % and RSS memory (psutil)
  - Machine info (CPU model, cores, OS)

Saves data/edge/edge_results.csv and data/edge/edge_fps.png.

Run from the project root:  python -m edge.benchmark
Compatible with any machine; suitable for Raspberry Pi / Jetson later.
"""
from __future__ import annotations

import os
import platform
import sys
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    print("[benchmark] psutil not found; CPU/memory metrics skipped", file=sys.stderr)

from alert.gate import alert_times
from features.extract import FeatureExtractor
from risk.score import DensityBaseline, FlowAwareRisk, scaled
from sim.bottleneck import (IMG_H, IMG_W, PX_PER_M, ZONE_PX, Scenario, run)

# ── configuration ──────────────────────────────────────────────────────────
SCENARIO    = Scenario(door_w=1.2, n_agents=140, seed=1, duration=60.0, fps=10)
HOLD_S      = 3.0
THRESHOLD   = 0.525          # density-only threshold from run_eval.py
N_WARMUP    = 10             # frames to discard before timing
OUT_DIR     = "data/edge"

RESOLUTIONS = [
    ("native",  IMG_W, IMG_H),   # sim native  ≈ 720 × 270
    ("640x480", 640,   480),
    ("1280x720",1280,  720),
]


def _machine_info() -> dict:
    info = {
        "os":       platform.system() + " " + platform.release(),
        "python":   platform.python_version(),
        "cpu_model": platform.processor() or "unknown",
        "cpu_cores_logical": os.cpu_count(),
    }
    if _HAS_PSUTIL:
        try:
            info["cpu_cores_physical"] = psutil.cpu_count(logical=False)
            info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
        except Exception:
            pass
    return info


def _zone_px_for(src_w: int, src_h: int, dst_w: int, dst_h: int) -> tuple[int, int, int, int]:
    """Scale ZONE_PX from native sim dims to target resolution."""
    sx, sy = dst_w / src_w, dst_h / src_h
    x0, y0, x1, y1 = ZONE_PX
    return int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)


def _px_per_m_for(dst_w: int) -> float:
    return PX_PER_M * dst_w / IMG_W


def benchmark_resolution(
    label: str, dst_w: int, dst_h: int,
    frames_native: list[np.ndarray],
    densities_true: list[float],
    risk_model: FlowAwareRisk,
) -> dict:
    """Run the pipeline on one set of resized frames and return metrics."""

    # Resize frames
    if dst_w == IMG_W and dst_h == IMG_H:
        frames = frames_native
    else:
        frames = [cv2.resize(f, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
                  for f in frames_native]

    zone_px = _zone_px_for(IMG_W, IMG_H, dst_w, dst_h)
    px_m    = _px_per_m_for(dst_w)
    extractor = FeatureExtractor(zone_px=zone_px, px_per_m=px_m, fps=SCENARIO.fps)
    baseline  = DensityBaseline()

    # -- Warm up (don't time these) --
    for f in frames[:N_WARMUP]:
        extractor.update(f)
    extractor = FeatureExtractor(zone_px=zone_px, px_per_m=px_m, fps=SCENARIO.fps)  # reset

    # -- Timed loop --
    proc = psutil.Process() if _HAS_PSUTIL else None
    if proc:
        proc.cpu_percent(interval=None)            # prime the CPU meter
        mem_before = proc.memory_info().rss

    t_frames: list[float] = []
    feat_rows: list[dict] = []
    scores: list[float] = []
    t_wall: list[float] = []

    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        feats = extractor.update(frame)
        t1 = time.perf_counter()
        t_frames.append((t1 - t0) * 1000.0)  # ms

        feat_rows.append({
            "density":  feats["density"],
            "entropy":  feats["entropy"],
            "slowdown": feats["slowdown"],
            "density_true": densities_true[i],
        })
        # Score using flow-aware (requires at least a rudimentary DataFrame)
        row_df = pd.DataFrame([feat_rows[-1]])
        s = float(baseline.score(row_df)[0])
        scores.append(s)
        t_wall.append(time.perf_counter())

    if proc:
        cpu_pct = proc.cpu_percent(interval=None)
        mem_after = proc.memory_info().rss
    else:
        cpu_pct = float("nan")
        mem_before = mem_after = 0

    t_arr  = np.array(t_frames)
    fps_achieved = 1000.0 / t_arr.mean()

    # -- Event-to-alert latency --
    # sim time axis
    sim_t = np.arange(len(frames)) / SCENARIO.fps
    alerts = alert_times(sim_t, np.array(scores), THRESHOLD, hold_s=HOLD_S)

    # first frame where score >= threshold
    above = [i for i, s in enumerate(scores) if s >= THRESHOLD]
    t_first_above_sim = sim_t[above[0]] if above else float("nan")
    t_first_alert_sim = alerts[0] if alerts else float("nan")

    # event time (sim) from true density
    danger_hold = int(HOLD_S * SCENARIO.fps)
    t_event_sim = float("nan")
    dens_arr = np.array(densities_true)
    for i in range(len(dens_arr) - danger_hold + 1):
        if (dens_arr[i: i + danger_hold] >= 3.5).all():
            t_event_sim = sim_t[i]
            break

    # score_leads_event_by_s > 0 means score crosses threshold BEFORE event (pre-warning)
    # score_leads_event_by_s < 0 means score crosses AFTER event (missed)
    if not np.isnan(t_event_sim) and not np.isnan(t_first_above_sim):
        score_leads_event_by_s = round(t_event_sim - t_first_above_sim, 3)
    else:
        score_leads_event_by_s = float("nan")

    # alert_leads_event_by_s > 0: alert fires before event (pre-warning after hold)
    if not np.isnan(t_event_sim) and not np.isnan(t_first_alert_sim):
        alert_leads_event_by_s = round(t_event_sim - t_first_alert_sim, 3)
    else:
        alert_leads_event_by_s = float("nan")

    # wall-clock elapsed from event frame to alert frame (processing overhead)
    if not np.isnan(t_event_sim) and not np.isnan(t_first_alert_sim):
        ev_idx   = int(round(t_event_sim * SCENARIO.fps))
        al_idx   = int(round(t_first_alert_sim * SCENARIO.fps))
        al_idx   = min(al_idx, len(t_wall) - 1)
        ev_idx   = min(ev_idx, len(t_wall) - 1)
        wall_event_to_alert_ms = round(abs(t_wall[al_idx] - t_wall[ev_idx]) * 1000.0, 1)
    else:
        wall_event_to_alert_ms = float("nan")

    # gate_hold_s is always HOLD_S by protocol (not changed)
    gate_hold_s = HOLD_S

    return {
        "resolution":              label,
        "frame_w":                 dst_w,
        "frame_h":                 dst_h,
        "n_frames":                len(frames),
        "mean_latency_ms":         round(t_arr.mean(), 3),
        "p95_latency_ms":          round(np.percentile(t_arr, 95), 3),
        "fps_achieved":            round(fps_achieved, 1),
        "t_event_sim_s":           round(t_event_sim, 2) if not np.isnan(t_event_sim) else float("nan"),
        "t_first_score_above_s":   round(t_first_above_sim, 2) if not np.isnan(t_first_above_sim) else float("nan"),
        "t_first_alert_sim_s":     round(t_first_alert_sim, 2) if not np.isnan(t_first_alert_sim) else float("nan"),
        # >0 means fires BEFORE event (pre-warning); <0 means fires after
        "score_leads_event_by_s":  score_leads_event_by_s,
        "alert_leads_event_by_s":  alert_leads_event_by_s,
        "gate_hold_s":             gate_hold_s,
        "wall_event_to_alert_ms":  wall_event_to_alert_ms,
        "cpu_pct":                 round(cpu_pct, 1),
        "mem_rss_mb":              round((mem_after - mem_before) / 1e6, 1),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    minfo = _machine_info()
    print("Machine info:")
    for k, v in minfo.items():
        print(f"  {k}: {v}")
    print()

    # -- Generate frames once at native resolution --
    print(f"Simulating scenario: door_w={SCENARIO.door_w} m, "
          f"N={SCENARIO.n_agents}, seed={SCENARIO.seed}, "
          f"duration={SCENARIO.duration}s ...", flush=True)
    t0 = time.perf_counter()
    frames_native, densities_true = [], []
    for _t, frame, d_true in run(SCENARIO):
        frames_native.append(frame.copy())
        densities_true.append(d_true)
    print(f"  {len(frames_native)} frames in {time.perf_counter()-t0:.1f}s", flush=True)

    # -- Train a FlowAwareRisk model (quick, on sim data) --
    try:
        sim_df = pd.read_csv("data/features.csv")
        for c in ["density", "entropy", "slowdown"]:
            sim_df[c] = sim_df.groupby("scenario")[c].transform(
                lambda s: s.rolling(20, min_periods=1).mean()
            )
        sim_df["y"] = ((sim_df["t"] >= sim_df["t_event"] - 15.0) &
                       (sim_df["t"] < sim_df["t_event"])).astype(int)
        tr = sim_df[sim_df["seed"].isin({0, 1})]
        fit = tr[~(tr["t"] >= tr["t_event"])]
        risk_model = FlowAwareRisk().fit(fit, fit["y"].to_numpy())
        print(f"  Trained FlowAwareRisk: weights={risk_model.weights}")
    except Exception as e:
        print(f"  [warn] Could not train FlowAwareRisk ({e}); using baseline only.")
        risk_model = None

    # -- Benchmark each resolution --
    results = []
    for label, dst_w, dst_h in RESOLUTIONS:
        print(f"Benchmarking {label} ({dst_w}x{dst_h}) ...", flush=True)
        r = benchmark_resolution(label, dst_w, dst_h,
                                 frames_native, densities_true, risk_model)
        results.append(r)
        print(f"  mean={r['mean_latency_ms']:.2f}ms  p95={r['p95_latency_ms']:.2f}ms  "
              f"fps={r['fps_achieved']:.1f}  cpu={r['cpu_pct']:.1f}%  "
              f"mem_delta={r['mem_rss_mb']:.1f}MB")

    # -- Add machine info as extra columns --
    for r in results:
        r.update(minfo)

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUT_DIR, "edge_results.csv"), index=False)
    print(f"\nSaved {OUT_DIR}/edge_results.csv")

    # -- Bar chart --
    labels   = [r["resolution"] for r in results]
    fps_vals = [r["fps_achieved"] for r in results]
    p95_vals = [r["p95_latency_ms"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(labels))
    w = 0.6
    bars = ax1.bar(x, fps_vals, w, color=["#4c72b0", "#dd8452", "#55a868"])
    ax1.axhline(SCENARIO.fps, color="red", linestyle="--", linewidth=1.2, label=f"target {SCENARIO.fps} fps")
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylabel("Achieved FPS"); ax1.set_title("Throughput per resolution")
    ax1.legend(); ax1.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, fps_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{v:.0f}", ha="center", va="bottom", fontsize=9)

    bars2 = ax2.bar(x, p95_vals, w, color=["#4c72b0", "#dd8452", "#55a868"])
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylabel("P95 latency (ms)"); ax2.set_title("P95 per-frame latency")
    ax2.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars2, p95_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{v:.1f}ms", ha="center", va="bottom", fontsize=9)

    cpu_str = minfo.get("cpu_model", "unknown")[:50]
    fig.suptitle(f"Edge benchmark  |  {cpu_str}  |  {minfo.get('os','')}", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "edge_fps.png"), dpi=150)
    print(f"Saved {OUT_DIR}/edge_fps.png")

    # -- Print summary table --
    print("\nResults:")
    print(res_df[["resolution", "mean_latency_ms", "p95_latency_ms",
                  "fps_achieved", "t_event_sim_s", "t_first_score_above_s",
                  "t_first_alert_sim_s", "score_leads_event_by_s",
                  "alert_leads_event_by_s", "gate_hold_s",
                  "wall_event_to_alert_ms", "cpu_pct", "mem_rss_mb"]].to_string(index=False))


if __name__ == "__main__":
    main()
