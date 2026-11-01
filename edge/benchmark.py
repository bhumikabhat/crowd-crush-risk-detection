"""Benchmark the existing per-frame pipeline on simulated video frames.

Pipeline under test (unchanged):
  features/extract.py  FeatureExtractor.update()
  risk/score.py        DensityBaseline.score()
  alert/gate.py        alert_times() with 3 s hold

Measures at three resolutions:
  - Native sim size  (720 x 270  at PX_PER_M=30, 24 m x 9 m)
  - 640 x 480
  - 1280 x 720

Reported metrics per resolution:
  - mean per-frame latency (ms)
  - 95th-percentile per-frame latency (ms)
  - throughput (FPS)
  - absolute process RSS memory (MB)  — taken after the timed loop
  - gate_hold_s: the 3.0 s alert-gate hold (protocol constant, not changed)
  - cpu_model, os, cores, ram_gb: machine info

NOTE on alert timing:
  t_first_score_above_s and t_first_alert_sim_s are scenario-dependent quantities
  that differ by resolution only because different resolutions produce slightly
  different density estimates from the same underlying agent positions.
  They are recorded in the CSV for reproducibility but are NOT a benchmark result;
  do not compare them across resolutions.

Saves data/edge/edge_results.csv and data/edge/edge_fps.png.
Summary table is printed by reading the saved CSV (not from in-memory objects).

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
    print("[benchmark] psutil not found; memory metric will be NaN", file=sys.stderr)

from alert.gate import alert_times
from features.extract import FeatureExtractor
from risk.score import DensityBaseline
from sim.bottleneck import IMG_H, IMG_W, PX_PER_M, ZONE_PX, Scenario, run

# ── configuration ──────────────────────────────────────────────────────────
SCENARIO  = Scenario(door_w=1.2, n_agents=140, seed=1, duration=60.0, fps=10)
HOLD_S    = 3.0             # alert-gate hold — protocol constant, do not change
THRESHOLD = 0.525           # density-only threshold from run_eval.py train split
N_WARMUP  = 10              # frames to discard before timing (JIT warm-up)
OUT_DIR   = "data/edge"

RESOLUTIONS = [
    ("native",   IMG_W, IMG_H),   # sim native  720 x 270
    ("640x480",  640,   480),
    ("1280x720", 1280,  720),
]


# ── helpers ────────────────────────────────────────────────────────────────
def _machine_info() -> dict:
    info = {
        "os":                platform.system() + " " + platform.release(),
        "python":            platform.python_version(),
        "cpu_model":         platform.processor() or "unknown",
        "cpu_cores_logical": os.cpu_count(),
    }
    if _HAS_PSUTIL:
        try:
            info["cpu_cores_physical"] = psutil.cpu_count(logical=False)
            info["ram_gb"]             = round(psutil.virtual_memory().total / 1e9, 1)
        except Exception:
            pass
    return info


def _zone_px_for(dst_w: int, dst_h: int) -> tuple[int, int, int, int]:
    """Scale ZONE_PX from native sim dims to target resolution."""
    sx, sy = dst_w / IMG_W, dst_h / IMG_H
    x0, y0, x1, y1 = ZONE_PX
    return int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)


def _px_per_m_for(dst_w: int) -> float:
    return PX_PER_M * dst_w / IMG_W


# ── per-resolution benchmark ───────────────────────────────────────────────
def benchmark_resolution(
    label: str,
    dst_w: int,
    dst_h: int,
    frames_native: list[np.ndarray],
    densities_true: list[float],
) -> dict:
    """Benchmark the unchanged pipeline at one resolution; return metrics dict."""

    # Resize (no-op for native)
    if dst_w == IMG_W and dst_h == IMG_H:
        frames = frames_native
    else:
        frames = [
            cv2.resize(f, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
            for f in frames_native
        ]

    zone_px   = _zone_px_for(dst_w, dst_h)
    px_m      = _px_per_m_for(dst_w)
    extractor = FeatureExtractor(zone_px=zone_px, px_per_m=px_m, fps=SCENARIO.fps)
    baseline  = DensityBaseline()

    # Warm-up (not timed)
    for f in frames[:N_WARMUP]:
        extractor.update(f)
    extractor = FeatureExtractor(zone_px=zone_px, px_per_m=px_m, fps=SCENARIO.fps)  # reset

    # RSS before timed loop
    proc = psutil.Process() if _HAS_PSUTIL else None
    mem_rss_before = proc.memory_info().rss if proc else 0

    # Timed loop
    t_frames: list[float] = []
    scores:   list[float] = []

    for frame in frames:
        t0    = time.perf_counter()
        feats = extractor.update(frame)
        t1    = time.perf_counter()
        t_frames.append((t1 - t0) * 1000.0)   # ms

        row_df = pd.DataFrame([{
            "density":  feats["density"],
            "entropy":  feats["entropy"],
            "slowdown": feats["slowdown"],
        }])
        scores.append(float(baseline.score(row_df)[0]))

    # Absolute RSS after timed loop
    mem_rss_mb = round(proc.memory_info().rss / 1e6, 1) if proc else float("nan")

    t_arr        = np.array(t_frames)
    fps_achieved = 1000.0 / t_arr.mean()

    # Alert timing — recorded for reproducibility, not a benchmark result
    # (see module docstring NOTE)
    sim_t             = np.arange(len(frames)) / SCENARIO.fps
    alerts            = alert_times(sim_t, np.array(scores), THRESHOLD, hold_s=HOLD_S)
    above             = [i for i, s in enumerate(scores) if s >= THRESHOLD]
    t_first_score_s   = round(float(sim_t[above[0]]), 2) if above else float("nan")
    t_first_alert_s   = round(float(alerts[0]), 2)       if alerts else float("nan")

    return {
        "resolution":          label,
        "frame_w":             dst_w,
        "frame_h":             dst_h,
        "n_frames":            len(frames),
        "mean_latency_ms":     round(t_arr.mean(), 3),
        "p95_latency_ms":      round(float(np.percentile(t_arr, 95)), 3),
        "fps_achieved":        round(fps_achieved, 1),
        "mem_rss_mb":          mem_rss_mb,          # absolute RSS after loop
        "gate_hold_s":         HOLD_S,              # protocol constant
        # -- scenario-dependent, not a benchmark result (see NOTE) --
        "t_first_score_above_s": t_first_score_s,
        "t_first_alert_sim_s":   t_first_alert_s,
    }


# ── main ───────────────────────────────────────────────────────────────────
def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    minfo = _machine_info()
    print("Machine info:")
    for k, v in minfo.items():
        print(f"  {k}: {v}")
    print()

    # Generate frames once at native resolution
    print(
        f"Simulating: door_w={SCENARIO.door_w} m, N={SCENARIO.n_agents}, "
        f"seed={SCENARIO.seed}, duration={SCENARIO.duration} s …",
        flush=True,
    )
    t0 = time.perf_counter()
    frames_native, densities_true = [], []
    for _t, frame, d_true in run(SCENARIO):
        frames_native.append(frame.copy())
        densities_true.append(d_true)
    print(f"  {len(frames_native)} frames in {time.perf_counter() - t0:.1f} s", flush=True)

    # Benchmark each resolution
    results: list[dict] = []
    for label, dst_w, dst_h in RESOLUTIONS:
        print(f"Benchmarking {label} ({dst_w}x{dst_h}) …", flush=True)
        r = benchmark_resolution(label, dst_w, dst_h, frames_native, densities_true)
        results.append(r)
        print(
            f"  mean={r['mean_latency_ms']:.2f} ms  "
            f"p95={r['p95_latency_ms']:.2f} ms  "
            f"fps={r['fps_achieved']:.1f}  "
            f"rss={r['mem_rss_mb']:.0f} MB"
        )

    # Attach machine info and save CSV
    for r in results:
        r.update(minfo)

    csv_path = os.path.join(OUT_DIR, "edge_results.csv")
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    # Bar chart (FPS and P95 latency)
    labels   = [r["resolution"]     for r in results]
    fps_vals = [r["fps_achieved"]    for r in results]
    p95_vals = [r["p95_latency_ms"] for r in results]
    colors   = ["#4c72b0", "#dd8452", "#55a868"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x, w = np.arange(len(labels)), 0.6

    bars1 = ax1.bar(x, fps_vals, w, color=colors)
    ax1.axhline(SCENARIO.fps, color="red", linestyle="--", linewidth=1.2,
                label=f"target {SCENARIO.fps} fps")
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylabel("Achieved FPS"); ax1.set_title("Throughput per resolution")
    ax1.legend(); ax1.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars1, fps_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{v:.0f}", ha="center", va="bottom", fontsize=9)

    bars2 = ax2.bar(x, p95_vals, w, color=colors)
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylabel("P95 latency (ms)"); ax2.set_title("P95 per-frame latency")
    ax2.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars2, p95_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{v:.1f} ms", ha="center", va="bottom", fontsize=9)

    cpu_str = minfo.get("cpu_model", "")[:50]
    fig.suptitle(f"Edge benchmark  |  {cpu_str}  |  {minfo.get('os', '')}", fontsize=9)
    plt.tight_layout()
    png_path = os.path.join(OUT_DIR, "edge_fps.png")
    plt.savefig(png_path, dpi=150)
    print(f"Saved {png_path}")

    # Print summary table by reading the saved CSV (not from memory)
    print("\nNOTE: t_first_score_above_s / t_first_alert_sim_s are scenario-dependent")
    print("      quantities recorded for reproducibility; do not compare across resolutions.")
    print(f"\ngate_hold_s = {HOLD_S}  (protocol constant — not changed)\n")
    print("Summary (read from CSV):")
    df_out = pd.read_csv(csv_path)
    display_cols = [
        "resolution", "mean_latency_ms", "p95_latency_ms",
        "fps_achieved", "mem_rss_mb", "gate_hold_s",
    ]
    print(df_out[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
