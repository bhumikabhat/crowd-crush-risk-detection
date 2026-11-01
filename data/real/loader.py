"""Load Wuppertal 2018 bottleneck trajectories from the Jülich Pedestrian Dynamics
Data Archive (DOI: 10.34735/ped.2018.1) and produce a feature table with the same
column schema as data/features.csv.

Trajectory files live in data/real/trajectories_txt/*.txt (PeTrack format, 25 fps).
Format (space-delimited, comment lines start with '#'):
    id  frame  x/m  y/m  z/m

Coordinate system (from the experimental setup sketch):
  • y = 0  is the entrance gate (bottleneck).
  • Positive y: upstream corridor (crowd forms here, max ≈ 6.7 m).
  • Negative y: downstream (past the gate).
  • x = 0: centreline; corridor half-width = corridor_width / 2.

Watched zone (density calculation):
  A 2 m deep strip directly in front of the gate: y ∈ [0, ZONE_DEPTH_M],
  x ∈ [−corridor_width/2 − ZONE_MARGIN, +corridor_width/2 + ZONE_MARGIN].
  Zone area = ZONE_DEPTH_M × corridor_width.

Danger rule (identical to the simulation):
  density_true >= DANGER_DENSITY  for at least  DANGER_HOLD_S  continuous seconds
  → t_event = first frame of that sustained period.

Feature derivation per frame (from ground-truth trajectories; no video needed):
  density_true  – true count in zone / zone_area (p/m²)
  density       – same as density_true (ground truth proxy; no camera noise)
  entropy       – Shannon entropy of velocity-direction histogram (8 bins, normalised)
  slowdown      – 1 − min(mean_speed / V_FREE, 1)
  speed         – mean speed of pedestrians in zone (m/s)
  t_event       – event time (NaN if never triggered)

Usage
-----
    from data.real.loader import load_all_runs
    df = load_all_runs()          # returns one big DataFrame
    # or per-run DataFrames:
    from data.real.loader import load_run
    df_one = load_run("data/real/trajectories_txt/030_c_56_h0.txt")
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

# ── constants ────────────────────────────────────────────────────────────────
FPS = 25                  # data is always 25 fps
ZONE_DEPTH_M = 2.0        # depth of the monitored strip in front of the gate (m)
ZONE_MARGIN = 0.0         # extra x margin beyond corridor walls (m)
V_FREE = 1.2              # free-flow walking speed (m/s) – matches sim default
N_BINS = 8                # velocity-direction bins for entropy
DANGER_DENSITY = 3.5      # people / m² threshold
DANGER_HOLD_S = 2.0       # must be sustained this long (s)
DANGER_HOLD_FRAMES = int(DANGER_HOLD_S * FPS)  # = 50

# Filename pattern: <run_id>_<priming>_<width_tenths>_<motivation>.txt
# e.g.  030_c_56_h0.txt  →  width = 5.6 m
_FNAME_RE = re.compile(
    r"^(?P<run_id>\d{3})_(?P<priming>[cq])_(?P<width_tenths>\d+)_(?P<motiv>[^.]+)\.txt$"
)


def _parse_filename(fname: str) -> dict:
    """Return {run_id, priming, door_w, motivation} parsed from a trajectory filename."""
    m = _FNAME_RE.match(fname)
    if m is None:
        raise ValueError(f"Cannot parse filename: {fname!r}")
    width_tenths = int(m.group("width_tenths"))
    # The width field stores tenths of a metre (e.g., '56' → 5.6 m, '12' → 1.2 m)
    door_w = round(width_tenths / 10.0, 1)
    return {
        "run_id": m.group("run_id"),
        "priming": m.group("priming"),
        "door_w": door_w,
        "motivation": m.group("motiv"),
    }


def _load_traj(path: str | Path) -> np.ndarray:
    """Read a PeTrack .txt trajectory file → array (N, 5): id frame x y z."""
    return np.loadtxt(str(path), comments="#", dtype=np.float64)


def _event_time(frames: np.ndarray, density: np.ndarray) -> float:
    """Return the time (s) of the first sustained danger event, or NaN."""
    over = density >= DANGER_DENSITY
    for i in range(len(over) - DANGER_HOLD_FRAMES + 1):
        if over[i : i + DANGER_HOLD_FRAMES].all():
            return float(frames[i] / FPS)
    return float("nan")


def load_run(path: str | Path, scenario_id: int = 0) -> pd.DataFrame:
    """Build a per-frame feature DataFrame for one trajectory file.

    Parameters
    ----------
    path        : path to the .txt trajectory file
    scenario_id : integer ID (used as the 'scenario' column value)

    Returns
    -------
    DataFrame with columns matching data/features.csv schema:
        scenario, door_w, n_agents, seed, t,
        density_true, density, entropy, slowdown, speed, t_event
    (seed is always NaN for real data; n_agents = number of unique IDs)
    """
    path = Path(path)
    meta = _parse_filename(path.name)
    door_w: float = meta["door_w"]

    traj = _load_traj(path)
    # columns: 0=id, 1=frame, 2=x, 3=y, 4=z
    ids, frames_raw, xs, ys = (
        traj[:, 0].astype(int),
        traj[:, 1].astype(int),
        traj[:, 2],
        traj[:, 3],
    )
    n_agents = len(np.unique(ids))

    half_w = door_w / 2.0 + ZONE_MARGIN
    zone_area = ZONE_DEPTH_M * door_w  # m²

    # ── Build a tidy per-agent-per-frame DataFrame for vectorised ops ─────────
    tdf = pd.DataFrame({"pid": ids, "frame": frames_raw, "x": xs, "y": ys})
    tdf = tdf.sort_values(["pid", "frame"])

    # Velocity via shift within each agent group
    tdf["dx"] = tdf.groupby("pid")["x"].diff()
    tdf["dy"] = tdf.groupby("pid")["y"].diff()
    tdf["mag"] = np.sqrt(tdf["dx"] ** 2 + tdf["dy"] ** 2) * FPS  # m/s
    tdf["angle"] = np.arctan2(tdf["dy"], tdf["dx"]) % (2 * math.pi)

    # Zone membership in current frame
    tdf["in_zone"] = (
        (tdf["y"] >= 0.0) & (tdf["y"] <= ZONE_DEPTH_M) & (tdf["x"].abs() <= half_w)
    )
    # Only keep velocity rows where previous frame was consecutive
    tdf["dframe"] = tdf.groupby("pid")["frame"].diff()
    tdf["vel_valid"] = (tdf["dframe"] == 1) & tdf["in_zone"]

    all_frames = np.arange(int(tdf["frame"].min()), int(tdf["frame"].max()) + 1, dtype=int)

    # Per-frame density (count of agents in zone)
    zone_counts = (
        tdf[tdf["in_zone"]].groupby("frame").size().reindex(all_frames, fill_value=0)
    )

    # Per-frame velocity stats (only valid zone+consecutive rows)
    vel_df = tdf[tdf["vel_valid"]]

    def _frame_entropy(grp: pd.DataFrame) -> float:
        if grp.empty:
            return 0.0
        w = grp["mag"].to_numpy()
        a = grp["angle"].to_numpy()
        if w.sum() < 1e-9:
            return 0.0
        hist, _ = np.histogram(a, bins=N_BINS, range=(0, 2 * math.pi), weights=w)
        p = hist / hist.sum()
        nz = p[p > 0]
        return float(-(nz * np.log2(nz)).sum() / math.log2(N_BINS))

    vel_agg = vel_df.groupby("frame").agg(mean_speed=("mag", "mean")).reindex(all_frames)
    entropy_by_frame = (
        vel_df.groupby("frame").apply(_frame_entropy).reindex(all_frames, fill_value=0.0)
    )

    # ── Assemble per-frame rows ────────────────────────────────────────────────
    rows = []
    for frame in all_frames:
        t = float(frame) / FPS
        n_in_zone = int(zone_counts.loc[frame])
        density_val = n_in_zone / zone_area if zone_area > 0 else 0.0

        speed_val = vel_agg.loc[frame, "mean_speed"]  # may be NaN
        if math.isnan(float(speed_val)):
            slowdown_val = 0.0
            speed_out = float("nan")
        else:
            speed_out = float(speed_val)
            slowdown_val = float(max(0.0, 1.0 - min(speed_out / V_FREE, 1.0)))

        entropy_val = float(entropy_by_frame.loc[frame])

        rows.append(
            {
                "scenario": scenario_id,
                "door_w": door_w,
                "n_agents": n_agents,
                "seed": float("nan"),
                "t": round(t, 6),
                "density_true": round(density_val, 6),
                "density": round(density_val, 6),
                "entropy": round(entropy_val, 6),
                "slowdown": round(slowdown_val, 6),
                "speed": round(speed_out, 6) if not math.isnan(speed_out) else float("nan"),
                "t_event": float("nan"),  # filled in below
            }
        )

    df = pd.DataFrame(rows)

    # Compute event time
    t_event = _event_time(all_frames, df["density_true"].to_numpy())
    df["t_event"] = t_event

    return df


def iter_runs(
    traj_dir: str | Path = "data/real/trajectories_txt",
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield (filename, DataFrame) for every .txt trajectory in *traj_dir*."""
    traj_dir = Path(traj_dir)
    txt_files = sorted(p for p in traj_dir.glob("*.txt") if _FNAME_RE.match(p.name))
    for sid, path in enumerate(txt_files):
        yield path.name, load_run(path, scenario_id=sid)


def load_all_runs(
    traj_dir: str | Path = "data/real/trajectories_txt",
    cache_path: str | Path = "data/real/features_real.csv",
    force: bool = False,
) -> pd.DataFrame:
    """Load all runs from *traj_dir*, optionally caching to *cache_path*.

    If *cache_path* already exists and *force* is False the cached CSV is
    returned immediately (fast path).

    Parameters
    ----------
    traj_dir   : directory with *.txt trajectory files
    cache_path : where to write / read the combined CSV cache
    force      : if True, ignore any existing cache and recompute
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        print(f"[loader] Loading cached features from {cache_path}")
        return pd.read_csv(cache_path)

    traj_dir = Path(traj_dir)
    txt_files = sorted(p for p in traj_dir.glob("*.txt") if _FNAME_RE.match(p.name))
    if not txt_files:
        raise FileNotFoundError(
            f"No .txt trajectory files found in {traj_dir}. "
            "Download trajectories_txt.zip from ped.fz-juelich.de/da "
            "(DOI: 10.34735/ped.2018.1) and extract into that directory."
        )

    frames = []
    for sid, path in enumerate(txt_files):
        print(f"[loader]  {sid+1:2d}/{len(txt_files)}  {path.name}", flush=True)
        frames.append(load_run(path, scenario_id=sid))

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["scenario", "t"])
    os.makedirs(cache_path.parent, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"[loader] Saved {len(df):,} rows -> {cache_path}")
    return df


# ── CLI convenience ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    force = "--force" in sys.argv
    df = load_all_runs(force=force)
    print(df.head())
    print("Shape:", df.shape)
    print("Runs with danger events:", df.groupby("scenario")["t_event"].first().notna().sum(),
          "of", df["scenario"].nunique())
