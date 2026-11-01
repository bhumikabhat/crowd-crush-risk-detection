"""Hypothesis tests on the Wuppertal 2018 real-data trajectories.

NOTE ON DATA SOURCE:
  Features are derived from ground-truth pedestrian trajectories (PeTrack tracking,
  25 fps), NOT from video pixel analysis. density/entropy/slowdown come from exact
  x,y positions and per-frame velocities.  There is NO camera noise or optical-flow
  estimation in this data.

NOTE ON LEAD TIME:
  Lead time cannot be measured on these runs. In all 5 runs where a danger event
  was detected (density >= 3.5 p/m2 for >= 2 s), the event starts within 0.84-3.84 s
  of t=0. The crowd is pre-loaded into the corridor before the 'Go' signal; there is
  effectively no pre-event build-up phase to measure advance warning over.

Analysis:
  (a) Per-run summary statistics for frames where zone density >= THRESH_LO (2.0 p/m2)
      and >= THRESH_HI (3.5 p/m2): mean entropy, mean slowdown, mean speed.
  (b) Paired Wilcoxon test across matched run pairs (same width+N, different
      motivation: h0 = high, h-/h+ = low) to test whether high-motivation runs
      have higher entropy.

Run from the project root:  python -m eval.real_hypothesis
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ── constants ────────────────────────────────────────────────────────────────
THRESH_LO = 2.0   # p/m²
THRESH_HI = 3.5   # p/m²

# Matched pairs: (high-motivation scenario id, low-motivation scenario id)
# Filenames: 010_c_12_h0 / 020_c_12_h+, 030_c_56_h0 / 040_c_56_h-, etc.
# scenario IDs are assigned by sorted filename order (0-indexed):
#   010→0, 020→1, 030→2, 040→3, 050→4, 060→5, 070→6, 080→7,
#   090→8, 100→9, 110→10, 120→11, 150→12, 160→13, 170→14, 180→15,
#   190→16, 200→17, 230→18, 240→19, 250→20, 260→21, 270→22, 280→23
PAIRS = [
    (0,  1,  "010/020", 1.2, 11),
    (2,  3,  "030/040", 5.6, 75),
    (4,  5,  "050/060", 4.5, 42),
    (6,  7,  "070/080", 2.3, 20),
    (8,  9,  "090/100", 1.2, 24),
    (10, 11, "110/120", 1.2, 63),
    (12, 13, "150/160", 5.6, 57),
    (14, 15, "170/180", 1.2, 25),
    (16, 17, "190/200", 3.4, 22),
    (18, 19, "230/240", 2.3, 42),
    (20, 21, "250/260", 4.5, 42),
    (22, 23, "270/280", 3.4, 67),
]

FEAT_PATH = "data/real/features_real.csv"
OUT_CSV   = "data/real/hypothesis.csv"

NOTE = (
    "Features are trajectory-derived (ground-truth positions, NOT video/optical-flow). "
    "Lead time cannot be measured: all danger events start within 4 s of run start "
    "(crowd is pre-loaded; no build-up phase exists)."
)


def run_summary(df: pd.DataFrame, sid: int, thresh: float) -> dict:
    """Mean feature values over frames where density_true >= thresh for one run."""
    sub = df[(df["scenario"] == sid) & (df["density_true"] >= thresh)]
    if sub.empty:
        return dict(n_frames=0, mean_entropy=float("nan"),
                    mean_slowdown=float("nan"), mean_speed=float("nan"))
    return dict(
        n_frames=len(sub),
        mean_entropy=sub["entropy"].mean(),
        mean_slowdown=sub["slowdown"].mean(),
        mean_speed=sub["speed"].mean(),
    )


def main():
    print(f"\nNOTE: {NOTE}\n")

    df = pd.read_csv(FEAT_PATH)
    print(f"Loaded {len(df):,} frames from {df['scenario'].nunique()} runs  ({FEAT_PATH})")
    print(f"Data source: trajectory-derived features (25 fps ground-truth tracking)\n")

    rows = []

    # ── (a) Per-run summary at both density thresholds ────────────────────────
    for thresh_label, thresh in [(">=2.0 p/m2", THRESH_LO), (">=3.5 p/m2", THRESH_HI)]:
        for hi_id, lo_id, pair_name, door_w, n_agents in PAIRS:
            for role, sid in [("h0_high", hi_id), ("h_low", lo_id)]:
                s = run_summary(df, sid, thresh)
                rows.append({
                    "pair": pair_name,
                    "door_w": door_w,
                    "n_agents": n_agents,
                    "motivation": role,
                    "scenario_id": sid,
                    "density_thresh": thresh_label,
                    **s,
                })

    result_df = pd.DataFrame(rows)

    # ── (b) Paired Wilcoxon tests ─────────────────────────────────────────────
    print("=" * 70)
    print("(a) Feature summary over frames with zone density above threshold")
    print("    (trajectory-derived, NOT video-derived)")
    print("=" * 70)

    for thresh_label, thresh in [(">=2.0 p/m2", THRESH_LO), (">=3.5 p/m2", THRESH_HI)]:
        sub = result_df[result_df["density_thresh"] == thresh_label]
        hi  = sub[sub["motivation"] == "h0_high"].set_index("pair")
        lo  = sub[sub["motivation"] == "h_low"].set_index("pair")

        print(f"\nDensity threshold: {thresh_label}")
        print(f"{'pair':12s} {'door_w':>7s} {'N':>5s}  "
              f"{'hi_ent':>8s} {'lo_ent':>8s}  "
              f"{'hi_slow':>8s} {'lo_slow':>8s}  "
              f"{'hi_spd':>7s} {'lo_spd':>7s}  "
              f"{'ent_hi>lo':>10s}")
        print("-" * 100)

        ent_hi_vals, ent_lo_vals = [], []
        for pair_name, door_w, n_agents in [(p[2], p[3], p[4]) for p in PAIRS]:
            if pair_name not in hi.index or pair_name not in lo.index:
                continue
            h = hi.loc[pair_name]
            l = lo.loc[pair_name]
            hi_ent  = h["mean_entropy"]
            lo_ent  = l["mean_entropy"]
            hi_slow = h["mean_slowdown"]
            lo_slow = l["mean_slowdown"]
            hi_spd  = h["mean_speed"]
            lo_spd  = l["mean_speed"]
            better  = "YES" if (not np.isnan(hi_ent) and not np.isnan(lo_ent)
                                and hi_ent > lo_ent) else "NO"
            print(f"{pair_name:12s} {door_w:>7.1f} {n_agents:>5d}  "
                  f"{hi_ent:>8.4f} {lo_ent:>8.4f}  "
                  f"{hi_slow:>8.4f} {lo_slow:>8.4f}  "
                  f"{hi_spd:>7.4f} {lo_spd:>7.4f}  "
                  f"{better:>10s}")
            if not np.isnan(hi_ent) and not np.isnan(lo_ent):
                ent_hi_vals.append(hi_ent)
                ent_lo_vals.append(lo_ent)

        # Paired Wilcoxon signed-rank test on entropy (hi vs lo motivation)
        if len(ent_hi_vals) >= 2:
            diffs = np.array(ent_hi_vals) - np.array(ent_lo_vals)
            nonzero = diffs[diffs != 0]
            if len(nonzero) >= 2:
                stat, pval = wilcoxon(ent_hi_vals, ent_lo_vals, alternative="greater")
            else:
                stat, pval = float("nan"), float("nan")
            n_better = sum(h > l for h, l in zip(ent_hi_vals, ent_lo_vals))
            print(f"\n  Paired Wilcoxon (entropy h0 > h-): "
                  f"stat={stat:.3f}, p={pval:.4f}, "
                  f"pairs where hi>lo: {n_better}/{len(ent_hi_vals)}")
        else:
            print("  (Insufficient non-NaN pairs for Wilcoxon test)")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    result_df.round(5).to_csv(OUT_CSV, index=False)

    print(f"\n{'=' * 70}")
    print(f"Saved {OUT_CSV}  ({len(result_df)} rows)")
    print(f"\nNOTE: {NOTE}")


if __name__ == "__main__":
    main()
