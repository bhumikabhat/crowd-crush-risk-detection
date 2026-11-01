"""Hypothesis tests on the Wuppertal 2018 real-data trajectories.

===========================================================================
MOTIVATION LABELS  (from the archive docs: DOI 10.34735/ped.2018.1)
===========================================================================
  h0  "baseline" high motivation: participants imagine they are at a concert
      and want to be close to the stage — they try to enter as fast as
      possible.  This is the reference high-urgency condition.

  h+  "increased" motivation: used ONLY in the very first pair of runs
      (010 / 020). Run 020 was intended to be even more urgent than h0:
      participants were told they absolutely need to be in the front row.

  h-  "reduced" motivation: participants are told everyone will have a good
      view, so they still want to enter quickly but feel less urgency.
      This is the reference low-urgency condition.

  Priming: 'c' = crowding scenario (picture of dense crowd shown on consent
           form); 'q' = queuing scenario (picture of orderly queue shown).
           Priming affects social norm awareness, not urgency per se.

===========================================================================
PAIR 010 / 020 (h0 vs h+)
===========================================================================
  Run 010 is the high-motivation baseline (h0); run 020 is the increased-
  motivation variant (h+), NOT the low-motivation condition. This pair does
  NOT fit the h0-vs-h- comparison used for all other pairs.
  Treatment here: 010/020 is INCLUDED in the pair table for completeness
  and transparency, with motivation labels shown explicitly. It is EXCLUDED
  from the Wilcoxon test (which tests h0 > h- only) because h+ is not h-.

===========================================================================
DATA SOURCE
===========================================================================
  Features are derived from ground-truth pedestrian trajectories (PeTrack
  tracking, 25 fps), NOT from video pixel analysis. density, entropy, and
  slowdown come from exact (x, y) positions and per-frame velocity vectors.
  There is NO camera noise or optical-flow estimation.

===========================================================================
LEAD TIME
===========================================================================
  Lead time cannot be measured on these runs. In all 5 runs where a danger
  event was detected (zone density >= 3.5 p/m2 sustained for >= 2 s), the
  event starts within 0.84-3.84 s of t=0. The crowd is pre-loaded into the
  corridor before the 'Go' signal; there is no pre-event build-up phase.

===========================================================================
ANALYSIS
===========================================================================
  (a) Per-run feature summary: for each run, compute mean entropy, mean
      slowdown, and mean speed over frames where zone density >= threshold.
      Two thresholds: 2.0 p/m2 (crowd present) and 3.5 p/m2 (danger zone).

  (b) Paired Wilcoxon signed-rank test (one-sided: h0 > h-): for each
      valid pair (same width, same N, h0 vs h-), compare mean entropy.
      Pair 010/020 (h0 vs h+) is excluded from the test.

Run from the project root:  python -m eval.real_hypothesis
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ── constants ────────────────────────────────────────────────────────────────
THRESH_LO = 2.0   # p/m²  — crowd present
THRESH_HI = 3.5   # p/m²  — danger zone

# Matched pairs: (sid_a, sid_b, run_a, run_b, motiv_a, motiv_b, door_w, N)
# Scenario IDs from sorted filename order (0-indexed):
#   010→0, 020→1, 030→2, 040→3, 050→4, 060→5, 070→6, 080→7,
#   090→8, 100→9, 110→10, 120→11, 150→12, 160→13, 170→14, 180→15,
#   190→16, 200→17, 230→18, 240→19, 250→20, 260→21, 270→22, 280→23
#
# motiv_a / motiv_b use archive labels: h0, h+, h-
# in_wilcoxon=False for 010/020 because h+ is not h-
PAIRS = [
    # sid_a, sid_b, run_a,   run_b,   motiv_a, motiv_b, door_w, N,  in_wilcoxon
    (0,   1,  "010",   "020",  "h0",    "h+",   1.2,  11,  False),
    (2,   3,  "030",   "040",  "h0",    "h-",   5.6,  75,  True),
    (4,   5,  "050",   "060",  "h0",    "h-",   4.5,  42,  True),
    (6,   7,  "070",   "080",  "h0",    "h-",   2.3,  20,  True),
    (8,   9,  "090",   "100",  "h0",    "h-",   1.2,  24,  True),
    (10, 11,  "110",   "120",  "h0",    "h-",   1.2,  63,  True),
    (12, 13,  "150",   "160",  "h0",    "h-",   5.6,  57,  True),
    (14, 15,  "170",   "180",  "h0",    "h-",   1.2,  25,  True),
    (16, 17,  "190",   "200",  "h0",    "h-",   3.4,  22,  True),
    (18, 19,  "230",   "240",  "h0",    "h-",   2.3,  42,  True),
    (20, 21,  "250",   "260",  "h0",    "h-",   4.5,  42,  True),
    (22, 23,  "270",   "280",  "h0",    "h-",   3.4,  67,  True),
]

FEAT_PATH = "data/real/features_real.csv"
OUT_CSV   = "data/real/hypothesis.csv"


def run_stats(df: pd.DataFrame, sid: int, thresh: float) -> dict:
    """Mean feature values over frames where density_true >= thresh."""
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


def main() -> None:
    df = pd.read_csv(FEAT_PATH)

    # Print header with legend and notes
    print("=" * 72)
    print("Wuppertal 2018 bottleneck — real trajectory hypothesis test")
    print("=" * 72)
    print(__doc__)  # full module docstring contains the legend

    print(f"Loaded {len(df):,} frames from {df['scenario'].nunique()} runs ({FEAT_PATH})")
    print("Data source: trajectory-derived features (25 fps PeTrack ground truth)")
    print()

    csv_rows = []

    for thresh_label, thresh in [(">=2.0 p/m2", THRESH_LO), (">=3.5 p/m2", THRESH_HI)]:
        print("=" * 72)
        print(f"Density threshold: {thresh_label}")
        print("(trajectory-derived, NOT video-derived)")
        print()
        print(
            f"{'run_a':>5s} {'run_b':>5s} {'motiv_a':>7s} {'motiv_b':>7s} "
            f"{'door_w':>7s} {'N':>4s} "
            f"{'n_a':>5s} {'n_b':>5s} "
            f"{'ent_a':>7s} {'ent_b':>7s} "
            f"{'slow_a':>7s} {'slow_b':>7s} "
            f"{'spd_a':>6s} {'spd_b':>6s} "
            f"{'ent_a>b':>8s} {'in_test':>7s}"
        )
        print("-" * 120)

        ent_a_vals, ent_b_vals = [], []   # only h0 vs h- pairs

        for sid_a, sid_b, run_a, run_b, motiv_a, motiv_b, door_w, N, in_wil in PAIRS:
            sa = run_stats(df, sid_a, thresh)
            sb = run_stats(df, sid_b, thresh)

            ent_a = sa["mean_entropy"]
            ent_b = sb["mean_entropy"]
            na, nb = sa["n_frames"], sb["n_frames"]

            direction = (
                "YES" if (not np.isnan(ent_a) and not np.isnan(ent_b) and ent_a > ent_b)
                else "NO"
            )

            print(
                f"{run_a:>5s} {run_b:>5s} {motiv_a:>7s} {motiv_b:>7s} "
                f"{door_w:>7.1f} {N:>4d} "
                f"{na:>5d} {nb:>5d} "
                f"{ent_a:>7.4f} {ent_b:>7.4f} "
                f"{sa['mean_slowdown']:>7.4f} {sb['mean_slowdown']:>7.4f} "
                f"{sa['mean_speed']:>6.4f} {sb['mean_speed']:>6.4f} "
                f"{'n/a' if np.isnan(ent_a) or np.isnan(ent_b) else direction:>8s} "
                f"{'YES' if in_wil else 'NO (h+)':>7s}"
            )

            # Collect for Wilcoxon (h0 vs h- only, both must be non-NaN)
            if in_wil and not np.isnan(ent_a) and not np.isnan(ent_b):
                ent_a_vals.append(ent_a)
                ent_b_vals.append(ent_b)

            # CSV row
            for role, sid, run_id, motiv, stats in [
                ("a", sid_a, run_a, motiv_a, sa),
                ("b", sid_b, run_b, motiv_b, sb),
            ]:
                csv_rows.append({
                    "pair":          f"{run_a}/{run_b}",
                    "role":          role,
                    "run_id":        run_id,
                    "motivation":    motiv,
                    "door_w":        door_w,
                    "n_agents":      N,
                    "in_wilcoxon":   in_wil,
                    "scenario_id":   sid_a if role == "a" else sid_b,
                    "density_thresh": thresh_label,
                    **stats,
                })

        # Wilcoxon test
        n_pairs = len(ent_a_vals)
        n_better = sum(a > b for a, b in zip(ent_a_vals, ent_b_vals))
        print()
        if n_pairs >= 2:
            diffs   = np.array(ent_a_vals) - np.array(ent_b_vals)
            nonzero = diffs[diffs != 0]
            if len(nonzero) >= 2:
                stat, pval = wilcoxon(ent_a_vals, ent_b_vals, alternative="greater")
                print(
                    f"  Paired Wilcoxon signed-rank test  (H1: entropy h0 > h-)\n"
                    f"  Valid h0-vs-h- pairs (both non-NaN): {n_pairs}\n"
                    f"  Pairs where entropy(h0) > entropy(h-): {n_better}/{n_pairs}\n"
                    f"  stat = {stat:.3f},  p = {pval:.4f}  (one-sided, greater)"
                )
            else:
                print(
                    f"  Wilcoxon skipped: only {len(nonzero)} non-zero difference(s) "
                    f"among {n_pairs} valid pairs."
                )
        else:
            print(f"  Wilcoxon skipped: only {n_pairs} valid pair(s) (need >= 2).")
        print()

    # Save CSV
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    pd.DataFrame(csv_rows).round(5).to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}  ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
