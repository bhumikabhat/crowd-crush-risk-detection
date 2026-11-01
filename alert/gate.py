"""Alert gate: fire when the score stays at or above a threshold for `hold_s` seconds."""
import numpy as np


def alert_times(t, score, threshold, hold_s=3.0):
    """Start times of alert episodes. Re-arms once the score drops below threshold."""
    t = np.asarray(t)
    above = np.asarray(score) >= threshold
    alerts, run_start, fired = [], None, False
    for ti, a in zip(t, above):
        if not a:
            run_start, fired = None, False
            continue
        if run_start is None:
            run_start = ti
        if not fired and ti - run_start >= hold_s:
            alerts.append(ti)
            fired = True
    return alerts
