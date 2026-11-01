"""Lead time, false alarms per hour and missed events for one scoring method.

Rules (per scenario, with warning window H):
- Event with first alert in [t_event - H, t_event]  -> detected, lead = t_event - t_alert
- Event with no alert in that window                  -> missed
- Alert episode with no event after it within H       -> false alarm
  (episodes after the event has begun are ignored: the danger already happened)
"""
import numpy as np

from alert.gate import alert_times


def evaluate(scenarios, score_col, threshold, hold_s=3.0, window_s=40.0):
    """scenarios: iterable of DataFrames (one per run) with columns t, t_event, `score_col`."""
    leads, missed, events, false_alarms, hours = [], 0, 0, 0, 0.0
    for df in scenarios:
        t = df["t"].to_numpy()
        hours += (t[-1] - t[0]) / 3600.0
        t_event = df["t_event"].iloc[0]
        alerts = alert_times(t, df[score_col].to_numpy(), threshold, hold_s)
        if np.isnan(t_event):
            false_alarms += len(alerts)
            continue
        events += 1
        in_window = [a for a in alerts if t_event - window_s <= a <= t_event]
        false_alarms += sum(1 for a in alerts if a < t_event - window_s)
        if in_window:
            leads.append(t_event - in_window[0])
        else:
            missed += 1
    return {"threshold": threshold, "events": events, "detected": len(leads), "missed": missed,
            "mean_lead_s": float(np.mean(leads)) if leads else np.nan,
            "false_alarms": false_alarms, "hours": hours,
            "false_alarms_per_hour": false_alarms / hours if hours else np.nan}


def sweep(scenarios, score_col, thresholds, **kw):
    return [evaluate(scenarios, score_col, th, **kw) for th in thresholds]
