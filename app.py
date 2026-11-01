"""Streamlit app — crowd-crush risk detection demo.

Two tabs:
  Tab 1  Live replay  — run simulation, score every frame, show alerts
  Tab 2  Results      — show eval tables and operating-curve chart

Run from project root:
    streamlit run app.py
"""
from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd
import streamlit as st

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crowd-Crush Risk Detection",
    page_icon=":warning:",
    layout="wide",
)

FOOTER = "Research prototype on simulated crowds, not a safety system."

# ── constants matching eval/run_eval.py exactly ──────────────────────────────
FPS        = 10
SMOOTH_S   = 2.0
WINDOW_S   = 15.0
HOLD_S     = 3.0
TRAIN_SEEDS = {0, 1}
THR_DENSITY = 0.525
THR_FLOW    = 0.825


# ── cached model (train once) ────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Train DensityBaseline and FlowAwareRisk exactly as run_eval.py does."""
    from risk.score import DensityBaseline, FlowAwareRisk

    df = pd.read_csv("data/features.csv")
    for c in ["density", "entropy", "slowdown"]:
        df[c] = df.groupby("scenario")[c].transform(
            lambda s: s.rolling(int(SMOOTH_S * FPS), min_periods=1).mean()
        )
    df["y"] = (
        (df["t"] >= df["t_event"] - WINDOW_S) & (df["t"] < df["t_event"])
    ).astype(int)
    train    = df[df["seed"].isin(TRAIN_SEEDS)]
    fit_rows = train[~(train["t"] >= train["t_event"])]
    baseline = DensityBaseline()
    risk     = FlowAwareRisk().fit(fit_rows, fit_rows["y"].to_numpy())
    return baseline, risk


# ── cached simulation ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_simulation(door_w: float, n_agents: int, seed: int):
    """Run the simulation and extract features. Cached by (door_w, n_agents, seed)."""
    from alert.gate import alert_times
    from features.extract import FeatureExtractor
    from run_sim_features import event_time, DANGER_DENSITY, DANGER_HOLD_S
    from sim.bottleneck import PX_PER_M, ZONE_PX, Scenario, run

    sc  = Scenario(door_w=door_w, n_agents=n_agents, seed=seed, duration=80.0)
    ext = FeatureExtractor(ZONE_PX, PX_PER_M, sc.fps)

    ts, frames_jpg, d_true_list = [], [], []
    density_raw, entropy_raw, slowdown_raw = [], [], []

    for t, frame, d_true in run(sc):
        f = ext.update(frame)
        ts.append(t)
        # encode frame as JPEG bytes to save memory
        import cv2
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frames_jpg.append(bytes(buf) if ok else b"")
        d_true_list.append(d_true)
        density_raw.append(f["density"])
        entropy_raw.append(f["entropy"])
        slowdown_raw.append(f["slowdown"])

    t_arr  = np.array(ts)
    d_true = np.array(d_true_list)

    # Smooth signals (causal rolling mean, same window as run_eval.py)
    win = int(SMOOTH_S * sc.fps)

    def _smooth(x):
        s = pd.Series(x)
        return s.rolling(win, min_periods=1).mean().to_numpy()

    density_s  = _smooth(density_raw)
    entropy_s  = _smooth(entropy_raw)
    slowdown_s = _smooth(slowdown_raw)

    # Event time from true density (same rule as run_sim_features.py)
    t_event = event_time(t_arr, d_true, sc.fps)

    return dict(
        t=t_arr,
        frames_jpg=frames_jpg,
        d_true=d_true,
        density=density_s,
        entropy=entropy_s,
        slowdown=slowdown_s,
        t_event=t_event,
        fps=sc.fps,
    )


def _score_run(sim: dict, baseline, risk) -> tuple[np.ndarray, np.ndarray]:
    """Score every frame with both models (no smoothing needed; sim is pre-smoothed)."""
    row_df = pd.DataFrame({
        "density":  sim["density"],
        "entropy":  sim["entropy"],
        "slowdown": sim["slowdown"],
    })
    sc_density = baseline.score(row_df)
    sc_flow    = risk.score(row_df)
    return sc_density, sc_flow


def _alerts(t: np.ndarray, scores: np.ndarray, thr: float) -> list[float]:
    from alert.gate import alert_times
    return alert_times(t, scores, thr, hold_s=HOLD_S)


def _status_at(t_idx: int, t: np.ndarray, scores: np.ndarray,
                thr: float, alerts: list[float]) -> str:
    """Return 'NORMAL', 'WATCH', or 'ALERT' for the frame at t_idx."""
    t_cur = t[t_idx]
    if any(a <= t_cur for a in alerts):
        return "ALERT"
    if scores[t_idx] >= thr:
        return "WATCH"
    return "NORMAL"


STATUS_COLOR = {"NORMAL": "green", "WATCH": "orange", "ALERT": "red"}


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — Live replay
# ═══════════════════════════════════════════════════════════════════════════════
def tab_live():
    # Sidebar controls
    with st.sidebar:
        st.header("Simulation parameters")
        door_w   = st.slider("Door width (m)",   min_value=0.8, max_value=2.5,
                             value=1.2, step=0.1)
        n_agents = st.slider("Number of agents", min_value=60,  max_value=200,
                             value=140, step=10)
        _SEED_LABELS = {
            "0 (train)": 0,
            "1 (train)": 1,
            "2 (held-out test)": 2,
            "3 (held-out test)": 3,
        }
        seed_label = st.selectbox("Seed", list(_SEED_LABELS.keys()), index=3)
        seed       = _SEED_LABELS[seed_label]
        run_btn  = st.button("Run simulation", type="primary", width='stretch')

    if run_btn or "sim_result" not in st.session_state:
        with st.spinner("Running simulation…"):
            st.session_state["sim_result"] = run_simulation(door_w, n_agents, seed)
            st.session_state["sim_params"] = (door_w, n_agents, seed)

    sim = st.session_state.get("sim_result")
    if sim is None:
        st.info("Set parameters in the sidebar and click **Run simulation**.")
        st.caption(FOOTER)
        return

    baseline, risk = load_models()
    sc_den, sc_flow = _score_run(sim, baseline, risk)

    t       = sim["t"]
    t_event = sim["t_event"]
    alerts_den  = _alerts(t, sc_den,  THR_DENSITY)
    alerts_flow = _alerts(t, sc_flow, THR_FLOW)

    # Time slider
    frame_idx = st.slider(
        "Time (s)", min_value=0.0, max_value=float(t[-1]),
        value=0.0, step=1.0 / sim["fps"], format="%.1f s"
    )
    idx = int(np.searchsorted(t, frame_idx, side="right") - 1)
    idx = max(0, min(idx, len(t) - 1))

    # Layout: frame | metrics | status
    col_frame, col_info = st.columns([3, 2])

    with col_frame:
        jpg = sim["frames_jpg"][idx]
        if jpg:
            st.image(jpg, caption=f"t = {t[idx]:.1f} s", width='stretch')

    with col_info:
        st.subheader("Sensor readings")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Density (p/m²)", f"{sim['density'][idx]:.2f}")
        mc2.metric("Entropy",        f"{sim['entropy'][idx]:.2f}")
        mc3.metric("Slowdown",       f"{sim['slowdown'][idx]:.2f}")

        st.subheader("Alert status")
        for label, scores, thr, alerts in [
            ("Density-only", sc_den,  THR_DENSITY, alerts_den),
            ("Flow-aware",   sc_flow, THR_FLOW,    alerts_flow),
        ]:
            status = _status_at(idx, t, scores, thr, alerts)
            color  = STATUS_COLOR[status]
            st.markdown(
                f"**{label}** &nbsp; :{color}[**{status}**]"
                f"&ensp; score {scores[idx]:.3f} / thr {thr}"
            )

    # Score chart
    st.subheader("Risk scores over time")
    chart_df = pd.DataFrame({
        "density-only": sc_den,
        "flow-aware":   sc_flow,
        "thr_density":  THR_DENSITY,
        "thr_flow":     THR_FLOW,
    }, index=t)

    import altair as alt

    base = alt.Chart(chart_df.reset_index().rename(columns={"index": "t"}))

    # Melt scores for colour encoding
    scores_long = chart_df[["density-only", "flow-aware"]].reset_index()
    scores_long = scores_long.rename(columns={"index": "t"})
    scores_long = scores_long.melt("t", var_name="method", value_name="score")

    line_chart = (
        alt.Chart(scores_long)
        .mark_line()
        .encode(
            x=alt.X("t:Q", title="Time (s)"),
            y=alt.Y("score:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("method:N"),
            tooltip=["t:Q", "method:N", alt.Tooltip("score:Q", format=".3f")],
        )
    )

    # Threshold rules
    rules_data = pd.DataFrame([
        {"thr": THR_DENSITY, "method": "density-only thr"},
        {"thr": THR_FLOW,    "method": "flow-aware thr"},
    ])
    thr_rules = (
        alt.Chart(rules_data)
        .mark_rule(strokeDash=[4, 4], opacity=0.6)
        .encode(y="thr:Q", color=alt.Color("method:N", legend=None))
    )

    # Alert vertical rules
    alert_rows = []
    for a_t in alerts_den:
        alert_rows.append({"t": a_t, "method": "density-only alert"})
    for a_t in alerts_flow:
        alert_rows.append({"t": a_t, "method": "flow-aware alert"})
    if not np.isnan(t_event):
        alert_rows.append({"t": float(t_event), "method": "danger event"})

    layers = [line_chart, thr_rules]
    if alert_rows:
        alert_df = pd.DataFrame(alert_rows)
        vert = (
            alt.Chart(alert_df)
            .mark_rule(strokeWidth=2)
            .encode(
                x="t:Q",
                color=alt.Color("method:N", scale=alt.Scale(
                    domain=["density-only alert", "flow-aware alert", "danger event"],
                    range=["#dd8452", "#4c72b0", "#c44e52"],
                )),
                tooltip=["t:Q", "method:N"],
            )
        )
        layers.append(vert)

    # Current-time marker
    cur_df = pd.DataFrame({"t": [float(t[idx])]})
    cur_rule = (
        alt.Chart(cur_df)
        .mark_rule(color="gray", strokeDash=[2, 2], opacity=0.7)
        .encode(x="t:Q")
    )
    layers.append(cur_rule)

    st.altair_chart(alt.layer(*layers).properties(height=260), width='stretch')

    # Summary text line
    def _fmt_time(lst, label="none"):
        return f"{lst[0]:.1f} s" if lst else label
    fa_str   = _fmt_time(alerts_flow)
    den_str  = _fmt_time(alerts_den)
    ev_str   = f"{t_event:.1f} s" if not np.isnan(t_event) else "none"
    st.info(
        f"Flow-aware alert at **{fa_str}** | "
        f"density-only alert at **{den_str}** | "
        f"danger at **{ev_str}**"
    )

    st.caption(FOOTER)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — Results
# ═══════════════════════════════════════════════════════════════════════════════
def tab_results():
    st.header("Evaluation results")
    st.caption(
        "Simulated crowds (JuPedSim). 120 runs. "
        "Gain is modest; repeated-split ranges shown below."
    )

    # results.csv
    if os.path.exists("data/results.csv"):
        st.subheader("Test-set results (fixed split: train seeds 0-1, test seeds 2-3)")
        st.dataframe(pd.read_csv("data/results.csv").round(3), width='stretch')
    else:
        st.warning("data/results.csv not found — run `python -m eval.run_eval` first.")

    # operating curves
    if os.path.exists("data/operating_curves.png"):
        st.subheader("Operating curves (test set)")
        st.image("data/operating_curves.png", width='stretch')
    else:
        st.warning("data/operating_curves.png not found.")

    # repeat_summary.csv
    if os.path.exists("data/repeat_summary.csv"):
        st.subheader("Stability across 200 random + 6 seed splits")
        st.dataframe(pd.read_csv("data/repeat_summary.csv").round(3), width='stretch')
    else:
        st.warning("data/repeat_summary.csv not found — run `python -m eval.repeat_eval` first.")

    st.caption(FOOTER)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["Live replay", "Results"])
with tab1:
    tab_live()
with tab2:
    tab_results()
