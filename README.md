# Flow-aware crush-risk index

Video-only early warning for crowd crush: density + optical-flow entropy + slowdown in one score,
compared with a density-only alarm on JuPedSim bottleneck simulations with exact ground truth.

## Run
```
pip install opencv-python-headless numpy scipy scikit-learn pandas matplotlib shapely jupedsim
python run_sim_features.py 0,1,2,3     # 120 scenarios -> data/features.csv (~15 min on 2 cores)
python -m eval.run_eval                # train seeds 0,1 / test seeds 2,3 -> data/results.csv, data/operating_curves.png
python -m eval.ablation                # which signals help
```

## Layout
- `sim/bottleneck.py`: JuPedSim hall + door, renders frames, true zone density
- `features/extract.py`: video-only density proxy, optical-flow entropy, slowdown (OpenCV Farneback)
- `risk/score.py`: `DensityBaseline`, `FlowAwareRisk` (logistic regression weights)
- `alert/gate.py`: threshold + hold-time alert gate
- `eval/lead_time.py`: lead time, false alarms per hour, missed events
- `eval/run_eval.py`, `eval/ablation.py`: experiments

## Setup
Door width 0.8–2.5 m, 60–200 agents, 4 seeds, 80 s each. Danger event: true density >= 3.5 people/m2
in the 3.5 m x 5 m zone in front of the door for 2 s. Warning window 15 s, gate hold 3 s.
Thresholds are chosen on training runs (false alarms <= 10/h) and reported on held-out runs.

## Results (held-out test, 60 runs, 25 danger events, 1.33 h of video)
| method | detected | missed | mean lead (s) | false alarms/h |
|---|---|---|---|---|
| density-only | 21 | 4 | 2.35 | 10.5 |
| flow-aware | 23 | 2 | 3.50 | 7.5 |

Ablation (same protocol): D 21/25, D+S 22/25, D+H 23/25 (3.49 s), D+H+S 23/25 (3.50 s).

## Caveats
- Small gain: 2 more events caught, ~1 s more lead, at strict false-alarm budgets. The operating curves
  overlap above ~15 false alarms/h. One train/test split, no confidence intervals yet.
- Entropy gets a NEGATIVE learned weight. JuPedSim's collision-free-speed model has no pushing forces, so it
  does not produce crowd turbulence; here low entropy (orderly compression toward the door) marks risk.
  Test on real bottleneck footage before claiming that high entropy warns of crush.
- Slowdown adds nothing beyond density + entropy in this simulator.
- Density is a foreground-area proxy that works on clean rendered video. Replace with a pretrained counter
  (e.g. DM-Count) for real footage.
- Not yet done: public footage, edge-latency benchmark, robustness tests.
