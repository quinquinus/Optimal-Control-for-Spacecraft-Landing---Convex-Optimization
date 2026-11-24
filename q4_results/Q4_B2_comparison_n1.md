# Question 4: Benchmark B2 (n=1 seeds per condition)

All three controllers were evaluated on the same five conditions and frozen seeds.
Fuel = mean seconds of full main-engine thrust among successful episodes.
Time = mean computation time per controller step in milliseconds.

## Overall

| Policy | Success | Mean score | Fuel (s, successes) | Time (ms/step) |
|---|---:|---:|---:|---:|
| LCvx cascade | 100.0% | 90.74 | 1.85 | 1.80 |
| Heuristic | 100.0% | 93.52 | 1.30 | 0.03 |
| PPO | 100.0% | 83.48 | 3.30 | 0.58 |

## By environment

| Environment | Policy | Success | Fuel (s, successes) | Time (ms/step) | Off-pad | Crash/hard |
|---|---|---:|---:|---:|---:|---:|
| nominal | LCvx cascade | 100.0% | 2.08 | 1.77 | 0.0% | 0.0% |
| nominal | Heuristic | 100.0% | 1.32 | 0.03 | 0.0% | 0.0% |
| nominal | PPO | 100.0% | 3.11 | 0.68 | 0.0% | 0.0% |
| heavy | LCvx cascade | 100.0% | 2.19 | 2.22 | 0.0% | 0.0% |
| heavy | Heuristic | 100.0% | 1.41 | 0.03 | 0.0% | 0.0% |
| heavy | PPO | 100.0% | 2.74 | 0.72 | 0.0% | 0.0% |
| light | LCvx cascade | 100.0% | 0.97 | 1.90 | 0.0% | 0.0% |
| light | Heuristic | 100.0% | 0.97 | 0.03 | 0.0% | 0.0% |
| light | PPO | 100.0% | 4.14 | 0.53 | 0.0% | 0.0% |
| gusty | LCvx cascade | 100.0% | 2.12 | 1.76 | 0.0% | 0.0% |
| gusty | Heuristic | 100.0% | 1.40 | 0.03 | 0.0% | 0.0% |
| gusty | PPO | 100.0% | 3.20 | 0.52 | 0.0% | 0.0% |
| storm | LCvx cascade | 100.0% | 1.89 | 1.34 | 0.0% | 0.0% |
| storm | Heuristic | 100.0% | 1.37 | 0.03 | 0.0% | 0.0% |
| storm | PPO | 100.0% | 3.33 | 0.47 | 0.0% | 0.0% |

## Interpretation guide

Success depends on the combined feasibility of convex guidance and the rigid-body attitude loop.
Higher replanning time is the cost of repeatedly solving a DPP SOCP; look at mean latency alongside reliability.
Fuel is measured only among successful episodes; do not compare it in isolation from the success rate.
The light-gravity case additionally uses relaxed engine-off planning with sigma-delta thrust pulses.
