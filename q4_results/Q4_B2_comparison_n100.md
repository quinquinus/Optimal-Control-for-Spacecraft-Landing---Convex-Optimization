# Question 4: Benchmark B2 (n=100 seeds per condition)

All three controllers were evaluated on the same five conditions and frozen seeds.
Fuel = mean seconds of full main-engine thrust among successful episodes.
Time = mean computation time per controller step in milliseconds.

## Overall

| Policy | Success | Mean score | Fuel (s, successes) | Time (ms/step) |
|---|---:|---:|---:|---:|
| LCvx cascade | 99.0% | 86.70 | 2.50 | 1.84 |
| Heuristic | 78.2% | 71.50 | 1.79 | 0.03 |
| PPO | 69.8% | 58.73 | 3.41 | 0.64 |

## By environment

| Environment | Policy | Success | Fuel (s, successes) | Time (ms/step) | Off-pad | Crash/hard |
|---|---|---:|---:|---:|---:|---:|
| nominal | LCvx cascade | 100.0% | 2.40 | 1.82 | 0.0% | 0.0% |
| nominal | Heuristic | 96.0% | 1.77 | 0.03 | 4.0% | 0.0% |
| nominal | PPO | 93.0% | 3.33 | 0.69 | 5.0% | 0.0% |
| heavy | LCvx cascade | 100.0% | 2.62 | 2.31 | 0.0% | 0.0% |
| heavy | Heuristic | 48.0% | 1.76 | 0.03 | 19.0% | 33.0% |
| heavy | PPO | 82.0% | 2.87 | 0.67 | 16.0% | 0.0% |
| light | LCvx cascade | 100.0% | 2.33 | 1.93 | 0.0% | 0.0% |
| light | Heuristic | 96.0% | 1.34 | 0.04 | 4.0% | 0.0% |
| light | PPO | 100.0% | 3.97 | 0.67 | 0.0% | 0.0% |
| gusty | LCvx cascade | 97.0% | 2.43 | 1.75 | 3.0% | 0.0% |
| gusty | Heuristic | 86.0% | 1.86 | 0.04 | 14.0% | 0.0% |
| gusty | PPO | 50.0% | 3.35 | 0.64 | 49.0% | 0.0% |
| storm | LCvx cascade | 98.0% | 2.70 | 1.38 | 2.0% | 0.0% |
| storm | Heuristic | 65.0% | 2.44 | 0.02 | 31.0% | 3.0% |
| storm | PPO | 24.0% | 3.33 | 0.55 | 71.0% | 1.0% |

## Interpretation guide

Success depends on the combined feasibility of convex guidance and the rigid-body attitude loop.
Higher replanning time is the cost of repeatedly solving a DPP SOCP; look at mean latency alongside reliability.
Fuel is measured only among successful episodes; do not compare it in isolation from the success rate.
The light-gravity case additionally uses relaxed engine-off planning with sigma-delta thrust pulses.
