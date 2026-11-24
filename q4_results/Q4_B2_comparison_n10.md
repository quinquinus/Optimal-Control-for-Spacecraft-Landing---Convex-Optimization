# Question 4: Benchmark B2 (n=10 seeds per condition)

All three controllers were evaluated on the same five conditions and frozen seeds.
Fuel = mean seconds of full main-engine thrust among successful episodes.
Time = mean computation time per controller step in milliseconds.

## Overall

| Policy | Success | Mean score | Fuel (s, successes) | Time (ms/step) |
|---|---:|---:|---:|---:|
| LCvx cascade | 100.0% | 87.59 | 2.48 | 1.79 |
| Heuristic | 80.0% | 72.73 | 1.89 | 0.03 |
| PPO | 76.0% | 64.35 | 3.34 | 0.59 |

## By environment

| Environment | Policy | Success | Fuel (s, successes) | Time (ms/step) | Off-pad | Crash/hard |
|---|---|---:|---:|---:|---:|---:|
| nominal | LCvx cascade | 100.0% | 2.73 | 1.94 | 0.0% | 0.0% |
| nominal | Heuristic | 90.0% | 1.87 | 0.03 | 10.0% | 0.0% |
| nominal | PPO | 100.0% | 3.31 | 0.69 | 0.0% | 0.0% |
| heavy | LCvx cascade | 100.0% | 2.59 | 2.55 | 0.0% | 0.0% |
| heavy | Heuristic | 70.0% | 1.76 | 0.03 | 0.0% | 30.0% |
| heavy | PPO | 90.0% | 2.83 | 0.74 | 10.0% | 0.0% |
| light | LCvx cascade | 100.0% | 1.97 | 1.79 | 0.0% | 0.0% |
| light | Heuristic | 100.0% | 1.45 | 0.03 | 0.0% | 0.0% |
| light | PPO | 100.0% | 3.84 | 0.53 | 0.0% | 0.0% |
| gusty | LCvx cascade | 100.0% | 2.33 | 1.40 | 0.0% | 0.0% |
| gusty | Heuristic | 80.0% | 2.09 | 0.03 | 20.0% | 0.0% |
| gusty | PPO | 70.0% | 3.37 | 0.53 | 30.0% | 0.0% |
| storm | LCvx cascade | 100.0% | 2.78 | 1.29 | 0.0% | 0.0% |
| storm | Heuristic | 60.0% | 2.54 | 0.02 | 40.0% | 0.0% |
| storm | PPO | 20.0% | 3.21 | 0.46 | 80.0% | 0.0% |

## Interpretation guide

Success depends on the combined feasibility of convex guidance and the rigid-body attitude loop.
Higher replanning time is the cost of repeatedly solving a DPP SOCP; look at mean latency alongside reliability.
Fuel is measured only among successful episodes; do not compare it in isolation from the success rate.
The light-gravity case additionally uses relaxed engine-off planning with sigma-delta thrust pulses.
