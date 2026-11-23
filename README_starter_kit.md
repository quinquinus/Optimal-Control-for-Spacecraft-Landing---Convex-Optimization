# Starter kit: Fuel-Optimal Landing with Lossless Convexification

This kit contains everything you need for the project: the simulators, the benchmarks, the evaluation code and
skeleton files. The project description is `docs/Questions.pdf`. The short note `docs/DPP_primer.pdf` explains
how to write a CVXPY problem with parameters so that it can be solved many times quickly (Question 3.1).

## Installation

Python 3.10 or newer. Box2D needs the SWIG compiler (`conda install swig` or `apt install swig`).

```
pip install -r requirements.txt
```

## What is where

| File | Content | Questions |
|---|---|---|
| `lander/params.py` | the parameters of the lander (masses, thrust, fuel use) | all |
| `lander/lcvx.py` | the convex guidance problem `LanderSOCP`, the time-of-flight search, the two-stage guidance | 3.1, 3.2 |
| `lander/nonconvex.py` | the nonconvex baseline: SLSQP on the original problem (complete) | 3.1 |
| `lander/pointmass.py` | the point-mass simulator and Benchmark B1 | 3 |
| `lander/env.py` | the rigid-body simulator: Gymnasium's LunarLander with a fuel tank, wind and hazards | 4, 5 |
| `lander/controllers.py` | the cascade controller | 4 |
| `lander/scenarios.py` | the B2 conditions and the ten scenarios of the challenge | 4, 5 |
| `lander/evaluate.py` | runs a policy on B2 or on the challenge scenarios | 4, 5 |
| `lander/wind.py` | the known wind model and the online gust estimator `GustEstimator` (complete) | 5 |
| `lander/geometry.py`, `lander/plot.py` | helpers for hazards and plots | 5 |
| `submissions/` | the baselines and the submission template | 4, 5 |
| `models/ppo_fuel_lander.zip` | the PPO agent (trained with `scripts/train_rl_baseline.py`) | 4 |

Every place where you must write code raises `NotImplementedError` with the number of the question (for example `"Question 3.1"` or `"Question 4"`). The questions build on each
other: the controllers of Question 4 use your convex problem of Question 3.

## Running things

```
# Benchmark B2 (Question 4), 100 seeds per condition
python -m lander.evaluate --policy lander.controllers:make_cascade --benchmark b2 --n 100

# Question 5: train seeds (as many as you like) and the public seeds of the leaderboard
python -m lander.evaluate --policy submissions/submission_template.py --split train --n 20
python -m lander.evaluate --policy submissions/submission_template.py --split public

# Watch your controller fly: one animated GIF per scenario in the folder gifs/
python -m lander.evaluate --policy submissions/submission_template.py --scenarios S05 --gif
python -m lander.evaluate --policy lander.controllers:make_cascade --benchmark b2 --scenarios storm --seed 7 --gif
```

With `--gif`, nothing is scored: the command records one episode per scenario (the first seed of the chosen split,
or the seed given with `--seed`) and saves it as `gifs/<policy>_<scenario>_seed<seed>.gif`. The animation shows the
wind arrow, the fuel gauge and the pad. Use `--gif my_folder` to choose another folder.

A policy is a Python file with a function `make_policy(spec)` that returns an object with the methods `reset()`
and `act(obs, info)`. You may only use `obs`, `info["state"]`, `info["t"]` and `spec`.

## Leaderboard

Zip your `submission.py` (and any helper modules; the `lander` package is available on the server) and upload
the zip on the leaderboard page. You can submit at most five times per day. The final ranking uses 30 secret seeds
per scenario, so do not tune your controller to the public seeds.
