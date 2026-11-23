"""Question 5 submission template (Windy Hazard Landing Challenge).

Zip this file (renamed submission.py) together with any helper modules and upload it to the leaderboard.
Test locally first:
    python -m lander.evaluate --policy submissions/submission_template.py --split train --n 20
    python -m lander.evaluate --policy submissions/submission_template.py --split public
"""
import numpy as np

from lander.wind import GustEstimator, known_wind   # provided: known wind model and online gust estimator


class MyPolicy:
    def __init__(self, spec):
        # spec: public description of the scenario (see lander/scenarios.py, Scenario.public_dict):
        #   spec["terrain"]  list of (x, y) knots, world frame       spec["pad"] = [x1, x2, y]
        #   spec["hazards"]  [{"kind": "rock"|"zone", "poly": [[x, y], ...] (convex, CCW)}]
        #   spec["wind"]     {"mean", "shear", "gust_std", "gust_tau", "torque_std"}
        #   spec["gravity"], spec["start"], spec["start_std"], spec["max_steps"]
        self.spec = spec
        self.gust = GustEstimator(spec)     # exponential filter of the lateral acceleration residual

    def reset(self):
        self.gust.reset()
        self.a_prev = np.zeros(2)

    def act(self, obs, info):
        # obs: LunarLander observation, x and y relative to THIS scenario's pad centre
        # info["state"] = [x, y, vx, vy, theta, omega, mass] in SI units (pad frame), info["t"] = time (s)
        s = info["state"]
        xi_hat = self.gust.update(s, self.a_prev)        # current gust estimate (m/s^2)
        # known wind at the current height: known_wind(self.spec["wind"], s[1]);
        # expected gust t seconds ahead: self.gust.predict(t)
        a = np.array([-1.0, 0.0])           # engine off: replace with your controller
        self.a_prev = a
        return a


def make_policy(spec):
    return MyPolicy(spec)
