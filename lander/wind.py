"""Known wind model and online gust estimator for the Windy Hazard Landing Challenge (Question 5).

The horizontal wind acceleration is  w(t, y) = mean + shear * (y + LEG_DOWN) / 10 + xi(t), where y is the
pad-frame height of the lander's center (y = 0 when resting on the pad) and xi is an Ornstein-Uhlenbeck gust
with standard deviation gust_std and correlation time gust_tau (see lander/env.py). The mean and the shear
are known from spec["wind"]; the gust is not, and GustEstimator estimates it online.

Usage inside a policy:
    est = GustEstimator(spec)          # once, in make_policy / __init__
    est.reset()                        # at the start of every episode
    xi_hat = est.update(s, a_prev)     # every step: current state, action applied at the previous step
    drift = est.predict(t_future)      # expected gust after t_future seconds: xi_hat * exp(-t / gust_tau)
"""
import numpy as np

from .params import P as PARAMS, LEG_DOWN
from .model import DT


def known_wind(wind, y):
    """Known part of the lateral wind acceleration (m/s^2) at pad-frame height y: mean + shear term."""
    return wind["mean"] + wind["shear"] * (np.asarray(y) + LEG_DOWN) / 10.0


def engine_powers(action):
    """Effective main-engine throttle p in {0} U [0.5, 1] and side command c, as applied by the simulator."""
    a0, a1 = float(action[0]), float(action[1])
    p = 0.0 if a0 <= 0 else (np.clip(a0, 0, 1) + 1) * 0.5
    c = 0.0 if abs(a1) <= 0.5 else np.sign(a1) * min(abs(a1), 1.0)
    return p, c


class GustEstimator:
    """Exponential filter of the lateral acceleration residual.

    At every step, the measured lateral acceleration (v_x(k+1) - v_x(k)) / DT is compared with the
    acceleration that the engines (with the throttle actually applied), the known mean wind and the shear
    explain. The residual is the gust plus noise (mainly the random engine dispersion of Gymnasium), and its
    exponential average over about `tau` seconds is the estimate xi_hat. A shorter `tau` follows the gust
    faster but lets more engine noise through.
    """

    def __init__(self, spec, params=PARAMS, tau=0.4, enabled=True):
        self.wind, self.p, self.tau, self.enabled = spec["wind"], params, tau, enabled
        self.gust_tau = spec["wind"]["gust_tau"]
        self.reset()

    def reset(self):
        self.xi_hat, self.prev_s = 0.0, None

    def update(self, s, a_prev):
        """s: current state (x, y, vx, vy, theta, omega, m); a_prev: action applied at the previous step.
        Returns the current gust estimate xi_hat (m/s^2)."""
        if self.prev_s is not None and self.enabled:
            ps = self.prev_s
            pm, c = engine_powers(a_prev)
            th, m = ps[4], ps[6]
            a_model = (self.p.F * pm * -np.sin(th) + self.p.F_side * c * np.cos(th)) / m + known_wind(self.wind, ps[1])
            a_meas = (s[2] - ps[2]) / DT
            beta = DT / self.tau
            self.xi_hat = (1 - beta) * self.xi_hat + beta * (a_meas - a_model)
        self.prev_s = np.array(s, dtype=float).copy()
        return self.xi_hat

    def predict(self, t):
        """Expected gust t seconds ahead, E[xi(now + t) | xi(now) = xi_hat] = xi_hat * exp(-t / gust_tau)."""
        return self.xi_hat * np.exp(-np.asarray(t) / self.gust_tau)
