"""Conversions between Gymnasium LunarLander observations and SI units, and action helpers.

LunarLander (Gymnasium v3) observation, as documented:
    obs = [x, y, vx, vy, angle, angular_velocity, leg0_contact, leg1_contact]
with x scaled by 10 m, y by 6.667 m, vx by 5 m/s, vy by 7.5 m/s and angular velocity by 2.5 rad/s
(units of 0.4 rad/s). (x, y) is the lander centre relative to the pad centre at landing height.

SI state used everywhere in this project:  s = [x, y, vx, vy, theta, omega]  (m, m/s, rad, rad/s)
theta > 0 is a counter-clockwise tilt. The main engine pushes along d(theta) = (-sin theta, cos theta).
"""
import numpy as np

FPS = 50
DT = 1.0 / FPS
OBS_SCALE = np.array([10.0, 20.0 / 3.0, 5.0, 7.5, 1.0, 2.5])


def obs_to_si(obs):
    obs = np.asarray(obs, dtype=float)
    return obs[:6] * OBS_SCALE


def thrust_dir(theta):
    return np.array([-np.sin(theta), np.cos(theta)])


def main_action(throttle):
    """Map a desired throttle in [0.5, 1] to action[0] in [0, 1]; throttle 0 -> engine off (-1)."""
    if throttle <= 0.0:
        return -1.0
    # NB: the engine fires only for action[0] > 0 (strictly), hence the 1e-3 floor
    return float(np.clip(2.0 * throttle - 1.0, 1e-3, 1.0))


def side_action(cmd):
    """Signed side-engine power. |power| < 0.25 -> off, otherwise mapped into the firing range
    0.5 < |action| <= 1 (the side engines fire only for |action[1]| > 0.5, strictly).
    A negative command produces a positive (CCW) angular acceleration."""
    if abs(cmd) < 0.25:
        return 0.0
    return float(np.sign(cmd) * np.clip(abs(cmd), 0.501, 1.0))
