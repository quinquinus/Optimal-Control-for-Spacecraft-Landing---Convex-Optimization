"""Nonconvex baseline for Question 3.1 (provided, complete).

We solve the ORIGINAL problem (P), without relaxation or change of variables, with the local solver SLSQP of
scipy.optimize.minimize. The decision variables are the thrust vectors T_k, held constant on each of the N
intervals (scaled by rho2). The constraints are the annulus rho1 <= ||T_k|| <= rho2, the tilt cone, the glide
slope at the nodes, the touchdown r_N = 0, v_N = 0, and the fuel limit m_N >= m_dry; the time of flight t_f is
fixed. The state is propagated interval by interval with the mass of the start of each interval.

Usage:
    from lander.nonconvex import nonconvex_baseline
    runs = nonconvex_baseline(x0, tf)       # one dict per initial guess
    # run["converged"]: SLSQP reports success, run["feasible"]: constraints hold to 1e-4,
    # run["fuel"]: fuel used (kg), run["time"]: wall-clock time (s)
"""
import time
import numpy as np
from scipy.optimize import minimize

from .params import P as PARAMS


def nonconvex_baseline(x0, tf, m0=None, N=20, n_starts=12, seed=0, p=PARAMS, g=-10.0, gamma_deg=45.0,
                       theta_max_deg=30.0, maxiter=500):
    """SLSQP on problem (P) from `n_starts` random initial guesses; x0 = (x, y, vx, vy), tf fixed.
    Returns a list with one dict per initial guess: converged, feasible, fuel (kg), time (s)."""
    x0 = np.asarray(x0, float)
    m0 = p.m_wet if m0 is None else m0
    dt, S = tf / N, p.rho2
    gv = np.array([0.0, g])
    ct = np.cos(np.radians(theta_max_deg))
    tg = np.tan(np.radians(gamma_deg))

    def rollout(x):
        T = x.reshape(N, 2) * S
        r, v, m = x0[:2].copy(), x0[2:].copy(), m0
        rs = [r.copy()]
        for k in range(N):
            a = T[k] / m + gv
            r = r + v * dt + 0.5 * a * dt ** 2
            v = v + a * dt
            m = m - p.alpha * np.linalg.norm(T[k]) * dt
            rs.append(r.copy())
        return np.array(rs), v, m

    nrm = lambda x: np.linalg.norm(x.reshape(N, 2), axis=1)
    cons = [{"type": "eq", "fun": lambda x: np.r_[rollout(x)[0][-1], rollout(x)[1]]},              # r_N = 0, v_N = 0
            {"type": "ineq", "fun": lambda x: nrm(x) - p.rho1 / S},                                 # ||T_k|| >= rho1
            {"type": "ineq", "fun": lambda x: 1.0 - nrm(x)},                                        # ||T_k|| <= rho2
            {"type": "ineq", "fun": lambda x: x.reshape(N, 2)[:, 1] - ct * nrm(x)},                 # tilt cone
            {"type": "ineq", "fun": lambda x: (lambda rs: rs[:, 1] - tg * np.abs(rs[:, 0] - rs[-1, 0]))(rollout(x)[0])},
            {"type": "ineq", "fun": lambda x: rollout(x)[2] - p.m_dry}]                             # m_N >= m_dry
    fuel = lambda x: p.alpha * dt * S * np.sum(nrm(x)) * 10                                        # scaled objective
    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(n_starts):
        ang = rng.uniform(-0.5, 0.5, size=N)                                    # random tilts and magnitudes
        xg = (np.c_[-np.sin(ang), np.cos(ang)] * rng.uniform(0.5, 1.0, size=(N, 1))).ravel()
        t0 = time.time()
        out = minimize(fuel, xg, method="SLSQP", constraints=cons, options=dict(maxiter=maxiter, ftol=1e-10))
        viol = max(np.max(np.abs(cons[0]["fun"](out.x))),
                   *[-min(0.0, float(np.min(c["fun"](out.x)))) for c in cons[1:]])
        runs.append(dict(converged=bool(out.success), feasible=bool(viol < 1e-4),
                         fuel=float(m0 - rollout(out.x)[2]), time=time.time() - t0))
    return runs
