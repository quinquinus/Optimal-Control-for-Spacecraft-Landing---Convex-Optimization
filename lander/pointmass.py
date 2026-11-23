"""Point-mass simulator of the lander (Questions 2-3) and Benchmark B1.

The simulator uses the same parameters as the rigid-body lander (lander/params.py) but none of its
complications: no rotation, no legs, no engine noise. It integrates
    r' = v,   v' = T/m + g,   m' = -alpha ||T||
with RK4 on a fine time grid. A guidance solution gives the thrust acceleration u_k on each interval;
the engine then delivers T(t) = m(t) u_k (mode "accel", what an onboard throttle loop with a mass
estimate does) or holds the thrust T_k = m_k u_k constant (mode "thrust").

Benchmark B1: 20 frozen initial states (x, y, vx, vy, fuel), gravity -10 m/s^2. Cases 17-19 start with
too little fuel to reach the pad centre: the best one can do is land as close as possible (the
minimum-landing-error problem of Question 3.2).
"""
import time
import numpy as np

from .params import P as PARAMS


def simulate(sol, p=PARAMS, g=-10.0, mode="accel", n_sub=40, x0=None, m0=None):
    """Simulate a LanderSolution open loop. Returns a dict with the fine trajectory and the errors."""
    N = len(sol.u)
    dt = sol.tgo / N
    h = dt / n_sub
    x = np.r_[sol.p[0], sol.v[0], np.exp(sol.z[0])] if x0 is None else np.r_[x0, m0]
    gv = np.array([0.0, g])
    traj, thrust = [x.copy()], []

    def f(x, k):
        m = x[4]
        T = sol.u[k] * m if mode == "accel" else sol.u[k] * np.exp(sol.z[k])
        return np.r_[x[2:4], T / m + gv, -p.alpha * np.linalg.norm(T)], np.linalg.norm(T)

    for k in range(N):
        for _ in range(n_sub):
            k1, T1 = f(x, k)
            k2, _ = f(x + 0.5 * h * k1, k)
            k3, _ = f(x + 0.5 * h * k2, k)
            k4, _ = f(x + h * k3, k)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            traj.append(x.copy())
            thrust.append(T1)
    traj, thrust = np.array(traj), np.array(thrust)
    xN = traj[-1]
    target = np.array([sol.p[-1, 0], 0.0])
    on = thrust > 1e-9
    viol_T = max(0.0, float(np.max(np.r_[p.rho1 - thrust[on], thrust[on] - p.rho2]))) if on.any() else 0.0
    hor, alt = np.abs(traj[:, 0] - xN[0]), traj[:, 1] - xN[1]
    return dict(traj=traj, thrust=thrust, final_pos_err=float(np.linalg.norm(xN[:2] - target)),
                final_vel_err=float(np.linalg.norm(xN[2:4])), fuel=float(traj[0, 4] - xN[4]),
                thrust_violation=viol_T, x_final=xN)


def glide_violation(traj, tan_gamma):
    """Largest depth (m) below the glide cone whose apex is the final point."""
    xN = traj[-1]
    return float(max(0.0, np.max(tan_gamma * np.abs(traj[:, 0] - xN[0]) - (traj[:, 1] - xN[1]))))


# ---------------------------------------------------------------------- Benchmark B1
def _b1_cases():
    rng = np.random.default_rng(20261108)
    cases = []
    while len(cases) < 17:
        x0 = rng.uniform(2.0, 8.0) * rng.choice([-1, 1])
        y0 = rng.uniform(7.0, 12.0)
        vx0 = -np.sign(x0) * rng.uniform(-1.0, 3.0)       # mostly towards the pad, sometimes away
        vy0 = rng.uniform(-3.0, 0.0)
        fuel0 = rng.uniform(0.25, PARAMS.m_fuel)
        cases.append(np.round([x0, y0, vx0, vy0, fuel0], 3))
    cases += [np.array([-6.0, 9.0, -6.0, 0.0, 0.176]),     # flying away from the pad with little fuel
              np.array([6.0, 9.0, 4.0, -1.0, 0.149]),
              np.array([-5.0, 8.0, -5.0, 0.0, 0.155])]
    return cases


B1_CASES = _b1_cases()
B1_UNREACHABLE = [17, 18, 19]


def run_b1(guidance, d_star_fn=None, p=PARAMS, g=-10.0, tan_gamma=np.tan(np.radians(45.0))):
    """guidance(x0, m0) -> LanderSolution. Simulates each solution with the point-mass simulator."""
    rows = []
    for i, case in enumerate(B1_CASES):
        x0, m0 = case[:4], p.m_dry + case[4]
        t0 = time.perf_counter()
        try:
            sol = guidance(x0, m0)
        except Exception:
            sol = None
        ts = time.perf_counter() - t0
        d_star = d_star_fn(x0, m0) if (d_star_fn is not None and i in B1_UNREACHABLE) else None
        row = dict(case=i, reachable=i not in B1_UNREACHABLE, t_solve=ts, d_star=d_star)
        if sol is None or not sol.feasible:
            row.update(ok=False, landed=False)
        else:
            sim = simulate(sol, p, g)
            d = float(abs(sim["x_final"][0]))
            soft = abs(sim["x_final"][1]) < 0.05 and sim["final_vel_err"] < 0.05
            row.update(ok=True, d_land=d, fuel=sim["fuel"], fuel_s=sim["fuel"] / (p.alpha * p.F),
                       viol_T=sim["thrust_violation"], viol_gs=glide_violation(sim["traj"], tan_gamma),
                       gap=float(np.max(np.abs(sol.relaxation_gap))),
                       landed=bool(soft and (d <= 0.05 if d_star is None else d <= d_star + 0.05)))
        rows.append(row)
    return rows
