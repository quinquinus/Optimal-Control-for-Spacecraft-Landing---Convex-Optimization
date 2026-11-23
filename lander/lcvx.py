"""Lossless convexification (LCvx) of fuel-optimal landing guidance for the planar lander.

The nonconvex problem (point mass, changing mass, thrust annulus) becomes a second-order cone program
(SOCP) in four steps:
  (1) lossless relaxation: a slack Gamma with ||T|| <= Gamma, rho1 <= Gamma <= rho2;
  (2) exact change of variables u = T/m, sigma = Gamma/m, z = ln m (the dynamics become linear);
  (3) conservative Taylor bounds on rho1 e^{-z} <= sigma <= rho2 e^{-z} around z0(t);
  (4) zero-order hold: u and sigma are constant on each of the N intervals.

The same class serves the whole project:
  * terminal="ground": land at the target with zero velocity (Questions 1-3, point-mass model);
  * terminal="gate":   reach a point y_gate above the pad, descending at v_gate (Questions 4-5, the
                       rigid-body lander finishes the landing with a simple descent controller);
  * optional wind drift, wind shear, hazard halfspaces with slacks, trust region, approach cone (Q5).
All data that change between solves are CVXPY Parameters, so the problem is compiled once (DPP).
"""
from dataclasses import dataclass
import time
import numpy as np
import cvxpy as cp

from .params import P as PARAMS


@dataclass
class LanderSolution:
    status: str
    tgo: float
    t: np.ndarray        # (N+1,)
    p: np.ndarray        # (N+1, 2) position (x, y)
    v: np.ndarray        # (N+1, 2) velocity
    z: np.ndarray        # (N+1,)   log-mass
    u: np.ndarray        # (N, 2)   thrust acceleration T/m
    sigma: np.ndarray    # (N,)     slack acceleration Gamma/m
    fuel: float          # kg
    fuel_s: float        # seconds of full thrust
    landing_error: float # m
    slack: float         # total hazard slack (0 if no hazards)
    solve_time: float    # s
    glide_dual: np.ndarray | None = None

    @property
    def m(self):
        return np.exp(self.z)

    @property
    def thrust(self):
        return self.u * np.exp(self.z[:-1])[:, None]

    @property
    def relaxation_gap(self):
        """sigma_k - ||u_k|| >= 0; zero means the relaxation is exact at node k."""
        return self.sigma - np.linalg.norm(self.u, axis=1)

    @property
    def feasible(self):
        return self.status in ("optimal", "optimal_inaccurate")


class LanderSOCP:
    """The relaxed, discretized landing problem for a fixed number of intervals N.

    Objective (argument mode):
      "fuel"                   minimum fuel; on the ground, land exactly at the target x_N = x_target (Question 3.1)
      "error"                  minimum landing error |x_N - x_target| (Question 3.2)
      "fuel_with_error_bound"  minimum fuel subject to |x_N - x_target| <= dmax (Question 3.2)
    Terminal condition (argument terminal):
      "ground"  touchdown at rest, y_N = 0 and v_N = 0 (Question 3)
      "gate"    approach gate above the pad: y_N = y_gate, v_N = (0, -v_gate) and, for mode "fuel",
                |x_N - x_target| <= x_tol (provided; used from Question 4 on)
    The target is x_target = 0 (the pad center). The remaining arguments (shear, n_obs, trust_region, cone_deg,
    map_limits) are provided extensions for Phase 3; with their default values they change nothing.
    """

    def __init__(self, p=PARAMS, N=40, g=-10.0, mode="fuel", terminal="ground", theta_max_deg=30.0,
                 gamma_deg=45.0, glide_from=0, thrust_frac=1.0, y_gate=1.5, v_gate=0.8, x_tol=0.3,
                 shear=0.0, n_obs=0, cone_deg=None, trust_region=False, w_slack=1e3, map_limits=None):
        self.p, self.N, self.g, self.mode, self.terminal = p, N, g, mode, terminal
        self.thrust_frac = thrust_frac
        r = cp.Variable((N + 1, 2), name="p")
        v = cp.Variable((N + 1, 2), name="v")
        z = cp.Variable(N + 1, name="z")
        u = cp.Variable((N, 2), name="u")
        s = cp.Variable(N, name="sigma")
        dz = cp.Variable(N, name="dz")                       # z_k - z0_k (keeps the problem DPP)
        self.r, self.v, self.z, self.u, self.s = r, v, z, u, s
        # parameters
        self.x0 = cp.Parameter(4, name="x0")                 # initial (x, y, vx, vy)
        self.zinit = cp.Parameter(name="zinit")              # initial log-mass
        self.dt = cp.Parameter(nonneg=True, name="dt")
        self.dt2 = cp.Parameter(nonneg=True, name="dt2")
        self.adt = cp.Parameter(nonneg=True, name="alpha_dt")
        self.z0 = cp.Parameter(N, name="z0")                 # Taylor expansion points
        self.a1 = cp.Parameter(N, nonneg=True, name="a1")    # rho1 e^{-z0}
        self.a2 = cp.Parameter(N, nonneg=True, name="a2")    # rho2 e^{-z0}
        self.zlo = cp.Parameter(N + 1, name="zlo")
        self.zhi = cp.Parameter(N + 1, name="zhi")
        self.tg = cp.Parameter(nonneg=True, name="tan_gamma", value=np.tan(np.radians(gamma_deg)))
        self.xt = cp.Parameter(name="x_target", value=0.0)
        self.dmax = cp.Parameter(nonneg=True, name="dmax", value=0.0)
        self.dvx = cp.Parameter(N, name="dvx", value=np.zeros(N))   # dt * w_k      (known wind drift)
        self.dpx = cp.Parameter(N, name="dpx", value=np.zeros(N))   # dt^2/2 * w_k

        gy = np.zeros((N, 2)); gy[:, 1] = g
        e1 = np.array([[1.0, 0.0]])
        # acceleration on each interval: thrust acceleration plus gravity, acc_k = u_k + g  (N x 2)
        acc = u + gy
        if shear:                                    # shear wind (shear/10) * y enters the x-acceleration (Phase 3)
            acc = acc + cp.reshape((shear / 10.0) * r[:-1, 1], (N, 1), order="C") @ e1
        # drift caused by a known horizontal wind on each interval (Phase 3); zero unless solve() receives a wind
        drift_v = cp.reshape(self.dvx, (N, 1), order="C") @ e1      # rows (dt w_k, 0)
        drift_r = cp.reshape(self.dpx, (N, 1), order="C") @ e1      # rows (dt^2/2 w_k, 0)
        # TODO (Question 3.1): the list `cons`: initial state (self.x0, self.zinit); dynamics of Question 2.1 written with `acc`, `drift_v`, `drift_r`; relaxation ||u_k|| <= sigma_k; Taylor bounds (with the variable `dz` = z - z0, which keeps the problem DPP); mass bounds zlo <= z <= zhi and z_N >= ln(m_dry); tilt cone; glide slope y_k - y_N >= tan(gamma) |x_k - x_N| for k = glide_from, ..., N. Then the fuel expression `fuel` = dt * sum(sigma)
        raise NotImplementedError("Question 3.1")
        err = cp.abs(r[N, 0] - self.xt)              # horizontal landing error |x_N - x_target| (the target is 0)
        if terminal == "ground":
            # TODO (Question 3.1): touchdown at rest, y_N = 0 and v_N = 0 (append them to `cons`)
            raise NotImplementedError("Question 3.1")
        else:                                        # approach gate above the pad (provided; used from Question 4 on)
            cons += [r[N, 1] == y_gate, v[N, 0] == 0.0, v[N, 1] == -v_gate]
        if mode == "fuel":
            if terminal == "ground":
                # TODO (Question 3.1): land at the target, x_N = x_target (`self.xt`, which is 0) (append it to `cons`)
                raise NotImplementedError("Question 3.1")
            else:
                cons += [err <= x_tol]
            obj = fuel
        elif mode == "error":
            # TODO (Question 3.2): the objective `obj`, the landing error `err`
            raise NotImplementedError("Question 3.2")
        elif mode == "fuel_with_error_bound":
            # TODO (Question 3.2): the objective `obj` (the fuel) and the bound `err` <= `self.dmax` (append it to `cons`)
            raise NotImplementedError("Question 3.2")
        else:
            raise ValueError(mode)
        # ---------------- extensions used in Question 5 (hazards, trust region, approach cone, map)
        self.H = [cp.Parameter((N + 1, 2), name=f"H{j}") for j in range(n_obs)]
        self.b = [cp.Parameter(N + 1, name=f"b{j}") for j in range(n_obs)]
        slack = []
        for H, b in zip(self.H, self.b):
            nu1, nu2 = cp.Variable(N, nonneg=True), cp.Variable(N - 1, nonneg=True)
            cons += [cp.sum(cp.multiply(H[1:], r[1:]), axis=1) >= b[1:] - nu1,      # node k
                     cp.sum(cp.multiply(H[1:N], r[2:]), axis=1) >= b[1:N] - nu2]     # node k+1 (whole chord)
            slack += [cp.sum(nu1), cp.sum(nu2)]
        self.slack = cp.sum(cp.hstack(slack)) if slack else cp.Constant(0.0)
        if trust_region:
            self.pbar = cp.Parameter((N + 1, 2), name="pbar")
            self.delta = cp.Parameter(nonneg=True, name="delta")
            cons += [cp.abs(r[1:] - self.pbar[1:]) <= self.delta]
        if cone_deg is not None:                     # final-approach cone on the last nodes (apex at the gate)
            cot = 1.0 / np.tan(np.radians(cone_deg))
            self.cone = cp.Parameter(N + 1, nonneg=True, name="cone", value=np.zeros(N + 1))
            yref = y_gate if terminal == "gate" else 0.0
            cons += [cp.multiply(self.cone, r[:, 0] - cot * (r[:, 1] - yref)) <= x_tol,
                     cp.multiply(self.cone, -r[:, 0] - cot * (r[:, 1] - yref)) <= x_tol]
        if map_limits is not None:
            (xlo, xhi), ymax = map_limits
            cons += [r[1:, 0] >= xlo, r[1:, 0] <= xhi, r[1:, 1] <= ymax]
        self.fuel_expr = fuel
        self.prob = cp.Problem(cp.Minimize(obj + w_slack * self.slack), cons)
        assert self.prob.is_dcp(dpp=True), "problem must be DPP"

    # ------------------------------------------------------------------ parameters
    def set_time(self, tgo, m0):
        """All time- and mass-dependent parameters for a horizon tgo starting at mass m0."""
        p, N = self.p, self.N
        dt = tgo / N
        t = np.arange(N + 1) * dt
        rho1, rho2 = p.rho1, self.thrust_frac * p.rho2
        # TODO (Question 3.1): set the values of every horizon-dependent parameter: dt, dt2 (= dt^2), adt (= alpha dt), z0, a1, a2, zlo, zhi at the times t_k (use the local rho1, rho2: rho2 includes the factor thrust_frac)
        raise NotImplementedError("Question 3.1")
        self.zinit.value = np.log(m0)
        return t

    def solve(self, x0, tgo, m0=None, wind=None, halfspaces=None, pbar=None, delta=None, t_cone=2.0,
              allow_off=False, solver=cp.CLARABEL):
        """x0 = (x, y, vx, vy); wind = known lateral wind at the nodes (N,) or None.
        allow_off=True relaxes the thrust set {0} U [rho1, rho2] (engine off or on) to [0, rho2]: the
        lower bound on sigma is dropped. Check the result: a node with 0 < ||T_k|| < rho1 cannot be flown."""
        N = self.N
        m0 = self.p.m_wet if m0 is None else m0
        t = self.set_time(tgo, m0)
        if allow_off:                                # engine may be off: no lower thrust bound,
            self.a1.value = np.zeros(N)              # and the mass may stay at m0
            self.zhi.value = np.full(N + 1, np.log(m0))
        dt = tgo / N
        self.x0.value = np.asarray(x0, float)
        w = np.zeros(N) if wind is None else np.asarray(wind, float)
        self.dvx.value, self.dpx.value = dt * w, 0.5 * dt * dt * w
        if self.H:
            for (H, b), (Hv, bv) in zip(zip(self.H, self.b), halfspaces):
                H.value, b.value = Hv, bv
        if pbar is not None:
            self.pbar.value, self.delta.value = pbar, delta
        if hasattr(self, "cone"):
            self.cone.value = (t >= tgo - t_cone).astype(float)
        t0 = time.perf_counter()
        try:
            self.prob.solve(solver=solver)
        except cp.SolverError:
            return None
        st = time.perf_counter() - t0
        if self.prob.status not in ("optimal", "optimal_inaccurate"):
            return None
        z = self.z.value
        fuel = float(np.exp(z[0]) - np.exp(z[-1]))
        gd = None
        if getattr(self, "glide", None) and self.glide[0].dual_value is not None:
            gd = np.asarray(self.glide[0].dual_value) + np.asarray(self.glide[1].dual_value)
        return LanderSolution(self.prob.status, tgo, t, self.r.value.copy(), self.v.value.copy(), z.copy(),
                              self.u.value.copy(), self.s.value.copy(), fuel, fuel / (self.p.alpha * self.p.F),
                              float(abs(self.r.value[-1, 0] - self.xt.value)), float(self.slack.value), st, gd)


# ---------------------------------------------------------------------- time-of-flight search
def golden_section(f, a, b, tol=0.05, max_iter=60):
    """Minimize a unimodal function on [a, b]; returns (x*, f(x*), evaluations)."""
    gr = (np.sqrt(5) - 1) / 2
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = f(c), f(d)
    evals = [(c, fc), (d, fd)]
    for _ in range(max_iter):
        if b - a < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
            evals.append((c, fc))
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
            evals.append((d, fd))
    x = c if fc < fd else d
    return x, min(fc, fd), evals


def tgo_bounds(p, x0, m0=None):
    """t_min = m_dry ||v0|| / rho2 (needs vy0 <= 0) and t_max = fuel / (alpha rho1)."""
    m0 = p.m_wet if m0 is None else m0
    t_min = p.m_dry * np.linalg.norm(x0[2:]) / p.rho2
    t_max = (m0 - p.m_dry) / (p.alpha * p.rho1)
    return max(t_min, 0.5), t_max


def is_lossless(sol, tol=1e-4):
    """Numerical certificate of Question 1: the relaxation is exact if sigma_k = ||u_k|| at every node."""
    return bool(np.max(sol.relaxation_gap / np.maximum(sol.sigma, 1e-9)) <= tol)


def solve_min_fuel(prob, x0, m0=None, n_grid=25, tol=0.02, certify=True):
    """Coarse grid over [t_min, t_max], then golden section around the best feasible grid point.
    With certify=True, a solution whose relaxation gap is not zero counts as infeasible: we search
    over CERTIFIED (lossless) solutions only. Returns (best LanderSolution or None, evaluations)."""
    # TODO (Question 3.1): coarse grid of n_grid points on tgo_bounds, then golden_section around the best grid point; with certify=True, a tgo whose solution fails is_lossless counts as infeasible (infinite fuel); return (best solution or None, sorted list of (tgo, fuel))
    raise NotImplementedError("Question 3.1")


def solve_min_landing_error(p, x0, m0=None, N=40, g=-10.0, n_grid=25, tol=0.02, certify=True, **kw):
    """Two-stage guidance of Blackmore et al. (2010) for targets that cannot be reached:
    stage 1 finds d* = min over tgo of the landing error; stage 2 minimizes the fuel with error <= d*."""
    # TODO (Question 3.2): stage 1: LanderSOCP with mode='error' and the same search as solve_min_fuel (landing error instead of fuel); stage 2: LanderSOCP with mode='fuel_with_error_bound', dmax = d_star + 1 mm, solved at the best tgo; return (solution, d_star, sorted list of (tgo, error))
    raise NotImplementedError("Question 3.2")
