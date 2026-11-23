"""Closed-loop landing controller for the rigid-body lander (Question 4).

  CascadeController  plans with the lossless point-mass SOCP (lander/lcvx.py). The plan says where to go
                     and how hard to push (a thrust acceleration); a tracking law corrects deviations
                     and an attitude loop rotates the body towards the commanded thrust direction.

Flight phases:
  POWERED   follow the latest plan; replan every `replan_every` steps from the measured state.
            In low gravity the minimum thrust exceeds the weight and the lander cannot hover; the plan
            then allows the engine to be off (thrust set {0} U [rho1, rho2] relaxed to [0, rho2]), which
            gives burn-coast-burn profiles, and throttle commands below 50 % are flown as pulses.
  TERMINAL  below the gate (y < y_gate + 0.1 m): constant-velocity vertical descent at -v_gate.
  TOUCHDOWN both legs on the ground: engine off.
"""
import numpy as np

from .model import obs_to_si, main_action, side_action
from .params import P as PARAMS
from .lcvx import LanderSOCP, is_lossless

DEFAULT_ID = dict(F=PARAMS.F, F_side=PARAMS.F_side, k_omega_dry=PARAMS.k_omega_dry)


class CascadeController:
    name = "LCvx cascade"

    def __init__(self, gravity=-10.0, ident=None, params=PARAMS, N=20, replan_every=25, derate=0.9, spec=None,
                 theta_plan_deg=25.0, theta_trk_deg=35.0, gamma_deg=30.0, v_gate=None, y_gate=None, x_tol=0.5,
                 kp_att=25.0, kd_att=10.0, kp_trk=(1.5, 2.0), kd_trk=(2.5, 3.0), tgo_margin=1.1,
                 tgo_grid=None):
        self.id = dict(DEFAULT_ID, **(ident or {}))
        self.p = params.with_(F=self.id["F"], F_side=self.id["F_side"], k_omega_dry=self.id["k_omega_dry"])
        self.g = gravity
        # thrust reserve for feedback: plan with at most `derate` of the maximum thrust, but always keep
        # a planned maximum acceleration of at least 1.15 |g| with a full tank (heavy gravity)
        derate = float(np.clip(max(derate, 1.15 * abs(gravity) * params.m_wet / self.id["F"]), 0.0, 0.95))
        self.N, self.replan_every, self.derate = N, replan_every, derate
        self.theta_trk = np.radians(theta_trk_deg)
        # the gate where the terminal descent starts: 1.0 m above the pad at 1.2 m/s if the lander can hover,
        # higher and slower (1.5 m, 0.8 m/s) if it cannot, because then the engine can only pulse
        can_hover = params.rho1 / params.m_dry < abs(gravity)
        y_gate = (1.0 if can_hover else 1.5) if y_gate is None else y_gate
        v_gate = (1.2 if can_hover else 0.8) if v_gate is None else v_gate
        self.v_gate, self.y_gate, self.x_tol = v_gate, y_gate, x_tol
        self.kp, self.kd = kp_att, kd_att
        self.kp_trk, self.kd_trk, self.tgo_margin = np.array(kp_trk), np.array(kd_trk), tgo_margin
        self.tgo_grid = np.linspace(1.0, 9.0, 10) if tgo_grid is None else tgo_grid
        lim = None
        if spec is not None:                       # stay on the map (x inside, y below the top + 1 m)
            c = 0.5 * (spec["pad"][0] + spec["pad"][1])
            lim = ((0.8 - c, 20.0 - 0.8 - c), 400 / 30.0 + 1.0 - (spec["pad"][2] + 0.6))
        self.G = LanderSOCP(self.p, N, gravity, mode="fuel", terminal="gate", theta_max_deg=theta_plan_deg,
                            gamma_deg=gamma_deg, glide_from=1, thrust_frac=self.derate, y_gate=y_gate,
                            v_gate=v_gate, x_tol=x_tol, map_limits=lim)
        self.reset()

    # ------------------------------------------------------------------ bookkeeping
    def reset(self):
        self.plan, self.k, self.k_since, self.sd_acc, self.k_next = None, 0, 0, 0.0, 0
        self.hover_possible = None
        self.phase = None
        self.log = dict(n_solves=0, track_err=[], fail=[])
        self.sd_main = 0.0

    def a_min(self, m):
        return self.p.rho1 / m

    # ------------------------------------------------------------------ planning (LCvx)
    def _solve(self, s, tgo, certify=True):  # noqa: C901
        """One SOCP. In the time-to-go search only certified (lossless) plans count; on a shrinking
        horizon we accept a slightly lossy plan (the throttle is then clipped at 50 %, and feedback
        corrects the small difference) rather than starting a new search."""
        self.log["n_solves"] += 1
        sol = self.G.solve(s[:4], tgo, s[6], allow_off=not self.hover_possible)
        if sol is None or (certify and not is_lossless(sol, 1e-3)):
            return None
        return dict(kind="lcvx", tgo=tgo, dt=tgo / self.N, p=sol.p, v=sol.v, a=sol.u, m=np.exp(sol.z),
                    sol=sol)

    def _search(self, s, refine=6):
        """Time-to-go search: coarse grid, then golden section around the best grid point (Question 3)."""
        # TODO (Question 4): time-to-go search as in solve_min_fuel (Question 3.1): the fuel of self._solve(s, t) (None = infeasible or not certified) on the grid self.tgo_grid, then `refine` golden-section steps around the best grid point; then solve the best plan again with a time-to-go self.tgo_margin times longer (keep the best plan if this fails); return the plan (dict) or None
        raise NotImplementedError("Question 4")

    def _replan(self, s):
        # TODO (Question 4): if there is a previous plan and its remaining time tgo = plan['tgo'] - self.k_since / 50 is above 0.6 s, return the first plan found by self._solve(s, f * tgo, certify=False) for f = 1.0, 1.15, 1.3; otherwise (or if all three fail) return a new search self._search(s)
        raise NotImplementedError("Question 4")

    # ------------------------------------------------------------------ main entry
    def act(self, obs, info=None):
        s = np.r_[obs_to_si(obs), info["state"][6]] if info is not None else np.r_[obs_to_si(obs), self.p.m_wet]
        if self.hover_possible is None:
            self.hover_possible = self.a_min(self.p.m_dry) < -self.g      # hover possible even when empty
            self.phase = "POWERED"
        legs = obs[6] > 0.5, obs[7] > 0.5
        if legs[0] and legs[1]:
            self.phase = "TOUCHDOWN"
            out = np.array([-1.0, 0.0])
        else:
            if self.phase != "TERMINAL" and (self.phase == "TOUCHDOWN" or self._go_terminal(s)):
                self.phase = "TERMINAL"
            out = self._step(s)
        self.k += 1
        self.k_since += 1
        return out

    def _go_terminal(self, s):
        """Switch to the terminal descent below the gate, or when the plan is about to end low."""
        short = self.plan is not None and self.plan["tgo"] - self.k_since / 50 < 0.6 and s[1] < 3.0
        return s[1] < self.y_gate + 0.1 or short

    def _step(self, s):
        if self.phase == "TERMINAL":
            return self._terminal(s)
        if self.k >= self.k_next and (self.plan is None or self.k_since >= self.replan_every):
            new = self._replan(s)
            self.k_next = self.k + (1 if new is not None else 10)        # cool-down after a failure
            if new is not None:
                self.plan, self.k_since = new, 0
            elif self.plan is not None and self.plan["tgo"] - self.k_since / 50 > 0.3:
                pass                                           # keep the last valid plan
            else:
                self.plan = None
        if self.plan is None:
            return self._fallback(s)
        return self._track(s)

    # ------------------------------------------------------------------ tracking and attitude
    def _reference(self, s):
        """Planned position, velocity and thrust acceleration at the current time (ZOH interpolation)."""
        P = self.plan
        t = self.k_since / 50
        i = min(int(t / P["dt"]), self.N - 1)
        tau = min(t - i * P["dt"], P["dt"])
        a = P["a"][i]
        acc = a + np.array([0.0, self.g])
        v_ref = P["v"][i] + tau * acc
        p_ref = P["p"][i] + tau * P["v"][i] + 0.5 * tau * tau * acc
        return i, p_ref, v_ref, a

    def _track(self, s):
        i, p_ref, v_ref, a = self._reference(s)
        self.log["track_err"].append(float(np.linalg.norm(p_ref - s[:2])))
        # TODO (Question 4): tracking law a_cmd = u_plan + Kp (r_ref - r) + Kd (v_ref - v), with u_plan = a, the reference (p_ref, v_ref) and the gains self.kp_trk, self.kd_trk
        raise NotImplementedError("Question 4")
        theta_ff = None
        if not self.hover_possible and np.linalg.norm(a) < 0.5 * self.a_min(s[6]):
            nxt = [k for k in range(i, self.N) if np.linalg.norm(self.plan["a"][k]) > 1e-3]
            ak = self.plan["a"][nxt[0]] if nxt else np.array([0.0, 1.0])
            theta_ff = np.arctan2(-ak[0], ak[1])         # coasting: point towards the next burn
        return self._to_action(s, a_cmd, theta_ff=theta_ff)

    def _throttle(self, s, a_cmd):
        """Throttle for a thrust acceleration a_cmd. Below the 50 % minimum the engine cannot run
        continuously: if the lander cannot hover, the demand is flown as 50 % pulses (sigma-delta)."""
        p = float(np.linalg.norm(a_cmd) * s[6] / self.p.F)
        if p >= 0.5 or self.hover_possible:
            self.sd_main = 0.0
            return float(np.clip(p, 0.5, 1.0))
        # the engine can only push along the body axis: a demand pointing downwards gets no thrust
        p = max(0.0, float(a_cmd @ np.array([-np.sin(s[4]), np.cos(s[4])]))) * s[6] / self.p.F
        self.sd_main += p
        if self.sd_main >= 0.5:
            self.sd_main -= 0.5
            return 0.5
        return 0.0

    def _to_action(self, s, a_cmd, engine=True, theta_ff=None, omega_ref=0.0, c_ff=0.0):
        theta_d = np.arctan2(-a_cmd[0], a_cmd[1]) if theta_ff is None else theta_ff
        theta_d = np.clip(theta_d, -self.theta_trk, self.theta_trk)
        throttle = self._throttle(s, a_cmd) if engine else 0.0
        return np.array([main_action(throttle), self._attitude(s, theta_d, omega_ref, c_ff)])

    def _attitude(self, s, theta_d, omega_ref=0.0, c_ff=0.0):
        """PD loop on the tilt angle with the side engines (+ optional feed-forward command c_ff).
        The side engines cannot fire below 50 % power: small commands are realised on average by a
        sigma-delta (pulse) modulator that fires +-50 % whenever the accumulated demand reaches 0.5."""
        kw = self.p.k_omega(s[6])
        alpha = self.kp * (theta_d - s[4]) + self.kd * (omega_ref - s[5])
        c = float(np.clip(c_ff + alpha / kw, -1.0, 1.0))
        if abs(c) >= 0.5:
            self.sd_acc = 0.0
            return side_action(c)
        self.sd_acc += c
        if abs(self.sd_acc) >= 0.5:
            out = 0.5 * np.sign(self.sd_acc)
            self.sd_acc -= out
            return side_action(out)
        return 0.0

    def _fallback(self, s):
        v_ref = np.array([np.clip(-0.5 * s[0], -1.5, 1.5), -np.clip(0.5 * s[1], 0.6, 3.0)])
        a_cmd = np.array([0.0, -self.g]) + np.array([1.5, 2.0]) * (v_ref - s[2:4])
        engine = self.hover_possible or s[3] < v_ref[1] - 0.3
        return self._to_action(s, a_cmd, engine=engine)

    def _terminal(self, s):
        v_ref = np.array([np.clip(-0.3 * s[0], -0.3, 0.3), -self.v_gate])
        a_cmd = np.array([0.0, -self.g]) + np.array([2.0, 3.0]) * (v_ref - s[2:4])
        engine = self.a_min(s[6]) < -self.g or s[3] < -self.v_gate      # cannot hover: pulse the engine
        return self._to_action(s, a_cmd, engine=engine)


def make_cascade(spec):
    return CascadeController(gravity=spec["gravity"], spec=spec)

