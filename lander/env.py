"""FuelLander: Gymnasium's LunarLander (same body, legs, engines and actions) with
  * a fuel tank: the mass drops as the main engine burns, and the engine stops when the tank is empty;
  * a scenario: terrain, landing pad, hazards (rocks and no-fly zones), gravity, wind, initial state.

Observation: the standard 8-dim LunarLander vector, with (x, y) relative to the pad centre at landing
height (same scaling: x/10 m, y/6.667 m, vx/5 m/s, vy/7.5 m/s, angle, omega/2.5, leg contacts).
info["state"] = [x, y, vx, vy, theta, omega, m] in SI units (pad frame, m = current mass in kg).
Actions: as LunarLanderContinuous (main engine off for a[0] <= 0, otherwise 50-100 %; side engines off
for |a[1]| <= 0.5).

Wind (horizontal acceleration while airborne):  w(t, y) = mean + shear (y - y_pad) / 10 + gust(t),
gust = Ornstein-Uhlenbeck process (std gust_std, correlation time gust_tau); plus an OU torque.

Episode outcome (info["outcome"] at termination):
  success   at rest on both legs with the lander's centre above the pad (x1 <= x <= x2) and a
            first-contact vertical speed <= 2 m/s
  off_pad   at rest (safely) elsewhere          hard     first-contact vertical speed > 2 m/s
  hazard    any part of the lander touched a hazard (rock or no-fly zone)
  crash     body touched the terrain            out      left the map
  timeout   max_steps reached
"""
import math
import numpy as np
import Box2D
from Box2D.b2 import edgeShape, fixtureDef, polygonShape, revoluteJointDef
from gymnasium.envs.box2d.lunar_lander import (LunarLander, ContactDetector, SCALE, VIEWPORT_W, VIEWPORT_H,
                                               LANDER_POLY, LEG_AWAY, LEG_DOWN, LEG_W, LEG_H, LEG_SPRING_TORQUE)

from .params import P as PARAMS, BODY_MASS_DENSITY5
from .scenarios import Scenario, SCENARIOS, B2_CONDITIONS, W, H

FPS = 50
HARD_LANDING = 2.0     # m/s
ALL_SCENARIOS = {**SCENARIOS, **B2_CONDITIONS}


class FuelLander(LunarLander):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, scenario="nominal", render_mode=None, params=PARAMS):
        self.scenario: Scenario = ALL_SCENARIOS[scenario] if isinstance(scenario, str) else scenario
        self.params = params
        super().__init__(render_mode=render_mode, continuous=True, gravity=-10.0, enable_wind=False)
        self.gravity = self.scenario.gravity          # used by reset() to build the world
        self.hazard_bodies = []

    # ------------------------------------------------------------------ world construction
    def _destroy(self):
        for b in getattr(self, "hazard_bodies", []):
            self.world.DestroyBody(b)
        self.hazard_bodies = []
        super()._destroy()

    def reset(self, *, seed=None, options=None):
        import gymnasium as gym
        gym.Env.reset(self, seed=seed)
        self._destroy()
        sc = self.scenario
        self.world = Box2D.b2World(gravity=(0, self.gravity))
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over, self.prev_shaping = False, None

        # terrain + pad
        self.helipad_x1, self.helipad_x2, self.helipad_y = sc.pad
        self.moon = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
        self.sky_polys = []
        for p1, p2 in zip(sc.terrain[:-1], sc.terrain[1:]):
            self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])
        self.moon.color1 = self.moon.color2 = (0, 0, 0)

        # hazards: static bodies (rocks collide, zones are sensors)
        for hz in sc.hazards:
            b = self.world.CreateStaticBody(
                fixtures=fixtureDef(shape=polygonShape(vertices=list(hz.poly)), isSensor=(hz.kind == "zone"),
                                    friction=0.3))
            b.color1 = (220, 70, 60) if hz.kind == "rock" else (250, 175, 160)
            b.color2 = (150, 30, 30) if hz.kind == "rock" else (230, 110, 90)
            b.hazard = hz
            self.hazard_bodies.append(b)

        # lander at the (randomized) initial state, with a full tank
        rng = self.np_random
        x0, y0, vx0, vy0 = (np.array(sc.start) + rng.normal(size=4) * np.array(sc.start_std))
        self.lander = self.world.CreateDynamicBody(
            position=(x0, y0), angle=0.0,
            fixtures=fixtureDef(shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]),
                                density=5.0, friction=0.1, categoryBits=0x0010, maskBits=0x001, restitution=0.0))
        self.lander.color1, self.lander.color2 = (128, 102, 230), (77, 77, 128)
        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(x0 - i * LEG_AWAY / SCALE, y0), angle=(i * 0.05),
                fixtures=fixtureDef(shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)), density=1.0,
                                    restitution=0.0, categoryBits=0x0020, maskBits=0x001))
            leg.ground_contact = False
            leg.color1, leg.color2 = (128, 102, 230), (77, 77, 128)
            rjd = revoluteJointDef(bodyA=self.lander, bodyB=leg, localAnchorA=(0, 0),
                                   localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE), enableMotor=True,
                                   enableLimit=True, maxMotorTorque=LEG_SPRING_TORQUE, motorSpeed=+0.3 * i)
            if i == -1:
                rjd.lowerAngle, rjd.upperAngle = +0.9 - 0.5, +0.9
            else:
                rjd.lowerAngle, rjd.upperAngle = -0.9, -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)
        self.fuel_mass = self.params.m_fuel
        self._set_mass()
        for b in [self.lander] + self.legs:
            b.linearVelocity = (vx0, vy0)
        self.drawlist = [self.lander] + self.legs + self.hazard_bodies

        # wind state (OU gust, turbulence torque), drawn from the episode seed
        self.gust = rng.normal() * sc.wind.gust_std
        self.torq = rng.normal() * sc.wind.torque_std
        self.t, self.k = 0.0, 0
        self.fuel, self.side_fuel = 0.0, 0.0
        self.touch = None
        self.outcome = None
        self.wind_now = 0.0
        if self.render_mode == "human":
            self.render()
        obs, _, _, _, info = self.step(np.array([0.0, 0.0]))
        self.k, self.t, self.fuel = 0, 0.0, 0.0
        self.fuel_mass = self.params.m_fuel
        self._set_mass()
        info["state"][6] = self.total_mass
        return obs, info

    def _set_mass(self):
        """Put the remaining fuel into the body (uniform density: the centre of mass does not move)."""
        p = self.params
        f = self.lander.fixtures[0]
        f.density = 5.0 * (p.m_body_dry + self.fuel_mass) / BODY_MASS_DENSITY5
        self.lander.ResetMassData()
        self.total_mass = self.lander.mass + sum(l.mass for l in self.legs)

    # ------------------------------------------------------------------ dynamics
    def _wind_accel(self):
        w = self.scenario.wind
        y = self.lander.position.y
        return w.mean + w.shear * (y - self.scenario.pad[2]) / 10.0 + self.gust

    def step(self, action):
        sc, w, p = self.scenario, self.scenario.wind, self.params
        dt = 1.0 / FPS
        airborne = not (self.legs[0].ground_contact or self.legs[1].ground_contact)
        self.wind_now = self._wind_accel()
        if airborne:
            self.lander.ApplyForceToCenter((self.total_mass * self.wind_now, 0.0), True)
            if w.torque_std > 0:
                self.lander.ApplyTorque(self.torq, True)
        # OU updates (exact discretization)
        rng = self.np_random
        if w.gust_std > 0:
            a = math.exp(-dt / w.gust_tau)
            self.gust = a * self.gust + w.gust_std * math.sqrt(1 - a * a) * rng.normal()
        if w.torque_std > 0:
            a = math.exp(-dt / 0.5)
            self.torq = a * self.torq + w.torque_std * math.sqrt(1 - a * a) * rng.normal()

        action = np.clip(np.asarray(action, dtype=np.float64), -1, 1).copy()
        if self.fuel_mass <= 0.0:
            action[0] = -1.0                                   # empty tank: the main engine is off
        pm = 0.0 if action[0] <= 0 else (action[0] + 1) * 0.5
        cs = 0.0 if abs(action[1]) <= 0.5 else abs(action[1])
        self.fuel += pm * dt
        self.side_fuel += cs * dt

        obs, ll_reward, terminated, _, _ = super().step(action)
        self.k += 1
        self.t += dt
        if pm > 0:                                             # burn fuel: dm/dt = -alpha * F * throttle
            self.fuel_mass = max(0.0, self.fuel_mass - p.alpha * p.F * pm * dt)
            self._set_mass()

        # pad-frame observation
        pos, vel = self.lander.position, self.lander.linearVelocity
        obs = obs.copy()
        obs[0] = (pos.x - sc.pad_center) / (VIEWPORT_W / SCALE / 2)
        state = np.array([pos.x - sc.pad_center, pos.y - (sc.pad[2] + LEG_DOWN / SCALE), vel.x, vel.y,
                          self.lander.angle, self.lander.angularVelocity, self.total_mass])

        if self.touch is None and (self.legs[0].ground_contact or self.legs[1].ground_contact):
            self.touch = dict(vx=vel.x, vy=vel.y, theta=self.lander.angle, x=state[0])

        outcome = None
        if self._hazard_contact():
            outcome = "hazard"
        elif self.game_over:
            outcome = "crash"
        elif pos.x < 0 or pos.x > W or pos.y > H + 3:
            outcome = "out"
        elif not self.lander.awake:
            on_pad = (self.helipad_x1 <= pos.x <= self.helipad_x2) and all(l.ground_contact for l in self.legs)
            hard = self.touch is not None and abs(self.touch["vy"]) > HARD_LANDING
            outcome = "hard" if hard else ("success" if on_pad else "off_pad")
        elif self.k >= sc.max_steps:
            outcome = "timeout"
        terminated = outcome is not None and outcome != "timeout"
        truncated = outcome == "timeout"
        if outcome is not None:
            self.outcome = outcome
        info = dict(state=state, t=self.t, outcome=outcome, fuel=self.fuel, side_fuel=self.side_fuel,
                    fuel_left=self.fuel_mass, touch=self.touch, ll_reward=ll_reward)
        if self.render_mode == "human":
            self.render()
        return obs.astype(np.float32), score_step(outcome, self.fuel), terminated, truncated, info

    def _hazard_contact(self):
        parts = [self.lander] + self.legs
        for b in self.hazard_bodies:
            for ce in b.contacts:
                c = ce.contact
                if c.touching and (c.fixtureA.body in parts or c.fixtureB.body in parts):
                    return True
        return False

    # ------------------------------------------------------------------ rendering
    def render(self):
        img = super().render()
        if self.render_mode != "rgb_array" or img is None:
            return img
        return annotate(img, self)


def score_step(outcome, fuel):
    """Reward = episode score, paid once at termination (see score_episode)."""
    if outcome is None:
        return 0.0
    return score_episode(outcome, fuel)


def score_episode(outcome, fuel):
    """Leaderboard score of one episode (fuel in seconds of full main-engine thrust)."""
    base = {"success": 100.0, "off_pad": 20.0, "hazard": -50.0}.get(outcome, 0.0)
    return base - 5.0 * fuel


def annotate(img, env):
    """Wind arrow, fuel gauge, pad highlight and scenario label on the rendered frame (for videos)."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(np.ascontiguousarray(img))
    d = ImageDraw.Draw(im)
    sc = env.scenario
    x1, x2, py = sc.pad
    Y = lambda y: VIEWPORT_H - y * SCALE
    d.rectangle([x1 * SCALE, Y(py) - 2, x2 * SCALE, Y(py) + 2], fill=(27, 175, 122))
    wv = getattr(env, "wind_now", 0.0)
    cx, cy = 300, 22
    d.line([cx, cy, cx + 20 * wv, cy], fill=(110, 170, 240), width=4)
    d.polygon([(cx + 20 * wv, cy - 6), (cx + 20 * wv + 8 * np.sign(wv), cy), (cx + 20 * wv, cy + 6)],
              fill=(110, 170, 240))
    d.text((8, 6), f"{sc.sid} {sc.name}", fill=(235, 235, 235))
    d.text((8, 20), f"t={env.t:4.1f}s", fill=(235, 235, 235))
    frac = env.fuel_mass / env.params.m_fuel
    d.rectangle([520, 8, 590, 18], outline=(235, 235, 235))
    d.rectangle([521, 9, 521 + 68 * frac, 17], fill=(240, 190, 60))
    d.text((480, 7), "fuel", fill=(235, 235, 235))
    d.text((cx - 30, 32), f"wind {wv:+.1f} m/s2", fill=(110, 170, 240))
    if env.outcome:
        d.text((8, 34), env.outcome.upper(), fill=(240, 90, 80) if env.outcome != "success" else (60, 200, 120))
    return np.asarray(im)
