"""Scenarios of the project: the five Benchmark-B2 conditions (Question 4) and the ten maps of the
Windy Hazard Landing Challenge (Question 5).

Everything here is PUBLIC information given to the students: terrain, pad, hazard polygons,
gravity, initial-state distribution and the STATISTICS of the wind. Only the wind and
initial-state REALIZATIONS (drawn from the episode seed) are unknown to the controller.

World coordinates: x in [0, 20] m (left to right), y up, as in Gymnasium's LunarLander
(VIEWPORT 600 x 400 px at 30 px/m). Hazards are convex polygons (counter-clockwise vertices):
  kind="rock"  solid static body: any contact with the lander (body or legs) = hazard failure
  kind="zone"  no-fly zone (sensor, no collision): entering it = hazard failure
Wind (horizontal acceleration, m/s^2, applied to the whole lander while airborne):
  w(t, y) = mean + shear * (y - pad_y) / 10 + gust(t),
  gust = Ornstein-Uhlenbeck process with stationary std `gust_std` and correlation time `gust_tau`.
A turbulence torque (OU, std `torque_std` N m, correlation 0.5 s) perturbs the attitude.
"""
from dataclasses import dataclass, field
import numpy as np

W, H = 20.0, 400 / 30.0


@dataclass(frozen=True)
class Wind:
    mean: float = 0.0
    shear: float = 0.0
    gust_std: float = 0.0
    gust_tau: float = 1.5
    torque_std: float = 0.0


@dataclass(frozen=True)
class Hazard:
    kind: str                      # "rock" | "zone"
    poly: tuple                    # ((x, y), ...) world coordinates, convex, CCW
    name: str = ""


@dataclass(frozen=True)
class Scenario:
    sid: str
    name: str
    description: str
    terrain: tuple                 # ((x, y), ...) from x=0 to x=20, piecewise linear
    pad: tuple                     # (x1, x2, y)
    hazards: tuple = ()
    wind: Wind = field(default_factory=Wind)
    gravity: float = -10.0
    start: tuple = (10.0, 11.0, 0.0, -1.0)   # x0, y0, vx0, vy0 (mean)
    start_std: tuple = (0.3, 0.2, 0.5, 0.5)
    max_steps: int = 1000

    @property
    def pad_center(self):
        return 0.5 * (self.pad[0] + self.pad[1])

    def to_pad_frame(self, pts):
        """World points -> pad frame used by observations (origin at the pad centre, landing height)."""
        pts = np.asarray(pts, float)
        return pts - np.array([self.pad_center, self.pad[2] + 18 / 30.0])

    def public_dict(self):
        return dict(sid=self.sid, name=self.name, description=self.description,
                    terrain=[list(p) for p in self.terrain], pad=list(self.pad),
                    hazards=[dict(kind=h.kind, name=h.name, poly=[list(p) for p in h.poly]) for h in self.hazards],
                    wind=self.wind.__dict__, gravity=self.gravity, start=list(self.start),
                    start_std=list(self.start_std), max_steps=self.max_steps)


def _terrain(heights, pad, flat_extra=0.0):
    """11 terrain knots every 2 m; the pad interval (+ flat_extra each side) is made flat."""
    xs = np.linspace(0, W, len(heights))
    ys = np.array(heights, float)
    pts = list(zip(xs, ys))
    x1, x2, py = pad
    # insert the flat pad segment exactly
    out = [p for p in pts if p[0] < x1 - flat_extra - 1e-6]
    out += [(x1 - flat_extra, py), (x2 + flat_extra, py)]
    out += [p for p in pts if p[0] > x2 + flat_extra + 1e-6]
    return tuple((round(float(x), 3), round(float(y), 3)) for x, y in out)


def _box(x1, y1, x2, y2):
    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


def _rock(xc, w, h, base=0.0, top_w=None):
    """Trapezoidal rock/spire standing on y=base (extends 1 m below the surface)."""
    tw = w if top_w is None else top_w
    return ((xc - w / 2, base - 1.0), (xc + w / 2, base - 1.0), (xc + tw / 2, base + h), (xc - tw / 2, base + h))


def make_scenarios():
    S = []
    # S01 -------------------------------------------------------------------------------------
    pad = (8.5, 11.5, 3.0)
    S.append(Scenario(
        "S01", "Tranquility Base", "Warm-up: light breeze, two rocks well away from the pad.",
        _terrain([4.0, 3.6, 3.3, 3.1, 3.0, 3.0, 3.0, 3.1, 3.4, 3.8, 4.2], pad), pad,
        (Hazard("rock", _rock(5.0, 0.9, 0.9, 3.0), "rock A"), Hazard("rock", _rock(14.8, 1.0, 1.1, 3.25), "rock B")),
        Wind(mean=0.3, gust_std=0.3, gust_tau=2.0), -10.0, (6.0, 11.0, 0.5, -1.0)))
    # S02 -------------------------------------------------------------------------------------
    pad = (12.5, 15.0, 4.5)
    S.append(Scenario(
        "S02", "Crosswind Mesa", "Pad on a mesa; strong steady wind to the left pushes you onto a boulder.",
        _terrain([2.5, 2.5, 2.6, 2.8, 3.0, 3.6, 4.5, 4.5, 4.5, 4.5, 4.0], pad, 0.5), pad,
        (Hazard("rock", _rock(11.3, 1.0, 1.4, 4.5, 0.6), "boulder"),),
        Wind(mean=-1.8, gust_std=0.4, gust_tau=2.0), -10.0, (7.0, 11.0, 1.0, -1.0)))
    # S03 -------------------------------------------------------------------------------------
    pad = (6.0, 8.5, 3.0)
    S.append(Scenario(
        "S03", "Gusty Crater", "Narrow pad between two crater rims; strong, fast gusts.",
        _terrain([4.5, 4.0, 3.5, 3.0, 3.0, 3.0, 3.2, 3.5, 3.9, 4.3, 4.8], pad, 1.5), pad,
        (Hazard("rock", _box(4.0, 2.0, 5.0, 3.35), "crater rim W"), Hazard("rock", _box(9.5, 2.0, 10.5, 3.35), "crater rim E")),
        Wind(mean=0.5, gust_std=1.2, gust_tau=1.0, torque_std=0.3), -10.0, (12.0, 11.0, -0.5, -1.0)))
    # S04 -------------------------------------------------------------------------------------
    pad = (14.5, 17.0, 3.0)
    S.append(Scenario(
        "S04", "Spire Pass", "A tall rock spire stands between the start and the pad: fly over it.",
        _terrain([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0], pad, 0.5), pad,
        (Hazard("rock", _rock(11.0, 1.6, 5.5, 3.0, 0.6), "spire"),),
        Wind(mean=0.8, gust_std=0.5, gust_tau=1.5), -10.0, (5.0, 11.5, 0.5, -0.5)))
    # S05 -------------------------------------------------------------------------------------
    pad = (9.0, 11.0, 2.0)
    S.append(Scenario(
        "S05", "Canyon", "Pad at the bottom of a 3.4 m wide canyon; wind shear is strong above the rims.",
        _terrain([5.5, 5.5, 5.5, 5.5, 2.0, 2.0, 2.0, 5.5, 5.5, 5.5, 5.5], pad, 1.0), pad,
        (Hazard("rock", ((6.0, 1.0), (8.3, 1.0), (8.3, 6.5), (6.0, 6.5)), "west wall"),
         Hazard("rock", ((11.7, 1.0), (14.0, 1.0), (14.0, 6.5), (11.7, 6.5)), "east wall")),
        Wind(mean=0.2, shear=2.0, gust_std=0.4, gust_tau=1.5), -10.0, (4.0, 11.5, 0.5, -0.5)))
    # S06 -------------------------------------------------------------------------------------
    pad = (13.0, 15.5, 3.0)
    S.append(Scenario(
        "S06", "Plume", "A no-fly dust plume hangs over the direct path: go over it, or under it.",
        _terrain([3.2, 3.1, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.1, 3.3], pad, 0.5), pad,
        (Hazard("zone", _box(7.0, 5.2, 11.8, 8.8), "plume"),),
        Wind(mean=0.6, gust_std=0.6, gust_tau=1.5), -10.0, (5.5, 11.5, 0.8, -0.8)))
    # S07 -------------------------------------------------------------------------------------
    pad = (7.0, 9.5, 3.0)
    S.append(Scenario(
        "S07", "Phobos Drift", "Low gravity (-6 m/s^2): minimum thrust exceeds weight, so the engine must be cycled.",
        _terrain([3.5, 3.4, 3.2, 3.0, 3.0, 3.0, 3.0, 3.2, 3.4, 3.6, 3.8], pad, 0.5), pad,
        (Hazard("rock", _rock(5.6, 0.7, 0.8, 3.0), "rock W"), Hazard("rock", _rock(11.3, 0.8, 0.9, 3.0), "rock E")),
        Wind(mean=1.0, gust_std=0.5, gust_tau=2.0), -6.0, (12.5, 11.5, -0.5, -0.5)))
    # S08 -------------------------------------------------------------------------------------
    pad = (9.0, 11.5, 5.0)
    S.append(Scenario(
        "S08", "Heavy Ledge", "High gravity (-11.5 m/s^2) and a pad on a ledge above a cliff.",
        _terrain([3.0, 3.0, 3.5, 4.2, 5.0, 5.0, 5.0, 2.0, 2.0, 2.0, 2.0], pad, 0.0), pad,
        (Hazard("rock", ((12.3, 1.0), (14.5, 1.0), (14.5, 3.0), (12.3, 3.0)), "cliff base"),
         Hazard("rock", _rock(8.0, 0.6, 0.7, 5.0), "rock")),
        Wind(mean=-1.0, gust_std=0.8, gust_tau=1.5), -11.5, (14.0, 11.5, -0.5, -0.5)))
    # S09 -------------------------------------------------------------------------------------
    pad = (2.5, 5.0, 3.0)
    S.append(Scenario(
        "S09", "Boulder Field", "Pad near the left edge behind a boulder field; wind blows you off the map.",
        _terrain([3.2, 3.0, 3.0, 3.0, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6], pad, 0.5), pad,
        (Hazard("rock", _rock(6.5, 0.7, 0.9, 3.0), "boulder 1"), Hazard("rock", _rock(7.9, 0.6, 1.3, 3.05), "boulder 2"),
         Hazard("rock", _rock(9.2, 0.8, 0.8, 3.1), "boulder 3")),
        Wind(mean=-1.2, gust_std=0.6, gust_tau=1.5), -10.0, (12.0, 11.5, -0.5, -0.5)))
    # S10 -------------------------------------------------------------------------------------
    pad = (15.5, 18.0, 3.5)
    S.append(Scenario(
        "S10", "Storm Gauntlet", "Spire, plume, shear and strong gusts together.",
        _terrain([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.2, 3.5, 3.5, 3.5, 3.5], pad, 0.5), pad,
        (Hazard("rock", _rock(8.0, 1.4, 4.8, 3.0, 0.5), "spire"),
         Hazard("zone", _box(11.0, 6.2, 14.5, 9.0), "plume"),
         Hazard("rock", _rock(14.3, 0.6, 0.7, 3.5), "rock")),
        Wind(mean=1.0, shear=1.0, gust_std=1.0, gust_tau=1.0, torque_std=0.3), -10.0, (3.0, 11.5, 0.5, -0.5)))
    return {s.sid: s for s in S}


SCENARIOS = make_scenarios()


def make_b2_conditions():
    """Benchmark B2 (Question 4): flat ground, a 4 m pad in the middle of the map (as in Gymnasium's
    LunarLander), no hazards. The five conditions differ in gravity and wind."""
    pad = (8.0, 12.0, H / 4)
    terrain = ((0.0, H / 4), (8.0, H / 4), (12.0, H / 4), (W, H / 4))
    start, std = (10.0, 12.0, 0.0, -1.0), (2.0, 0.3, 1.5, 0.8)
    mk = lambda sid, name, desc, wind, g: Scenario(sid, name, desc, terrain, pad, (), wind, g, start, std)
    C = [mk("nominal", "Nominal", "Gravity -10 m/s^2, no wind.", Wind(), -10.0),
         mk("heavy", "Heavy", "Gravity -12 m/s^2, no wind: little thrust margin with a full tank.", Wind(), -12.0),
         mk("light", "Light", "Gravity -6 m/s^2: the minimum thrust exceeds the weight, the engine cannot hover.",
            Wind(), -6.0),
         mk("gusty", "Gusty", "Gusts (std 1.0 m/s^2, correlation 1.5 s), no mean wind.",
            Wind(gust_std=1.0, gust_tau=1.5), -10.0),
         mk("storm", "Storm", "Steady crosswind 1 m/s^2, shear, strong gusts and a turbulence torque.",
            Wind(mean=1.0, shear=1.0, gust_std=1.2, gust_tau=1.0, torque_std=0.3), -10.0)]
    return {c.sid: c for c in C}


B2_CONDITIONS = make_b2_conditions()
B2_SEEDS = range(5000, 5100)

# Seed splits. TRAIN: use freely. PUBLIC: shown on the public leaderboard (and runnable locally).
# PRIVATE: drawn on the server from a secret salt; revealed only at the end (Kaggle-style).
TRAIN_SEEDS = range(0, 1000)
PUBLIC_SEEDS = range(1000, 1010)


def private_seeds(salt: str, n=30):
    import hashlib
    h = hashlib.sha256(salt.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    return [int(s) for s in rng.integers(10_000, 2**31 - 1, size=n)]
