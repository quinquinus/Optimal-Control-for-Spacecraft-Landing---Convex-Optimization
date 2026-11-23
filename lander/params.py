"""Physical parameters of the lander, shared by the simulator, the models and the controllers.

The rigid body is Gymnasium's LunarLander (Box2D works in SI units: m, kg, s, N). We add a fuel tank:
the fuel is stored in the lander body (uniformly, so the centre of mass does not move) and burns in
proportion to the main-engine throttle. A lighter lander accelerates more, exactly like a rocket.

Values marked (measured) were measured once in the Box2D simulator (scripts/measure_params.py).
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class LanderParams:
    m_dry: float = 4.50           # kg, body + legs without fuel
    m_body_dry: float = 4.358     # kg, body only without fuel (the legs weigh 0.142 kg)
    m_fuel: float = 0.80          # kg, full tank (15 % of the full mass)
    F: float = 87.0               # N, mean main-engine force at full throttle               (measured)
    F_side: float = 12.27         # N, lateral force of one side engine at full power        (measured)
    k_omega_dry: float = -5.40    # rad/s^2, angular acceleration per unit side command, dry (measured)
    burn_time: float = 10.0       # s of full thrust that empty a full tank
    g: float = -10.0              # m/s^2 (each scenario sets its own gravity)
    throttle_min: float = 0.5     # the engine is off or runs between 50 % and 100 %

    @property
    def m_wet(self):
        return self.m_dry + self.m_fuel

    @property
    def alpha(self):
        """Fuel use per unit thrust (kg / (N s)):  dm/dt = -alpha * ||T||."""
        return self.m_fuel / (self.F * self.burn_time)

    @property
    def rho1(self):
        return self.throttle_min * self.F

    @property
    def rho2(self):
        return self.F

    def k_omega(self, m):
        """Angular acceleration per unit side command at mass m (the inertia grows with the fuel)."""
        return self.k_omega_dry * self.m_dry / m

    def with_(self, **kw):
        d = dict(self.__dict__)
        d.update(kw)
        return LanderParams(**d)


P = LanderParams()

BODY_MASS_DENSITY5 = 4.8167       # kg, mass of the LunarLander body polygon at Box2D density 5 (Gymnasium's value)

# Geometry used for hazard avoidance (Question 5): half-width of the lander including the feet when tilted
# by up to ~20 deg, and distance from the centre to the feet (plus a small margin).
R_X, R_Y = 1.0, 1.15
LEG_DOWN = 18 / 30.0              # m, height of the lander centre above the ground at touchdown
