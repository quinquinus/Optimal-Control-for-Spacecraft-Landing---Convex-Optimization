"""Leaderboard baseline: Gymnasium's built-in hand-tuned PD lander (ignores wind, hazards and pad offset)."""
from types import SimpleNamespace
from gymnasium.envs.box2d.lunar_lander import heuristic


class Heuristic:
    def __init__(self):
        self._env = SimpleNamespace(continuous=True)
        self._env.unwrapped = self._env

    def reset(self):
        pass

    def act(self, obs, info=None):
        return heuristic(self._env, obs)


def make_policy(spec):
    return Heuristic()
