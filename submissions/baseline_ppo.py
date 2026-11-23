"""Baseline: PPO (stable-baselines3) trained for 2M steps on the nominal B2 condition of the fuel-tank lander
with Gymnasium's original LunarLander reward (scripts/train_rl_baseline.py)."""
import os
import numpy as np
from stable_baselines3 import PPO

_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "ppo_fuel_lander.zip")


class PPOPolicy:
    def __init__(self):
        self.model = PPO.load(_MODEL, device="cpu")

    def reset(self):
        pass

    def act(self, obs, info=None):
        a, _ = self.model.predict(np.asarray(obs, np.float32), deterministic=True)
        return a


def make_policy(spec):
    return PPOPolicy()
