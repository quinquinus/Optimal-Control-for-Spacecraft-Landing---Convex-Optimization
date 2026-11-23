"""Train the RL baseline (instructor-provided): PPO on the fuel-tank lander, nominal B2 condition.

Reward: Gymnasium's original LunarLander reward (distance/velocity shaping, leg contacts, engine use,
+-100 at the end), so that the agent learns the standard task with the heavier, fuel-burning lander.
Hyperparameters from rl-baselines3-zoo for LunarLanderContinuous.
Usage: python scripts/train_rl_baseline.py [timesteps]
"""
import os, sys, time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lander.env import FuelLander  # noqa: E402


class ShapedReward(gym.Wrapper):
    def step(self, action):
        obs, _, term, trunc, info = self.env.step(action)
        return obs, info["ll_reward"], term, trunc, info


def make(i):
    def f():
        env = ShapedReward(FuelLander("nominal"))
        env.reset(seed=1000 * i)
        return env
    return f


if __name__ == "__main__":
    steps = int(float(sys.argv[1])) if len(sys.argv) > 1 else 2_000_000
    out = os.path.join(ROOT, "models")
    os.makedirs(out, exist_ok=True)
    env = DummyVecEnv([make(i) for i in range(16)])
    model = PPO("MlpPolicy", env, n_steps=1024, batch_size=64, gae_lambda=0.98, gamma=0.999, n_epochs=4,
                ent_coef=0.01, verbose=0, seed=0, device="cpu")
    t0, chunk, done = time.time(), 100_000, 0
    while done < steps:
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        done += chunk
        model.save(os.path.join(out, "ppo_fuel_lander"))
        print(f"{done} steps, {time.time() - t0:.0f} s", flush=True)
