"""Evaluation harness: Benchmark B2 (Question 4) and the Windy Hazard Landing Challenge (Question 5).
The same code runs locally and on the leaderboard server.

A policy is created by a factory  make_policy(spec: dict) -> policy, where spec is the public scenario
description (Scenario.public_dict()) and the policy has
        policy.reset()                  called at the start of every episode
        policy.act(obs, info) -> action  (np.array of shape (2,), LunarLanderContinuous convention)
The factory is given either as a file (a submission that defines make_policy) or as "module:function".

Usage:
    python -m lander.evaluate --policy lander.controllers:make_cascade --benchmark b2
    python -m lander.evaluate --policy submissions/reference.py --split public --out results/x.json
    python -m lander.evaluate --policy submission.py --scenarios S05 --gif          # animation -> gifs/
    python -m lander.evaluate --policy submission.py --scenarios S05,S10 --seed 3 --gif my_gifs
With --gif, nothing is evaluated: one episode per scenario is recorded as an animated GIF (real-time speed),
with the first seed of the chosen split unless --seed is given.
"""
import argparse, importlib, importlib.util, json, os, sys, time
from multiprocessing import Pool
import numpy as np

from .env import FuelLander, score_episode
from .scenarios import SCENARIOS, PUBLIC_SEEDS, B2_CONDITIONS, B2_SEEDS, private_seeds

OUTCOMES = ["success", "off_pad", "hazard", "crash", "hard", "out", "timeout"]
ALL = {**SCENARIOS, **B2_CONDITIONS}


def load_submission(path):
    spec = importlib.util.spec_from_file_location("submission", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec.loader.exec_module(mod)
    return mod


def get_factory(source):
    if source.endswith(".py"):
        return load_submission(source).make_policy
    mod, fn = source.split(":")
    return getattr(importlib.import_module(mod), fn)


def run_episode(env, policy, seed, record=False, every=2):
    obs, info = env.reset(seed=seed)
    policy.reset()
    frames, traj = [], []
    t_pol = 0.0
    for k in range(env.scenario.max_steps + 5):
        t0 = time.perf_counter()
        a = np.asarray(policy.act(obs, info), float).reshape(2)
        t_pol += time.perf_counter() - t0
        traj.append(np.r_[info["state"], a, env.wind_now])
        if record and k % every == 0:
            frames.append(env.render())
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            break
    if record:
        for _ in range(10):
            frames.append(env.render())
    outcome = info["outcome"] or "timeout"
    touch = info["touch"] or {}
    res = dict(scenario=env.scenario.sid, seed=int(seed), outcome=outcome,
               score=score_episode(outcome, env.fuel), fuel=float(env.fuel), side_fuel=float(env.side_fuel),
               fuel_left=float(env.fuel_mass), steps=k + 1, t=float(info["t"]),
               vy_td=float(touch.get("vy", np.nan)), vx_td=float(touch.get("vx", np.nan)),
               x_td=float(touch.get("x", np.nan)), policy_ms=1000 * t_pol / (k + 1))
    return res, frames, np.array(traj)


def _worker(args):
    import warnings
    warnings.filterwarnings("ignore")
    source, sid, seeds = args
    make_policy = get_factory(source)
    sc = ALL[sid]
    env = FuelLander(sc)
    policy = make_policy(sc.public_dict())
    rows = []
    for s in seeds:
        r, _, _ = run_episode(env, policy, s)
        rows.append(r)
    env.close()
    return rows


def evaluate(source, seeds, sids, processes=2):
    jobs = [(source, sid, list(seeds)) for sid in sids]
    with Pool(processes) as pool:
        out = pool.map(_worker, jobs)
    return [r for rows in out for r in rows]


def save_gif(frames, path, every=3):
    """Write the frames of run_episode(..., record=True, every=every) as a looping GIF at real-time speed."""
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=round(1000 * every / 50), loop=0)


def animate(source, sids, seed, out_dir="gifs", every=3):
    """Record one episode per scenario (or B2 condition) and save it as out_dir/<policy>_<scenario>_seed<seed>.gif."""
    import warnings
    warnings.filterwarnings("ignore")
    os.makedirs(out_dir, exist_ok=True)
    make_policy = get_factory(source)
    name = os.path.splitext(os.path.basename(source.split(":")[-1]))[0]
    paths = []
    for sid in sids:
        sc = ALL[sid]
        env = FuelLander(sc, render_mode="rgb_array")
        policy = make_policy(sc.public_dict())
        res, frames, _ = run_episode(env, policy, seed, record=True, every=every)
        env.close()
        path = os.path.join(out_dir, f"{name}_{sid}_seed{seed}.gif")
        save_gif(frames, path, every)
        print(f"{sid:8s}: {res['outcome']:8s} score {res['score']:6.1f}  ->  {path}")
        paths.append(path)
    return paths


def summarize(rows):
    def agg(rs):
        d = dict(score=float(np.mean([r["score"] for r in rs])), n=len(rs))
        for o in OUTCOMES:
            d[o] = float(np.mean([r["outcome"] == o for r in rs]))
        succ = [r for r in rs if r["outcome"] == "success"]
        d["fuel_success"] = float(np.mean([r["fuel"] for r in succ])) if succ else float("nan")
        d["vy_success"] = float(np.mean([abs(r["vy_td"]) for r in succ])) if succ else float("nan")
        d["policy_ms"] = float(np.mean([r["policy_ms"] for r in rs]))
        return d
    per = {sid: agg([r for r in rows if r["scenario"] == sid]) for sid in dict.fromkeys(r["scenario"] for r in rows)}
    return dict(overall=agg(rows), per_scenario=per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", "--submission", dest="policy", required=True)
    ap.add_argument("--benchmark", default=None, choices=[None, "b2"])
    ap.add_argument("--split", default="public", choices=["public", "train", "private"])
    ap.add_argument("--n", type=int, default=None, help="number of seeds (train split, or B2)")
    ap.add_argument("--scenarios", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--processes", type=int, default=2)
    ap.add_argument("--gif", nargs="?", const="gifs", default=None, metavar="DIR",
                    help="record one episode per scenario as an animated GIF in DIR (default: gifs) instead of evaluating")
    ap.add_argument("--seed", type=int, default=None, help="seed of the animated episode (with --gif)")
    a = ap.parse_args()
    if a.benchmark == "b2":
        seeds, sids = list(B2_SEEDS)[:a.n or 100], list(B2_CONDITIONS)
    else:
        sids = list(SCENARIOS)
        if a.split == "public":
            seeds = list(PUBLIC_SEEDS)
        elif a.split == "train":
            seeds = list(range(a.n or 10))
        else:
            seeds = private_seeds(os.environ["PDG_PRIVATE_SALT"], a.n or 30)
    if a.scenarios:
        sids = a.scenarios.split(",")
    if a.gif is not None:
        animate(a.policy, sids, seeds[0] if a.seed is None else a.seed, a.gif)
        return
    t0 = time.time()
    rows = evaluate(a.policy, seeds, sids, a.processes)
    summ = summarize(rows)
    summ["wall_s"] = time.time() - t0
    print(json.dumps(summ["overall"], indent=1))
    for sid, d in summ["per_scenario"].items():
        print(f"{sid:8s}: score {d['score']:7.1f}  success {d['success']:.2f}  hazard {d['hazard']:.2f}  "
              f"crash {d['crash'] + d['hard']:.2f}  off_pad {d['off_pad']:.2f}  out {d['out']:.2f}  "
              f"timeout {d['timeout']:.2f}  fuel {d['fuel_success']:.2f}  ms {d['policy_ms']:.1f}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(summary=summ, episodes=rows), f, indent=1, default=float)


if __name__ == "__main__":
    main()
