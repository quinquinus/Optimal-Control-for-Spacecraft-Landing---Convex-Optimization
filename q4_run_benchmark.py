"""Run Question 4's Benchmark B2 for the cascade and both reference policies.

From the project root:
    python q4_run_benchmark.py --n 10      # quick smoke test
    python q4_run_benchmark.py --n 100     # final report (500 episodes/policy)

Runs all policies on the same frozen B2 seed range. Produces JSON episode
results and a text/Markdown comparison with per-condition metrics.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

POLICIES = {
    "LCvx cascade": "lander.controllers:make_cascade",
    "Heuristic": "submissions/baseline_heuristic.py",
    "PPO": "submissions/baseline_ppo.py",
}
CONDITIONS = ("nominal", "heavy", "light", "gusty", "storm")


def percentage(x):
    return f"{100*x:.1f}%"


def number(x, digits=2, suffix=""):
    return (f"{x:.{digits}f}{suffix}" if math.isfinite(x) else "n/a")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="B2 seeds per condition (100 for final report)")
    parser.add_argument("--processes", type=int, default=2)
    parser.add_argument("--out-dir", default="q4_results")
    args = parser.parse_args()
    if not 1 <= args.n <= 100:
        parser.error("--n must be in [1,100], since B2 has exactly 100 frozen seeds")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    summaries = {}
    for name, source in POLICIES.items():
        slug = name.lower().replace(" ", "_")
        result = out_dir / f"b2_{slug}_n{args.n}.json"
        cmd = [sys.executable, "-m", "lander.evaluate", "--policy", source,
               "--benchmark", "b2", "--n", str(args.n),
               "--processes", str(args.processes), "--out", str(result)]
        print(f"\n=== {name} ({args.n} seeds / condition) ===", flush=True)
        subprocess.run(cmd, check=True)
        summaries[name] = json.loads(result.read_text())["summary"]

    lines = [f"# Question 4: Benchmark B2 (n={args.n} seeds per condition)", "",
             "All three controllers were evaluated on the same five conditions and frozen seeds.",
             "Fuel = mean seconds of full main-engine thrust among successful episodes.",
             "Time = mean computation time per controller step in milliseconds.",
             "", "## Overall", "",
             "| Policy | Success | Mean score | Fuel (s, successes) | Time (ms/step) |",
             "|---|---:|---:|---:|---:|"]
    for name, s in summaries.items():
        a = s["overall"]
        lines.append(f"| {name} | {percentage(a['success'])} | {number(a['score'])} | "
                     f"{number(a['fuel_success'])} | {number(a['policy_ms'])} |")
    lines.extend(["", "## By environment", "",
                  "| Environment | Policy | Success | Fuel (s, successes) | Time (ms/step) | Off-pad | Crash/hard |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for c in CONDITIONS:
        for name, s in summaries.items():
            a = s["per_scenario"][c]
            lines.append(f"| {c} | {name} | {percentage(a['success'])} | "
                         f"{number(a['fuel_success'])} | {number(a['policy_ms'])} | "
                         f"{percentage(a['off_pad'])} | {percentage(a['crash']+a['hard'])} |")
    lines.extend(["", "## Interpretation guide", "",
                  "Success depends on the combined feasibility of convex guidance and the rigid-body attitude loop.",
                  "Higher replanning time is the cost of repeatedly solving a DPP SOCP; look at mean latency alongside reliability.",
                  "Fuel is measured only among successful episodes; do not compare it in isolation from the success rate.",
                  "The light-gravity case additionally uses relaxed engine-off planning with sigma-delta thrust pulses.", ""])
    report = out_dir / f"Q4_B2_comparison_n{args.n}.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nComparison created: {report.resolve()}")


if __name__ == "__main__":
    main()
