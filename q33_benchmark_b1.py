"""Question 3.3 -- Benchmark B1 (20 frozen point-mass landing scenarios).

Place this file NEXT TO the ``lander/`` package in the starter kit, then run:
    python q33_benchmark_b1.py

It uses the Question 3.1 and 3.2 implementations already in lander/lcvx.py.
The existing project files lander/pointmass.py and lander/params.py are untouched.

Two runs are performed:
  1) certify=False: expose any non-lossless solutions and actuator violations;
  2) certify=True: reject non-lossless horizons during time-of-flight search.

Results are saved in q33_results/: per-case CSVs, summaries, figures and report.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import numpy as np

from lander.params import P
from lander.lcvx import LanderSOCP, is_lossless, solve_min_fuel, solve_min_landing_error
from lander.pointmass import B1_CASES, B1_UNREACHABLE, run_b1, simulate


# The B1 assignment specifies N=40, and the existing Q3.1/Q3.2 search defaults.
N_DEFAULT = 40
GRID_DEFAULT = 25
TOL_DEFAULT = 0.02
OUTPUT_COLUMNS = (
    'case', 'reachable', 'method', 'ok', 'landed', 'tgo_s', 'fuel_g',
    'landing_error_m', 'd_star_m', 'gap_max_mps2', 'gap_relative_max',
    'lossless', 'thrust_violation_N', 'glide_violation_m', 'solve_time_s',
)


def _state_key(x0, m0):
    """Use the full initial state and mass, not a mutable NumPy array, as key."""
    return tuple(float(x) for x in np.r_[x0, m0])


def _finite_float(value):
    """Use empty CSV cells for metrics absent when planning failed."""
    try:
        x = float(value)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


class BenchmarkGuidance:
    """Adapter from Q3.1/Q3.2 planners to run_b1's two callbacks.

    Keeps the minimum-fuel CVXPY problem compiled once (DPP), and caches
    fallback results so that d_star_fn never repeats the same Q3.2 search.
    """

    def __init__(self, certify: bool, N=N_DEFAULT, n_grid=GRID_DEFAULT, tol=TOL_DEFAULT):
        self.certify, self.N, self.n_grid, self.tol = certify, N, n_grid, tol
        self.fuel_socp = LanderSOCP(P, N=N)
        self.cache = {}

    def _two_stage(self, x0, m0):
        return solve_min_landing_error(
            P, x0, m0, N=self.N, n_grid=self.n_grid,
            tol=self.tol, certify=self.certify,
        )

    def guidance(self, x0, m0):
        """Fuel-optimal landing if possible; otherwise the Q3.2 two-stage plan."""
        key = _state_key(x0, m0)
        if key not in self.cache:
            sol, _ = solve_min_fuel(
                self.fuel_socp, x0, m0, n_grid=self.n_grid,
                tol=self.tol, certify=self.certify,
            )
            d_star, method = None, 'min_fuel'
            if sol is None:
                sol, d_star, _ = self._two_stage(x0, m0)
                method = 'min_landing_error'
            self.cache[key] = dict(sol=sol, d_star=d_star, method=method)
        return self.cache[key]['sol']

    def d_star_fn(self, x0, m0):
        """Return minimum miss distance for the three known unreachable cases.

        run_b1 calls this after guidance for cases 17--19.  If an uncertified
        min-fuel plan happened to look feasible, still independently calculate
        d_star for the prescribed benchmark acceptance threshold.
        """
        key = _state_key(x0, m0)
        if key not in self.cache:
            self.guidance(x0, m0)
        entry = self.cache[key]
        if entry['d_star'] is None:
            _sol, d_star, _ = self._two_stage(x0, m0)
            entry['d_star'] = d_star
        value = _finite_float(entry['d_star'])
        # No computed d* is not a license to count arbitrarily distant landings.
        return value if value is not None else 0.0


def run_pass(certify, N=N_DEFAULT, n_grid=GRID_DEFAULT, tol=TOL_DEFAULT):
    """Run the teacher's run_b1 unchanged, then enrich its 20 result rows."""
    planner = BenchmarkGuidance(certify=certify, N=N, n_grid=n_grid, tol=tol)
    t_start = time.perf_counter()
    official = run_b1(planner.guidance, planner.d_star_fn, p=P)
    elapsed = time.perf_counter() - t_start
    table, sols, rel_gaps = [], {}, np.full((len(B1_CASES), N), np.nan)

    for i, row in enumerate(official):
        case = B1_CASES[i]
        x0, m0 = case[:4], P.m_dry + case[4]
        entry = planner.cache.get(_state_key(x0, m0), {})
        sol = entry.get('sol')
        sols[i] = sol
        if sol is None:
            max_gap = relative = None
            lossless = False
        else:
            gap = np.maximum(sol.relaxation_gap, 0.0)
            relative_gaps = gap / np.maximum(sol.sigma, 1e-9)
            rel_gaps[i, :] = relative_gaps
            max_gap = float(np.max(gap))
            relative = float(np.max(relative_gaps))
            lossless = bool(is_lossless(sol))

        # In run_b1, the official timing is guidance() only; on unreachable
        # cases, an additional d_star_fn() can be called outside that timer.
        table.append({
            'case': i,
            'reachable': bool(i not in B1_UNREACHABLE),
            'method': entry.get('method', 'no_solution'),
            'ok': bool(row.get('ok', False)),
            'landed': bool(row.get('landed', False)),
            'tgo_s': _finite_float(sol.tgo if sol is not None else None),
            'fuel_g': _finite_float(1000.0 * row['fuel']) if 'fuel' in row else None,
            'landing_error_m': _finite_float(row.get('d_land')),
            'd_star_m': _finite_float(row.get('d_star')),
            'gap_max_mps2': max_gap,
            'gap_relative_max': relative,
            'lossless': lossless,
            'thrust_violation_N': _finite_float(row.get('viol_T')),
            'glide_violation_m': _finite_float(row.get('viol_gs')),
            'solve_time_s': float(row['t_solve']),
        })
    return dict(certify=certify, rows=table, solutions=sols,
                relative_gaps=rel_gaps, elapsed_s=elapsed)


def summarize(result):
    rows = result['rows']
    valid = [r for r in rows if r['ok']]
    def maximum(key):
        vals = [r[key] for r in valid if r[key] is not None]
        return max(vals) if vals else None
    def mean(key):
        vals = [r[key] for r in valid if r[key] is not None]
        return float(np.mean(vals)) if vals else None
    return dict(
        certified_search=result['certify'],
        cases=len(rows), solved=len(valid), landed=sum(r['landed'] for r in rows),
        reachable_landed=sum(r['landed'] for r in rows if r['reachable']),
        unreachable_landed=sum(r['landed'] for r in rows if not r['reachable']),
        plans_lossy=sum(not r['lossless'] for r in valid),
        mean_fuel_g=mean('fuel_g'), total_fuel_g=sum(r['fuel_g'] for r in valid),
        max_gap_mps2=maximum('gap_max_mps2'),
        max_relative_gap=maximum('gap_relative_max'),
        max_thrust_violation_N=maximum('thrust_violation_N'),
        max_glide_violation_m=maximum('glide_violation_m'),
        mean_guidance_time_s=float(np.mean([r['solve_time_s'] for r in rows])),
        max_guidance_time_s=float(np.max([r['solve_time_s'] for r in rows])),
        total_run_time_s=result['elapsed_s'],
    )


def save_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows):
    header = '| Case | Method | Landed | Fuel (g) | Miss (m) | Max gap (m/s²) | Viol. thrust (N) | Viol. glide (m) | Guidance (s) |'
    separator = '|---:|---|:---:|---:|---:|---:|---:|---:|---:|'
    lines = [header, separator]
    def num(x, digits):
        return '—' if x is None else f'{x:.{digits}f}'
    for r in rows:
        lines.append('| {case} | {method} | {landed} | {fuel} | {miss} | {gap} | {T} | {gs} | {sec} |'.format(
            case=r['case'], method=('fuel' if r['method'] == 'min_fuel' else 'error'),
            landed='Yes' if r['landed'] else 'No',
            fuel=num(r['fuel_g'], 2), miss=num(r['landing_error_m'], 3),
            gap=num(r['gap_max_mps2'], 5), T=num(r['thrust_violation_N'], 3),
            gs=num(r['glide_violation_m'], 4), sec=num(r['solve_time_s'], 3),
        ))
    return '\n'.join(lines)


def plot_results(uncertified, certified, out_dir):
    """Plots are produced only after both benchmark passes complete."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Nodal relative relaxation gap, one row per case and one column per node.
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.3), sharex=True, sharey=True,
                             constrained_layout=True)
    # Use a shared colour scale for a meaningful side-by-side comparison.
    all_finite = np.r_[uncertified['relative_gaps'].ravel(),
                       certified['relative_gaps'].ravel()]
    all_finite = all_finite[np.isfinite(all_finite)]
    common_vmax = max(0.01, float(np.max(all_finite)) if all_finite.size else 0.01)
    for ax, res, title in zip(axes, (uncertified, certified),
                              ('certify=False', 'certify=True')):
        heat = np.ma.masked_invalid(res['relative_gaps'])
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad(color='#dddddd')
        mesh = ax.imshow(heat, origin='lower', aspect='auto', interpolation='none',
                         cmap=cmap, vmin=0.0, vmax=common_vmax)
        ax.set_title(title)
        ax.set_xlabel('Time-step k')
        ax.set_yticks(np.arange(len(B1_CASES)))
        ax.set_ylabel('B1 case index')
        ax.axhline(16.5, color='white', ls='--', lw=1.0)
        fig.colorbar(mesh, ax=ax, shrink=0.78, label='max(0, σ − ||u||) / σ')
    fig.suptitle('Q3.3: nodal relaxation gaps across all B1 cases')
    fig.savefig(out_dir / 'b1_nodal_relaxation_gaps.png', dpi=180)
    plt.close(fig)

    # Pick the most clearly non-lossless flight for a concrete demonstration.
    lossy = [(r['gap_relative_max'], r['case']) for r in uncertified['rows']
             if r['ok'] and r['gap_relative_max'] is not None and r['gap_relative_max'] > 1e-4]
    if not lossy:
        return None
    _, case_index = max(lossy)
    sol = uncertified['solutions'][case_index]
    sim = simulate(sol, P, mode='accel')
    t_sub = np.linspace(0, sol.tgo, len(sim['thrust']), endpoint=False)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    ax = axes[0]
    ax.plot(t_sub, sim['thrust'], lw=1.5, label='Simulated physical thrust ||T(t)||')
    ax.axhline(P.rho1, color='red', ls='--', label='Minimum allowed thrust ρ₁')
    ax.axhline(P.rho2, color='gray', ls='--', label='Maximum allowed thrust ρ₂')
    ax.set_ylabel('Thrust (N)')
    ax.legend(loc='best', fontsize=8)
    ax = axes[1]
    ax.step(sol.t[:-1], np.maximum(sol.relaxation_gap, 0.0), where='post',
            label='Relaxation gap σₖ − ||uₖ||')
    ax.axhline(0.0, color='gray', lw=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Gap (m/s²)')
    ax.legend(loc='best', fontsize=8)
    fig.suptitle(f'B1 case {case_index}: an uncertified plan can demand impossible thrust')
    fig.savefig(out_dir / 'b1_lossy_case_thrust.png', dpi=180)
    plt.close(fig)
    return case_index


def report_text(uncertified, certified, example):
    a, b = summarize(uncertified), summarize(certified)
    def fmt(value, dp=3):
        return 'n/a' if value is None else f'{value:.{dp}f}'
    lines = [
        '# Q3.3 — Benchmark B1 (20 frozen initial states)',
        '',
        '## Method',
        'We use the Q3.1 fuel-optimal SOCP (`mode="fuel"`) and its one-dimensional '
        'time-of-flight search. If no acceptable pad-centre solution is found, '
        'we use the two-stage Q3.2 guidance: first minimize horizontal miss distance '
        '`d*`, then minimize fuel under `|x_N| <= d* + 0.001 m`. '
        'We evaluate each plan with the provided `run_b1` point-mass simulation '
        '(N=40). The three designated unreachable cases (17–19) are accepted '
        'when horizontal miss is at most `d* + 0.05 m`, provided touchdown is soft.',
        '',
        '## Comparison of certification modes',
        '',
        '| Metric | certify=False | certify=True |',
        '|---|---:|---:|',
        f'| Landed / 20 | {a["landed"]}/20 | {b["landed"]}/20 |',
        f'| Reachable cases landed / 17 | {a["reachable_landed"]}/17 | {b["reachable_landed"]}/17 |',
        f'| Unreachable cases landed / 3 | {a["unreachable_landed"]}/3 | {b["unreachable_landed"]}/3 |',
        f'| Solved plans | {a["solved"]} | {b["solved"]} |',
        f'| Non-lossless plans (relative gap > 1e-4) | {a["plans_lossy"]} | {b["plans_lossy"]} |',
        f'| Mean simulated fuel, solved cases (g) | {fmt(a["mean_fuel_g"],2)} | {fmt(b["mean_fuel_g"],2)} |',
        f'| Max thrust-bound violation (N) | {fmt(a["max_thrust_violation_N"],3)} | {fmt(b["max_thrust_violation_N"],3)} |',
        f'| Max glide-slope violation (m) | {fmt(a["max_glide_violation_m"],4)} | {fmt(b["max_glide_violation_m"],4)} |',
        f'| Mean guidance time (s/case) | {fmt(a["mean_guidance_time_s"],3)} | {fmt(b["mean_guidance_time_s"],3)} |',
        f'| Max guidance time (s/case) | {fmt(a["max_guidance_time_s"],3)} | {fmt(b["max_guidance_time_s"],3)} |',
        f'| Total wall-clock evaluation (s) | {fmt(a["total_run_time_s"],2)} | {fmt(b["total_run_time_s"],2)} |',
        '',
        '## Why the relaxation can become lossy',
        'The lossless-convexification theorem holds under its continuous-time '
        'regularity assumptions. Near the shortest feasible flight time, the '
        'minimum-thrust and tilt constraints can force the optimizer onto the '
        'flat boundary of the relaxed thrust region rather than its circular arc. '
        'At a discrete node this means `||u_k|| < σ_k`, and the planned '
        'pseudo-thrust `Γ_k = m_k σ_k` is larger than the force actually flown '
        '`||T_k|| = m_k ||u_k||`. In continuous time such exceptional instants '
        'can have zero measure, but with a zero-order hold each node spans an '
        'entire interval, making the discrepancy measurable.',
        '',
        'The provided simulator flies `u_k` in acceleration mode, even if its '
        'corresponding thrust is below the engine minimum `ρ₁`. The trajectory '
        'can therefore appear to touch down softly while demanding an impossible '
        'throttle. Its physical fuel burn also follows `||T||`, not the larger '
        'relaxed `Γ`. The benchmark `landed` flag checks final position and '
        'velocity, **not** whether thrust bounds were violated; report '
        '`viol_T` and the relaxation gap as well.',
        '',
        'Rejecting horizons with `max_k (σ_k - ||u_k||)/σ_k > 1e-4` '
        'prevents these uncertified plans from being reported as physically '
        'valid by the guidance search. A failed case is not silently counted '
        'as a successful landing.',
        '',
    ]
    if example is not None:
        rec = uncertified['rows'][example]
        lines += [
            f'**Illustration:** uncertified B1 case {example} has maximum '
            f'relative nodal gap {fmt(rec["gap_relative_max"],5)} and a '
            f'simulated minimum/maximum thrust-bound violation of '
            f'{fmt(rec["thrust_violation_N"],3)} N. '
            'See `b1_lossy_case_thrust.png`.',
            '',
        ]
    lines += [
        '## Full B1 table: uncertified search', '', markdown_table(uncertified['rows']), '',
        '## Full B1 table: certified search (final)', '', markdown_table(certified['rows']), '',
        '## Notes on reported times',
        '`run_b1` measures the elapsed time inside `guidance(x0, m0)`, '
        'which includes the Q3.2 fallback if needed. Any separate '
        '`d_star_fn(x0, m0)` computation occurs after that per-case timer, '
        'so we additionally report the full wall-clock time for each benchmark pass.',
        '',
        '## Generated figures',
        '- `b1_nodal_relaxation_gaps.png`: per-node relative gaps for all 20 cases and both modes.',
        '- `b1_lossy_case_thrust.png`: an illustrative invalid-throttle case if one was found.',
        '',
    ]
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=Path('q33_results'))
    ap.add_argument('--N', type=int, default=N_DEFAULT)
    ap.add_argument('--n-grid', type=int, default=GRID_DEFAULT)
    ap.add_argument('--tol', type=float, default=TOL_DEFAULT)
    args = ap.parse_args()
    if args.N < 2 or args.n_grid < 2:
        ap.error('N and n-grid must be >= 2')
    args.output.mkdir(parents=True, exist_ok=True)

    results = []
    for certify in (False, True):
        print(f'\nRunning B1 (20 cases), certify={certify} ...', flush=True)
        res = run_pass(certify, N=args.N, n_grid=args.n_grid, tol=args.tol)
        results.append(res)
        label = 'certified' if certify else 'uncertified'
        save_csv(args.output / f'b1_{label}.csv', res['rows'])
        s = summarize(res)
        print(f'  Landed {s["landed"]}/20 '
              f'(reachable {s["reachable_landed"]}/17, '
              f'unreachable {s["unreachable_landed"]}/3)')
        print(f'  Solved {s["solved"]}/20; lossy plans {s["plans_lossy"]}')
        print(f'  Mean fuel (solved) {s["mean_fuel_g"]} g; '
              f'max thrust violation {s["max_thrust_violation_N"]} N')
        print(f'  Mean guidance time {s["mean_guidance_time_s"]:.3f} s')
    example = plot_results(results[0], results[1], args.output)
    report = report_text(results[0], results[1], example)
    (args.output / 'q33_report.md').write_text(report, encoding='utf-8')
    print('\n' + report.split('## Full B1 table')[0])
    print(f'All tables, figures and the full report are in {args.output.resolve()}')


if __name__ == '__main__':
    main()
