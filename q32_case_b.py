"""Question 3.2: unreachable pad and two-stage landing guidance."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from lander.params import P
from lander.lcvx import LanderSOCP, solve_min_fuel, solve_min_landing_error, is_lossless
from lander.pointmass import simulate, glide_violation


def main():
    x0 = np.array([-6., 9., -6., 0.])
    m0 = P.m_dry + 0.17
    out = Path('q32_results'); out.mkdir(exist_ok=True)
    center = LanderSOCP(P, N=40)
    # Rejecting uncertified solutions alone does not establish infeasibility.
    relaxed, _ = solve_min_fuel(center, x0, m0, certify=False)
    print('Pad center reachable in relaxed search:', relaxed is not None)
    assert relaxed is None, 'A relaxed center solution was found; investigate.'
    sol, d_star, evaluations = solve_min_landing_error(P, x0, m0, N=40)
    if sol is None:
        raise RuntimeError(f'No certified two-stage solution; Stage 1 d* = {d_star}')
    sim = simulate(sol, P, mode='accel')
    report = '\n'.join([
        f'Stage 1 minimum miss distance: {d_star:.9f} m',
        f'Flight time: {sol.tgo:.9f} s',
        f'Stage 2 miss distance: {sol.landing_error:.9f} m',
        f'Fuel used: {1000*sol.fuel:.6f} g',
        f'Fuel remaining: {1000*(0.17-sol.fuel):.6f} g',
        f'Lossless: {is_lossless(sol)}',
        f'Maximum relative gap: {np.max(sol.relaxation_gap/np.maximum(sol.sigma,1e-9)):.9g}',
        f'Simulated position error from planned touchdown: {sim["final_pos_err"]:.9g} m',
        f'Simulated terminal speed: {sim["final_vel_err"]:.9g} m/s',
        f'Thrust violation: {sim["thrust_violation"]:.9g} N',
        f'Glide-slope violation: {glide_violation(sim["traj"], 1.):.9g} m',
    ])
    print(report); (out/'summary.txt').write_text(report+'\n')
    np.savetxt(out/'time_search.csv', evaluations, delimiter=',', header='tf,landing_error', comments='')
    fig, ax = plt.subplots()
    ax.plot(sol.p[:,0],sol.p[:,1], 'o-', markersize=2,label='SOCP trajectory')
    ax.plot(sim['traj'][:,0],sim['traj'][:,1], '--',label='Simulation')
    xx = np.linspace(min(sol.p[:,0].min(),sol.p[-1,0])-2, max(0.,sol.p[:,0].max())+1,300)
    ax.plot(xx,abs(xx-sol.p[-1,0]), ':',label='Glide slope')
    ax.scatter([0,sol.p[-1,0]],[0,0],label='Pad / touchdown')
    ax.set(xlabel='Horizontal position [m]',ylabel='Altitude [m]',title='Case B: minimum landing error'); ax.set_aspect('equal'); ax.legend(); fig.tight_layout(); fig.savefig(out/'trajectory.png',dpi=200)
    fig, ax = plt.subplots()
    vals = np.asarray(evaluations); mask = np.isfinite(vals[:,1]); ax.plot(vals[mask,0],vals[mask,1],'o-'); ax.scatter(sol.tgo,d_star,label='Best evaluated time'); ax.set(xlabel='Flight time [s]',ylabel='Miss distance [m]'); ax.legend(); fig.tight_layout(); fig.savefig(out/'error_vs_time.png',dpi=200)
    plt.close('all')

if __name__ == '__main__':
    main()
