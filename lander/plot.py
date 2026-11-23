"""Map / trajectory plots for the hazard scenarios (used for debugging and for the report figures)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from .style import C
from .scenarios import W, H

LEG_DOWN = 18 / 30.0


def draw_map(ax, sc, frame="world", label=True):
    off = np.zeros(2) if frame == "world" else np.array([sc.pad_center, sc.pad[2] + LEG_DOWN])
    T = np.array(sc.terrain) - off
    ax.fill_between(T[:, 0], T[:, 1], -5 - off[1], color="#d9d7d0", lw=0)
    ax.plot(T[:, 0], T[:, 1], color=C["ink2"], lw=1)
    for h in sc.hazards:
        P = np.array(h.poly) - off
        ax.add_patch(Polygon(P, closed=True, fc=C["red"] if h.kind == "rock" else "#f6b8b0",
                             ec=C["red"], lw=1, alpha=0.9 if h.kind == "rock" else 0.6,
                             hatch=None if h.kind == "rock" else "///"))
    x1, x2, py = sc.pad
    ax.plot([x1 - off[0], x2 - off[0]], [py - off[1]] * 2, color=C["aqua"], lw=4, solid_capstyle="butt")
    ax.set_xlim(0 - off[0], W - off[0]); ax.set_ylim(max(-1, min(T[:, 1]) - 1), H - off[1])
    ax.set_aspect("equal")
    if label:
        ax.set_title(f"{sc.sid} {sc.name}", fontsize=9)


def plot_episode(ax, sc, traj, color=None, lw=1.5, label=None, frame="world"):
    off = np.zeros(2) if frame == "world" else np.array([sc.pad_center, sc.pad[2] + LEG_DOWN])
    xy = traj[:, :2] + np.array([sc.pad_center, sc.pad[2] + LEG_DOWN]) - off
    ax.plot(xy[:, 0], xy[:, 1], color=color or C["blue"], lw=lw, label=label)
    ax.plot(*xy[0], "o", color=color or C["blue"], ms=3)
    return xy
