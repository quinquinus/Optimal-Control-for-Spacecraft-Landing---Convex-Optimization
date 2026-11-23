"""Geometry helpers for hazard avoidance (Question 5). Pad frame: origin at the pad centre, landing height."""
import numpy as np

from .params import LEG_DOWN


def to_pad(spec, pts):
    """World coordinates -> pad frame."""
    pts = np.asarray(pts, float)
    c = 0.5 * (spec["pad"][0] + spec["pad"][1])
    return pts - np.array([c, spec["pad"][2] + LEG_DOWN])


def closest_point_convex(poly, p):
    """Closest point q of a convex polygon (counter-clockwise vertices) to the point p.
    Returns (q, signed distance d (< 0 inside), unit normal n pointing from the polygon to p)."""
    P = np.asarray(poly, float)
    n = len(P)
    best, inside, min_pen, pen_normal, pen_pt = None, True, np.inf, None, None
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        e = b - a
        nrm = np.array([e[1], -e[0]]) / np.linalg.norm(e)        # outward normal of the edge
        sd = nrm @ (p - a)
        if sd > 0:
            inside = False
        if -sd < min_pen:
            min_pen, pen_normal, pen_pt = -sd, nrm, p - sd * nrm
        t = np.clip((p - a) @ e / (e @ e), 0, 1)
        q = a + t * e
        d = np.linalg.norm(p - q)
        if best is None or d < best[0]:
            best = (d, q)
    if inside:
        return pen_pt, -min_pen, pen_normal
    d, q = best
    return q, d, (p - q) / max(d, 1e-9)


def obstacle_polygons(spec):
    """All hazards of a scenario as convex polygons in the pad frame, with their kind:
    'ground' = terrain trapezoids (each terrain segment and the ground below it) and rocks,
    'zone'   = floating no-fly zones."""
    polys, kinds = [], []
    terr = to_pad(spec, spec["terrain"]) if len(spec["terrain"]) else np.zeros((0, 2))
    for a, b in zip(terr[:-1], terr[1:]):
        if b[0] - a[0] > 1e-6:
            polys.append(np.array([[a[0], -30.0], [b[0], -30.0], b, a]))
            kinds.append("ground")
    for h in spec["hazards"]:
        polys.append(to_pad(spec, h["poly"]))
        kinds.append("ground" if h["kind"] == "rock" else "zone")
    return polys, kinds
