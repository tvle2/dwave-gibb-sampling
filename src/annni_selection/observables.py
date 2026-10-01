"""Arithmetic contrasts and log-mean invariants are deliberately distinct."""
import numpy as np
from .models import normalize


def summarize(space, values):
    space.require_eight()
    p = normalize(values)
    mass = {name: float(p[mask].sum()) for name, mask in space.orbits.items()}
    mean = {name: mass[name] / int(mask.sum()) for name, mask in space.orbits.items()}
    logs = {name: float(np.log(p[mask]).mean()) if np.all(p[mask] > 0) else None
            for name, mask in space.orbits.items()}
    r1 = float(np.log(mean["53"] / mean["62"])) if min(mean["53"], mean["62"]) > 0 else None
    r2 = float(np.log(mean["44"] / mean["53"])) if min(mean["44"], mean["53"]) > 0 else None
    w = float(space.w_coefficients[space.w_coefficients != 0] @ np.log(p[space.w_coefficients != 0])) if np.all(p[space.w_coefficients != 0] > 0) else None
    il = lock = None
    if all(v is not None for v in logs.values()):
        lt = (2 * logs["62"] + 2 * logs["53"] + logs["44"]) / 5
        il = logs["F"] + logs["A"] - 2 * lt
        lock = -3.2 * (logs["53"] - logs["62"]) - 2.4 * (logs["44"] - logs["53"])
    return {"P_T": float(p[space.target].sum()), "P_G": float(p[space.ground].sum()),
            "orbit_mass": mass, "orbit_log_mean": logs, "R1": r1, "R2": r2,
            "D_chi_kappa_half": r2 + r1 / 3 if r1 is not None and r2 is not None else None,
            "I_L": il, "I_L_pair": lock, "W": w,
            "class_mass": np.bincount(space.class_id, weights=p).tolist(),
            "global_flip_l1": float(abs(p - p[::-1]).sum() / 2),
            "within_orbit_l1": {k: float(abs(p[m] - p[m].mean()).sum() / 2) for k, m in space.orbits.items()}}


def w_variance_coefficient(space, p):
    """Coefficient n_G Var(W), conditional on G, for independent reads only."""
    p = normalize(p)
    pg = p / p[space.ground].sum()
    mask = space.w_coefficients != 0
    if np.any(pg[mask] <= 0):
        return float("inf")
    return float(np.sum(space.w_coefficients[mask] ** 2 / pg[mask]))


def contrast_covariance(space, p, n):
    """Independent-read delta-method planning covariance, not cluster uncertainty."""
    p = normalize(p)
    if n <= 0:
        raise ValueError("n must be positive")
    gradients = []
    for terms in ({"53": 1, "62": -1}, {"44": 1, "53": -1}):
        g = np.zeros(space.size)
        for name, sign in terms.items():
            mask = space.orbits[name]
            if p[mask].sum() <= 0:
                raise ValueError("Zero orbit mass")
            g[mask] = sign / p[mask].sum()
        gradients.append(g)
    g = np.asarray(gradients)
    return ((g * p) @ g.T - np.outer(g @ p, g @ p)) / n


def kl_decomposition(space, p, q):
    p, q = normalize(p), normalize(q)
    if np.any(q <= 0):
        raise ValueError("Reference must be strictly positive")
    q_class = np.bincount(space.class_id, weights=q)
    counts = np.bincount(space.class_id)
    if not np.allclose(q, (q_class / counts)[space.class_id]):
        raise ValueError("Reference must be uniform within each (C1,C2) class")
    p_class = np.bincount(space.class_id, weights=p)
    m = p > 0
    mc = p_class > 0
    total = float(np.sum(p[m] * np.log(p[m] / q[m])))
    occupancy = float(np.sum(p_class[mc] * np.log(p_class[mc] / q_class[mc])))
    return {"total": total, "class_occupancy": occupancy, "within_class": total - occupancy}
