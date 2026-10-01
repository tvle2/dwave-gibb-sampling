"""Normalized latent distributions. Average full distributions before conditioning."""
from functools import lru_cache
import numpy as np
from scipy.linalg import eigh
from scipy.special import logsumexp
from .states import StateSpace


def normalize(p):
    p = np.asarray(p, float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any(p < 0) or p.sum() <= 0:
        raise ValueError("Invalid probability vector")
    return p / p.sum()


def from_log_weights(log_weights):
    a = np.asarray(log_weights, float)
    if not np.isfinite(a).all():
        raise ValueError("Log weights must be finite")
    return np.exp(a - logsumexp(a))


def pair_energy(space, fields=None, couplings=None):
    energy = np.zeros(space.size)
    if fields is not None:
        fields = np.asarray(fields, float)
        if fields.shape != (space.n,) or not np.isfinite(fields).all():
            raise ValueError("fields must have N finite entries")
        energy += space.spins @ fields
    if couplings is not None:
        couplings = np.asarray(couplings, float)
        if couplings.shape != (space.n, space.n) or not np.isfinite(couplings).all() or not np.allclose(couplings, couplings.T):
            raise ValueError("couplings must be a finite symmetric N by N matrix")
        for i in range(space.n):
            for j in range(i + 1, space.n):
                energy += couplings[i, j] * space.spins[:, i] * space.spins[:, j]
    return energy


def classical_gibbs(space, beta, kappa=0.5, fields=None, couplings=None):
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    return from_log_weights(-beta * (space.energy(kappa) + pair_energy(space, fields, couplings)))


@lru_cache(maxsize=24)
def _spectrum(n, kappa, gamma):
    space = StateSpace(n)
    return eigh(np.diag(space.energy(kappa)) - gamma * space.flip_adjacency, check_finite=False)


def quantum_gibbs(space, beta, gamma, kappa=0.5, fields=None, couplings=None):
    if not np.isfinite([beta, gamma, kappa]).all() or min(beta, gamma) < 0:
        raise ValueError("beta and gamma must be finite and nonnegative")
    if fields is None and couplings is None:
        values, vectors = _spectrum(space.n, float(kappa), float(gamma))
    else:
        energy = space.energy(kappa) + pair_energy(space, fields, couplings)
        values, vectors = eigh(np.diag(energy) - gamma * space.flip_adjacency)
    weights = from_log_weights(-beta * (values - values.min()))
    return normalize((vectors * vectors) @ weights)


def projected_gibbs(space, x, sector_weights=None):
    space.require_eight()
    if not np.isfinite(x) or x < 0:
        raise ValueError("x must be finite and nonnegative")
    ids = np.flatnonzero(space.ground)
    values, vectors = _projected_spectrum()
    p = np.zeros(space.size)
    p[ids] = (vectors * vectors) @ from_log_weights(x * (values - values.max()))
    if sector_weights is not None:
        weights = normalize(np.asarray(sector_weights))
        if weights.shape != (3,):
            raise ValueError("sector_weights must be ordered F,T,A")
        for weight, name in zip(weights, ("F", "T", "A")):
            mask = space.sector_masks[name]
            if name in {"F", "A"}:
                p[mask] = weight / mask.sum()
            else:
                p[mask] *= weight / p[mask].sum()
    return normalize(p)


@lru_cache(maxsize=1)
def _projected_spectrum():
    s = StateSpace()
    return eigh(s.flip_adjacency[np.ix_(s.ground, s.ground)])


def conditional(p, mask):
    return normalize(np.asarray(p)[mask])


def mixture(probabilities, weights):
    ps = np.array([normalize(p) for p in probabilities])
    weights = normalize(weights)
    if len(weights) != len(ps):
        raise ValueError("A mixture weight is required for each full distribution")
    return normalize(weights @ ps)


def emulating_pairwise(space, r1, r2):
    space.require_eight()
    a, b = (r2 - r1) / 4, -r2 / 4
    return from_log_weights(a * space.correlations[:, 2] + b * space.correlations[:, 3])


def susceptibility_gibbs(space, beta, chi, energy_scale=1.0, kappa=0.5):
    """beta is normalized beta_c = lambda B/(2 kBT); J is in programmed units."""
    if not np.isfinite([beta, chi, energy_scale, kappa]).all() or beta < 0 or energy_scale <= 0:
        raise ValueError("Invalid susceptibility parameters")
    j = np.zeros((space.n, space.n))
    for i in range(space.n):
        for distance, strength in ((1, -energy_scale), (2, kappa * energy_scale)):
            k = (i + distance) % space.n
            j[i, k] = j[k, i] = strength
    return from_log_weights(-(beta / energy_scale) * pair_energy(space, couplings=j + chi * (j @ j)))


def distribution(space, spec, condition=None):
    """Evaluate an explicit model spec; no parameter is inferred from test data."""
    condition = condition or {}
    spec = dict(spec)
    family = spec.pop("family")
    kappa = spec.pop("kappa", condition.get("kappa", 0.5))
    if family == "classical":
        return classical_gibbs(space, kappa=kappa, **spec)
    if family == "quantum":
        return quantum_gibbs(space, kappa=kappa, **spec)
    if family == "projected":
        if kappa != 0.5:
            raise ValueError("The projected model is restricted to kappa=0.5")
        return projected_gibbs(space, **spec)
    if family == "susceptibility":
        spec.setdefault("energy_scale", condition.get("energy_scale", 1.0))
        return susceptibility_gibbs(space, kappa=kappa, **spec)
    if family == "susceptibility_transfer":
        scale = condition.get("energy_scale", 1.0)
        beta_per_scale = spec.pop("beta_per_scale")
        return susceptibility_gibbs(space, beta=scale * beta_per_scale,
                                    energy_scale=scale, kappa=kappa, **spec)
    if family == "physical_snapshot":
        from .dynamics import snapshot_parameters
        parameters = snapshot_parameters(spec["A_GHz"], spec["B_GHz"], spec["temperature_mK"],
                                          condition.get("energy_scale", 1.0))
        return quantum_gibbs(space, parameters["beta"], parameters["gamma"], kappa)
    if family == "dynamics":
        from .dynamics import predict_dynamics
        return predict_dynamics(space, spec, {**condition, "kappa": kappa})
    if family == "tabulated":
        if not spec.get("provenance"):
            raise ValueError("Tabulated physical predictions need provenance")
        p = normalize(spec["probabilities"])
        if p.shape != (space.size,):
            raise ValueError("Tabulated prediction has the wrong state space")
        for key, value in spec.get("condition", {}).items():
            if condition.get(key) != value:
                raise ValueError(f"Tabulated prediction condition mismatch: {key}")
        return p
    if family == "uniform_field":
        beta, eta = spec.pop("beta"), spec.pop("eta")
        if spec:
            raise ValueError(f"Unknown parameters {list(spec)}")
        return from_log_weights(-beta * space.energy(kappa) + eta * space.spins.sum(axis=1))
    if family == "pairwise":
        theta = np.asarray(spec["theta"], float)
        if theta.shape != (space.pair_features.shape[1],):
            raise ValueError("Incorrect pairwise feature vector")
        return from_log_weights(space.pair_features @ theta)
    if family == "mixture":
        return mixture([distribution(space, s, condition) for s in spec["components"]], spec["weights"])
    raise ValueError(f"Unknown model family: {family}")
