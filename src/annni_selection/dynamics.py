"""Explicit schedules, classical master equations and completely positive quantum maps.

A and B are ENERGY/h in GHz, time is microseconds, temperature is mK.
The thermal-reset bath is a phenomenological sensitivity model, not a microscopic
Davies, hybrid-noise or strong-coupling model of a D-Wave device.
"""
from dataclasses import dataclass
from functools import lru_cache
import json
import csv
import numpy as np
from scipy.constants import Boltzmann, Planck
from scipy.linalg import eigh
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.special import expit
from .io import file_hash
from .models import normalize, from_log_weights, pair_energy

KB_OVER_H_GHZ_PER_MK = Boltzmann / Planck / 1e12
GHZ_US_ANGULAR = 2 * np.pi * 1000


@dataclass
class Schedule:
    fraction: np.ndarray
    a_ghz: np.ndarray
    b_ghz: np.ndarray
    protocol: str
    provenance: str

    def __post_init__(self):
        self.fraction = np.asarray(self.fraction, float)
        self.a_ghz, self.b_ghz = np.asarray(self.a_ghz, float), np.asarray(self.b_ghz, float)
        if self.fraction.ndim != 1 or len(self.fraction) < 2 or self.a_ghz.shape != self.fraction.shape or self.b_ghz.shape != self.fraction.shape:
            raise ValueError("Schedule arrays must have equal lengths >= 2")
        if not np.isfinite([self.fraction, self.a_ghz, self.b_ghz]).all() or np.any(np.diff(self.fraction) <= 0):
            raise ValueError("Finite, strictly increasing time fractions are required")
        if self.fraction[0] != 0 or self.fraction[-1] != 1 or np.any(self.a_ghz < 0) or np.any(self.b_ghz < 0):
            raise ValueError("Schedule fractions must run 0..1 and A,B must be nonnegative")
        if self.protocol not in {"fast", "standard", "synthetic"} or not self.provenance:
            raise ValueError("Schedule protocol and calibration provenance are mandatory")

    @classmethod
    def from_csv(cls, path, protocol, provenance):
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        # fraction is elapsed-time fraction, NOT an unconverted hardware control coordinate.
        return cls([float(r["fraction"]) for r in rows], [float(r["A_GHz"]) for r in rows],
                   [float(r["B_GHz"]) for r in rows], protocol, provenance + "; sha256=" + file_hash(path))

    def at(self, fraction):
        if not 0 <= fraction <= 1:
            raise ValueError("Schedule queried outside [0,1]")
        return float(np.interp(fraction, self.fraction, self.a_ghz)), float(np.interp(fraction, self.fraction, self.b_ghz))


def snapshot_parameters(a_ghz, b_ghz, temperature_mk, energy_scale=1.0):
    if not np.isfinite([a_ghz, b_ghz, temperature_mk, energy_scale]).all() or a_ghz < 0 or min(b_ghz, temperature_mk, energy_scale) <= 0:
        raise ValueError("Positive B, temperature and scale are required for beta/Gamma")
    theta = KB_OVER_H_GHZ_PER_MK * temperature_mk
    return {"beta": energy_scale * b_ghz / (2 * theta),
            "gamma": a_ghz / (energy_scale * b_ghz), "x": a_ghz / (2 * theta)}


def classical_generator(space, energy_ghz, temperature_mk, rates):
    """Column generator. rates maps contiguous block length -> per-move rate / us."""
    if temperature_mk <= 0:
        raise ValueError("Temperature must be positive")
    energy = np.asarray(energy_ghz, float)
    if energy.shape != (space.size,) or not np.isfinite(energy).all():
        raise ValueError("Incorrect classical energy vector")
    theta = KB_OVER_H_GHZ_PER_MK * temperature_mk
    source = np.arange(space.size)
    row, col, data = [], [], []
    escape = np.zeros(space.size)
    for length, amplitude in rates.items():
        length = int(length)
        if not 1 <= length <= space.n or not np.isfinite(amplitude) or amplitude < 0:
            raise ValueError("Block lengths and rates must be valid and nonnegative")
        masks = {sum(int(space.bit_weights[(start + k) % space.n]) for k in range(length)) for start in range(space.n)}
        for mask in sorted(masks):
            destination = source ^ mask
            rate = amplitude * expit(-(energy[destination] - energy) / theta)
            row.extend(destination)
            col.extend(source)
            data.extend(rate)
            escape += rate
    row.extend(source)
    col.extend(source)
    data.extend(-escape)
    return coo_matrix((data, (row, col)), shape=(space.size, space.size)).tocsr()


def _settings(duration_us, steps, temperature_mk=None):
    if not np.isfinite(duration_us) or duration_us <= 0 or steps < 1:
        raise ValueError("Positive duration and step count are required")
    if temperature_mk is not None and (not np.isfinite(temperature_mk) or temperature_mk <= 0):
        raise ValueError("Temperature must be finite and positive")


def _probability_roundoff(p):
    p = np.asarray(p, float)
    if p.min() < -1e-10 or not np.isfinite(p).all():
        raise RuntimeError("Dynamical propagation produced an invalid probability")
    return normalize(np.maximum(p, 0))


def classical_evolve(space, schedule, duration_us, temperature_mk, *, energy_scale=1.0,
                     kappa=.5, amplitudes=None, exponents=None, steps=200, initial=None, fields=None):
    _settings(duration_us, steps, temperature_mk)
    amplitudes = amplitudes or {1: 1.0}
    exponents = exponents or {length: 0 for length in amplitudes}
    if set(map(int, amplitudes)) != set(map(int, exponents)):
        raise ValueError("Each block rate needs one exponent")
    p = np.full(space.size, 1 / space.size) if initial is None else normalize(initial)
    energy = space.energy(kappa) + pair_energy(space, fields=fields)
    a0 = schedule.a_ghz[0]
    dt = duration_us / steps
    for step in range(steps):
        a, b = schedule.at((step + .5) / steps)
        rates = {}
        for length, rate in amplitudes.items():
            alpha = exponents[length]
            if alpha < 0 or (alpha > 0 and a0 <= 0):
                raise ValueError("Invalid amplitude-suppression exponent or initial A")
            rates[int(length)] = rate * ((a / a0) ** alpha if alpha else 1)
        generator = classical_generator(space, energy_scale * b * energy / 2, temperature_mk, rates)
        p = expm_multiply(generator * dt, p, traceA=float(generator.diagonal().sum() * dt))
    return _probability_roundoff(p)


def quantum_evolve(space, schedule, duration_us, *, energy_scale=1.0, kappa=.5,
                   steps=200, initial=None, fields=None, temperature_mk=20,
                   reset_rate=0.0, dephasing_rate=0.0, return_density=False):
    """Midpoint Hamiltonian + symmetric CP splitting; rates in inverse microseconds.

    A nonzero reset rate implements d rho/dt = gamma (rho_beta Tr rho - rho).
    Local Z dephasing is separate. Slow-noise ensembles can average this map.
    Refine steps and check convergence for EVERY quantitative hardware result.
    """
    _settings(duration_us, steps, temperature_mk)
    if not np.isfinite([reset_rate, dephasing_rate, energy_scale]).all() or min(reset_rate, dephasing_rate) < 0 or energy_scale <= 0:
        raise ValueError("Invalid rates or scale")
    pure = initial is None or np.asarray(initial).ndim == 1
    state = np.ones(space.size, complex) / np.sqrt(space.size) if initial is None else np.asarray(initial, complex).copy()
    if pure:
        if state.shape != (space.size,) or not np.isclose(np.vdot(state, state), 1):
            raise ValueError("Initial state vector must be normalized")
        if reset_rate or dephasing_rate or return_density:
            state = np.outer(state, state.conj())
            pure = False
    else:
        if state.shape != (space.size, space.size) or not np.allclose(state, state.conj().T) or not np.isclose(np.trace(state), 1) or np.linalg.eigvalsh(state).min() < -1e-10:
            raise ValueError("Initial density matrix must be a quantum state")
    energy = space.energy(kappa) + pair_energy(space, fields=fields)
    distance = (space.spins[:, None, :] != space.spins[None, :, :]).sum(2) if dephasing_rate else None
    dt = duration_us / steps
    theta = KB_OVER_H_GHZ_PER_MK * temperature_mk
    for step in range(steps):
        a, b = schedule.at((step + .5) / steps)
        h = np.diag(energy_scale * b * energy / 2) - a * space.flip_adjacency / 2
        vals, vecs = eigh(h, check_finite=False)
        if pure:
            phase = np.exp(-1j * GHZ_US_ANGULAR * dt * (vals - vals.min()))
            state = vecs @ (phase * (vecs.T @ state))
        else:
            phase = np.exp(-.5j * GHZ_US_ANGULAR * dt * (vals - vals.min()))
            unitary = (vecs * phase) @ vecs.T
            state = unitary @ state @ unitary.conj().T
            if dephasing_rate:
                state *= np.exp(-dephasing_rate * dt * distance)
            if reset_rate:
                equilibrium = (vecs * from_log_weights(-(vals - vals.min()) / theta)) @ vecs.T
                survival = np.exp(-reset_rate * dt)
                state = survival * state + (1 - survival) * equilibrium
            if dephasing_rate:
                state *= np.exp(-dephasing_rate * dt * distance)
            state = unitary @ state @ unitary.conj().T
    if pure:
        return _probability_roundoff(abs(state) ** 2)
    state = (state + state.conj().T) / 2
    state /= np.trace(state)
    return state if return_density else _probability_roundoff(np.diag(state).real)


def noise_ensemble(propagator, field_samples, weights=None, **kwargs):
    fields = np.asarray(field_samples, float)
    if fields.ndim != 2 or not len(fields):
        raise ValueError("Supply explicit slow-field samples [sample,N]")
    weights = np.full(len(fields), 1 / len(fields)) if weights is None else normalize(weights)
    if len(weights) != len(fields):
        raise ValueError("Noise weights and field samples differ")
    return normalize(sum(w * propagator(fields=f, **kwargs) for f, w in zip(fields, weights)))


def convergence_check(propagator, steps=100, **kwargs):
    a, b = propagator(steps=steps, **kwargs), propagator(steps=2 * steps, **kwargs)
    return b, {"steps": 2 * steps, "previous_steps": steps, "output_TV_change": float(abs(a - b).sum() / 2),
               "interpretation": "successive refinement diagnostic, not an error bound"}


def predict_dynamics(space, spec, condition):
    """Inference adapter: shared rates/schedules, condition-specific time and scale.

    All predictions are pre-readout; inference applies its independent channel.
    Schedule bytes are fingerprinted in the cache key so a changed calibration is
    never served from an old cache entry.
    """
    from .states import StateSpace
    settings = dict(spec)
    schedule = settings["schedule"]
    settings["schedule_sha256"] = file_hash(schedule["path"])
    keys = {k: condition[k] for k in ("kappa", "energy_scale", "anneal_time_us", "fast_anneal", "device") if k in condition}
    return _predict_cached(space.n, json.dumps(settings, sort_keys=True), json.dumps(keys, sort_keys=True)).copy()


@lru_cache(maxsize=24)
def _predict_cached(n, spec_json, condition_json):
    from .states import StateSpace
    spec, condition = json.loads(spec_json), json.loads(condition_json)
    sch = spec.pop("schedule"); spec.pop("schedule_sha256")
    kind = spec.pop("kind")
    steps = spec.pop("steps", 100)
    tolerance = spec.pop("max_refinement_tv")
    if not 0 < tolerance < 1:
        raise ValueError("Specify a positive numerical refinement tolerance smaller than 1")
    schedule = Schedule.from_csv(sch["path"], sch["protocol"], sch["provenance"])
    if schedule.protocol == "synthetic" and condition.get("device") != "synthetic":
        raise ValueError("Synthetic schedules cannot be used as hardware calibration")
    if "fast_anneal" in condition and schedule.protocol != "synthetic":
        expected = "fast" if condition["fast_anneal"] else "standard"
        if schedule.protocol != expected:
            raise ValueError("Schedule protocol mismatch")
    if kind not in {"classical", "coherent", "thermal_reset"}:
        raise ValueError("Unknown dynamics family")
    if kind == "coherent" and (spec.get("reset_rate", 0) or spec.get("dephasing_rate", 0)):
        raise ValueError("Coherent dynamics cannot have dissipative rates")
    fn = classical_evolve if kind == "classical" else quantum_evolve
    if "amplitudes" in spec:
        spec["amplitudes"] = {int(k): v for k, v in spec["amplitudes"].items()}
        spec["exponents"] = {int(k): v for k, v in spec["exponents"].items()}
    fields = spec.pop("field_samples", None)
    settings = dict(space=StateSpace(n), schedule=schedule, duration_us=condition["anneal_time_us"],
                    energy_scale=condition.get("energy_scale", 1), kappa=condition.get("kappa", .5), **spec)
    if fields is None:
        result, check = convergence_check(fn, steps, **settings)
    else:
        def ensemble(**kw):
            return noise_ensemble(fn, fields, **kw)
        result, check = convergence_check(ensemble, steps, **settings)
    if check["output_TV_change"] > tolerance:
        raise ValueError(f"Dynamics need finer integration: successive-refinement TV={check['output_TV_change']:.6g}")
    return result
