"""Independent asymmetric channel and separate known-input calibration controls."""
import csv
from pathlib import Path
import numpy as np
from .io import canonical_hash, file_hash, read_json, write_json, output_directory
from .models import normalize


class ReadoutChannel:
    def __init__(self, minus_to_plus, plus_to_minus):
        self.minus = np.atleast_1d(np.asarray(minus_to_plus, float))
        self.plus = np.atleast_1d(np.asarray(plus_to_minus, float))
        if self.minus.shape != self.plus.shape or self.minus.ndim != 1:
            raise ValueError("Rate vectors must have equal one-dimensional shapes")
        if not np.isfinite([self.minus, self.plus]).all() or np.any(np.array([self.minus, self.plus]) < 0) or np.any(np.array([self.minus, self.plus]) > 1):
            raise ValueError("Readout rates must lie in [0,1]")
        self.n = len(self.minus)

    @classmethod
    def symmetric(cls, n, epsilon):
        return cls(np.full(n, epsilon), np.full(n, epsilon))

    def _apply(self, values, transpose=False):
        values = np.asarray(values, float)
        if values.shape != (1 << self.n,):
            raise ValueError("Distribution/channel dimensions differ")
        result = values.reshape((2,) * self.n).copy()
        for i, (em, ep) in enumerate(zip(self.minus, self.plus)):
            a = np.moveaxis(result, i, 0)
            p0, p1 = a[0].copy(), a[1].copy()
            if transpose:
                a[0], a[1] = (1 - em) * p0 + em * p1, ep * p0 + (1 - ep) * p1
            else:
                a[0], a[1] = (1 - em) * p0 + ep * p1, em * p0 + (1 - ep) * p1
        return result.ravel()

    def apply(self, p):
        return normalize(self._apply(normalize(p)))

    def adjoint(self, vector):
        return self._apply(vector, transpose=True)

    def logical_gauge(self, gauge):
        gauge = np.asarray(gauge)
        if gauge.shape != (self.n,) or not np.isin(gauge, [-1, 1]).all():
            raise ValueError("Gauge must contain N signs")
        return ReadoutChannel(np.where(gauge == 1, self.minus, self.plus),
                              np.where(gauge == 1, self.plus, self.minus))


def leading_classical_bias(beta):
    if beta < 0 or not np.isfinite(beta):
        raise ValueError("beta must be finite and nonnegative")
    u = np.exp(-2 * beta)
    return np.array([(1 - u) ** 2 * (2 + u), -u * (1 - u) ** 2])


def calibration_budget(target_r1, fraction=0.1, slope=2.0, epsilon=0.001, z=1.96):
    if target_r1 <= 0 or not 0 < fraction < 1 or slope <= 0 or not 0 <= epsilon <= 1:
        raise ValueError("Invalid calibration budget")
    delta = fraction * target_r1 / slope
    return {"epsilon_half_width": delta, "percentage_points": 100 * delta,
            "iid_trials_per_rate_normal_approx": int(np.ceil(z ** 2 * epsilon * (1 - epsilon) / delta ** 2))}


def calibrate_controls(csv_path, output, *, preparation_status, provenance, context, draws=32,
                       seed=0, method="block_bootstrap", preparation_error_bound=0.0):
    """CSV: block_id,qubit,true_spin,trials,errors. Controls must be separate from targets.

    Block bootstrap resamples ALL qubits jointly. Jeffreys draws are conditional
    binomial approximations; neither supplies unknown preparation fidelity.
    """
    if preparation_status not in {"verified", "bounded", "unresolved", "synthetic"}:
        raise ValueError("Declare preparation_status explicitly")
    if not provenance or draws < 1 or not 0 <= preparation_error_bound <= 1:
        raise ValueError("Provenance, positive draws and a valid preparation bound are required")
    if not context.get("device") or not context.get("graph_id"):
        raise ValueError("Calibration must identify its device and graph ID")
    if preparation_status == "bounded" and preparation_error_bound == 0:
        raise ValueError("Bounded preparation requires a positive preparation_error_bound")
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Empty calibration data")
    blocks = sorted({r["block_id"] for r in rows})
    qubits = sorted({int(r["qubit"]) for r in rows})
    bindex, qindex = {b: i for i, b in enumerate(blocks)}, {q: i for i, q in enumerate(qubits)}
    trials = np.zeros((len(blocks), len(qubits), 2), dtype=np.int64)
    errors = np.zeros_like(trials)
    for row in rows:
        spin, total, wrong = int(row["true_spin"]), int(row["trials"]), int(row["errors"])
        if spin not in [-1, 1] or total < 0 or not 0 <= wrong <= total:
            raise ValueError("Invalid calibration count")
        idx = bindex[row["block_id"]], qindex[int(row["qubit"])], (spin + 1) // 2
        trials[idx] += total
        errors[idx] += wrong
    total, wrong = trials.sum(0), errors.sum(0)
    if np.any(total == 0):
        raise ValueError("Each physical qubit requires both prepared spin signs")
    rng = np.random.default_rng(seed)
    mean = (wrong + 0.5) / (total + 1)
    ensemble = [mean]
    if method not in {"block_bootstrap", "iid_jeffreys"}:
        raise ValueError("Unknown calibration uncertainty method")
    if method == "block_bootstrap" and len(blocks) < 2:
        raise ValueError("Block calibration requires at least two independent acquisition blocks")
    for _ in range(draws - 1):
        if method == "block_bootstrap":
            idx = rng.integers(0, len(blocks), len(blocks))
            nn, kk = trials[idx].sum(0), errors[idx].sum(0)
            ensemble.append((kk + 0.5) / (nn + 1))
        else:
            ensemble.append(rng.beta(wrong + 0.5, total - wrong + 0.5))
    if preparation_error_bound:
        # Explicit sensitivity endpoints, not a posterior over preparation errors.
        ensemble += [np.clip(mean - preparation_error_bound, 0, 1),
                     np.clip(mean + preparation_error_bound, 0, 1)]
    with output_directory(output) as out:
        np.savez_compressed(out / "rates.npz", qubits=qubits, rates=np.array(ensemble),
                            trials=trials, errors=errors)
        write_json(out / "manifest.json", {"schema_version": 1, "kind": "independent_controls",
                   "source_sha256": file_hash(csv_path), "provenance": provenance, "context": context,
                   "preparation_status": preparation_status, "preparation_error_bound": preparation_error_bound,
                   "method": method, "blocks": blocks, "seed": seed,
                   "rates_sha256": file_hash(out / "rates.npz"),
                   "interpretation": "calibration sensitivity ensemble; preparation endpoints are not a posterior"})
    return Calibration(output)


class Calibration:
    def __init__(self, path):
        self.path = Path(path)
        self.manifest = read_json(self.path / "manifest.json")
        if file_hash(self.path / "rates.npz") != self.manifest["rates_sha256"]:
            raise ValueError("Calibration fingerprint mismatch")
        with np.load(self.path / "rates.npz", allow_pickle=False) as f:
            self.qubits, self.rates = f["qubits"], f["rates"]
        if self.rates.ndim != 3 or self.rates.shape[1:] != (len(self.qubits), 2):
            raise ValueError("Invalid calibration shape")
        self.lookup = {int(q): i for i, q in enumerate(self.qubits)}
        self.fingerprint = canonical_hash(self.manifest)

    def channel(self, physical, draw=0, gauge=None):
        try:
            rates = self.rates[draw, [self.lookup[int(q)] for q in physical]]
        except KeyError as exc:
            raise ValueError(f"Uncalibrated physical qubit {exc}") from exc
        channel = ReadoutChannel(rates[:, 0], rates[:, 1])
        return channel if gauge is None else channel.logical_gauge(gauge)
