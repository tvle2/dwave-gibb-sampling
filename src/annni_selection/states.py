"""Exact configuration dictionary; logical spin 0 is the most significant bit.

Bit 0 means spin -1, bit 1 means spin +1. This convention is independent of
Qiskit's basis convention. All fields, gauges and readout matrices use it.
"""
from functools import cached_property
import numpy as np


class StateSpace:
    def __init__(self, n=8):
        if not isinstance(n, int) or n < 6 or n > 12 or n % 2:
            raise ValueError("Use an even ring size between 6 and 12.")
        self.n = n
        self.size = 1 << n
        self.bit_weights = 1 << np.arange(n - 1, -1, -1)
        self.spins = (2 * ((np.arange(self.size)[:, None] & self.bit_weights) != 0) - 1).astype(np.int8)
        self.correlations = np.column_stack([
            (self.spins * np.roll(self.spins, -r, axis=1)).sum(axis=1)
            for r in range(1, n // 2 + 1)
        ])
        walls = (1 - self.spins * np.roll(self.spins, -1, axis=1)) // 2
        self.m = walls.sum(axis=1)
        self.q = (walls * np.roll(walls, -1, axis=1)).sum(axis=1)
        self.ground = self.q == 0
        self.target = (self.m == 2) & self.ground
        self.sector_masks = {"F": self.ground & (self.m == 0),
                             "T": self.target, "A": self.ground & (self.m == n // 2)}
        if n == 8:
            self.orbits = {"F": self.sector_masks["F"],
                           "62": self.target & (abs(self.spins.sum(axis=1)) == 4),
                           "53": self.target & (abs(self.spins.sum(axis=1)) == 2),
                           "44": self.target & (self.spins.sum(axis=1) == 0),
                           "A": self.sector_masks["A"]}
        else:
            self.orbits = {}
        self.class_values, self.class_id = np.unique(self.correlations[:, :2], axis=0, return_inverse=True)

    def encode(self, spins):
        spins = np.asarray(spins)
        if spins.shape[-1] != self.n or not np.isin(spins, [-1, 1]).all():
            raise ValueError("Every active logical spin must be -1 or +1.")
        return ((spins == 1) @ self.bit_weights).astype(np.uint16 if self.n > 8 else np.uint8)

    def energy(self, kappa=0.5):
        if not np.isfinite(kappa):
            raise ValueError("kappa must be finite")
        return -self.correlations[:, 0] + kappa * self.correlations[:, 1]

    @cached_property
    def flip_adjacency(self):
        result = np.zeros((self.size, self.size))
        ids = np.arange(self.size)
        for bit in self.bit_weights:
            result[ids, ids ^ bit] = 1
        return result

    @cached_property
    def symmetry_orbits(self):
        canonical = []
        for spins in self.spins:
            images = [np.roll(sign * direction, k)
                      for direction in (spins, spins[::-1]) for sign in (-1, 1) for k in range(self.n)]
            canonical.append(int(self.encode(np.asarray(images)).min()))
        return np.unique(canonical, return_inverse=True)[1]

    @cached_property
    def pair_features(self):
        pairs = [(i, j) for i in range(self.n) for j in range(i + 1, self.n)]
        return np.column_stack([self.spins] + [self.spins[:, i] * self.spins[:, j] for i, j in pairs]).astype(float)

    def require_eight(self):
        if self.n != 8:
            raise ValueError("The five-orbit contrasts and W are defined here only for N=8.")

    @cached_property
    def w_coefficients(self):
        self.require_eight()
        c = np.zeros(self.size)
        for name, weight in {"F": 1, "A": 1, "62": -4, "44": 2}.items():
            mask = self.orbits[name]
            c[mask] = weight / mask.sum()
        return c

    def dictionary(self):
        return [{"id": i, "spins": z.tolist(), "C": self.correlations[i].tolist(),
                 "m": int(self.m[i]), "q": int(self.q[i]),
                 "class_id": int(self.class_id[i]), "symmetry_orbit": int(self.symmetry_orbits[i]),
                 "ground_orbit": next((name for name, mask in self.orbits.items() if mask[i]), None)}
                for i, z in enumerate(self.spins)]
