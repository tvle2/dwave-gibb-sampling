import unittest
import tempfile
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from annni_selection.states import StateSpace
from annni_selection.models import from_log_weights, distribution
from annni_selection.dynamics import (Schedule, classical_generator, classical_evolve,
    quantum_evolve, snapshot_parameters, KB_OVER_H_GHZ_PER_MK, GHZ_US_ANGULAR)


class Dynamics(unittest.TestCase):
    def setUp(self):
        self.s = StateSpace()
        self.schedule = Schedule([0, 1], [.1, .1], [.2, .2], "synthetic", "unit test")

    def test_units_and_snapshot(self):
        self.assertAlmostEqual(KB_OVER_H_GHZ_PER_MK, .020836619123, places=11)
        a = snapshot_parameters(.2, 2, 20, 1)
        b = snapshot_parameters(.2, 2, 20, .1)
        self.assertAlmostEqual(a["x"], b["x"])
        self.assertAlmostEqual(b["beta"] / a["beta"], .1)
        self.assertAlmostEqual(b["gamma"] / a["gamma"], 10)

    def test_classical_generator_detailed_balance_single_and_collective(self):
        energy = .2 * self.s.energy()
        p = from_log_weights(-energy / (20 * KB_OVER_H_GHZ_PER_MK))
        for rates in ({1: 1}, {1: 1, 2: .3, 3: .1, 4: .05}):
            generator = classical_generator(self.s, energy, 20, rates)
            np.testing.assert_allclose(np.asarray(generator.sum(0)), 0, atol=1e-13)
            np.testing.assert_allclose(generator @ p, 0, atol=1e-15)
            flow = generator.toarray() * p
            np.testing.assert_allclose(flow, flow.T, atol=1e-15)

    def test_classical_propagation_constant_schedule(self):
        generator = classical_generator(self.s, .1 * self.s.energy(), 20, {1: 1, 2: .2})
        initial = np.ones(256) / 256
        expected = expm(generator.toarray() * .1) @ initial
        actual = classical_evolve(self.s, self.schedule, .1, 20, amplitudes={1: 1, 2: .2},
                                   exponents={1: 0, 2: 0}, steps=3)
        np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_unitary_propagation_against_direct_exponential(self):
        h = np.diag(.1 * self.s.energy()) - .05 * self.s.flip_adjacency
        initial = np.ones(256) / 16
        expected = abs(expm(-1j * GHZ_US_ANGULAR * .001 * h) @ initial) ** 2
        actual = quantum_evolve(self.s, self.schedule, .001, steps=3)
        np.testing.assert_allclose(actual, expected, atol=2e-14)

    def test_thermal_reset_map_preserves_positive_quantum_state(self):
        rho = quantum_evolve(self.s, self.schedule, .001, steps=3, reset_rate=100,
                             dephasing_rate=10, return_density=True)
        np.testing.assert_allclose(rho, rho.conj().T, atol=1e-14)
        self.assertAlmostEqual(np.trace(rho).real, 1)
        self.assertGreater(np.linalg.eigvalsh(rho).min(), -1e-12)

    def test_reset_equilibrium_is_stationary_without_extra_dephasing(self):
        h = np.diag(.1 * self.s.energy()) - .05 * self.s.flip_adjacency
        rho = expm(-h / (20 * KB_OVER_H_GHZ_PER_MK)); rho /= np.trace(rho)
        result = quantum_evolve(self.s, self.schedule, .003, steps=2, reset_rate=200,
                                initial=rho, return_density=True)
        np.testing.assert_allclose(result, rho, atol=1e-13)

    def test_inference_adapter_uses_condition_time_and_rejects_synthetic_hardware_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schedule.csv"
            path.write_text("fraction,A_GHz,B_GHz\n0,0.1,0.2\n1,0.1,0.2\n")
            candidate = {"family": "dynamics", "kind": "classical",
                         "schedule": {"path": str(path), "protocol": "synthetic", "provenance": "unit test"},
                         "temperature_mk": 20, "amplitudes": {"1": 1, "2": .2},
                         "exponents": {"1": 0, "2": 0}, "steps": 3, "max_refinement_tv": 1e-10}
            condition = {"device": "synthetic", "kappa": .5, "energy_scale": .1, "anneal_time_us": .1}
            actual = distribution(self.s, candidate, condition)
            expected = classical_evolve(self.s, self.schedule, .1, 20, energy_scale=.1,
                                        amplitudes={1: 1, 2: .2}, exponents={1: 0, 2: 0}, steps=6)
            np.testing.assert_allclose(actual, expected, atol=1e-14)
            with self.assertRaisesRegex(ValueError, "Synthetic schedules"):
                distribution(self.s, candidate, {**condition, "device": "actual_qpu"})


if __name__ == "__main__":
    unittest.main()
