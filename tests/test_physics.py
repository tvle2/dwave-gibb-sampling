import unittest
import numpy as np
from scipy.linalg import expm
from scipy.sparse.csgraph import connected_components
from annni_selection.states import StateSpace
from annni_selection.models import (classical_gibbs, quantum_gibbs, projected_gibbs,
    conditional, emulating_pairwise, susceptibility_gibbs, mixture, from_log_weights, distribution)
from annni_selection.observables import summarize, contrast_covariance, w_variance_coefficient, kl_decomposition
from annni_selection.readout import ReadoutChannel, leading_classical_bias, calibration_budget


class ExactPhysics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = StateSpace()

    def test_enumeration_and_classes(self):
        s = self.s
        self.assertEqual(s.ground.sum(), 46)
        self.assertEqual(s.target.sum(), 40)
        self.assertEqual({k: int(v.sum()) for k, v in s.orbits.items()}, {"F": 2, "62": 16, "53": 16, "44": 8, "A": 4})
        e, count = np.unique(s.energy(), return_counts=True)
        self.assertEqual(dict(zip(e, count)), {-4: 46, -2: 64, 0: 72, 2: 16, 4: 40, 6: 16, 12: 2})
        np.testing.assert_array_equal(s.energy(), -4 + 2 * s.q)
        self.assertEqual(len(np.unique(s.symmetry_orbits)), 18)
        self.assertEqual(len(s.class_values), 10)
        np.testing.assert_array_equal(s.encode(s.spins), np.arange(256))

    def test_graph_walk_counts(self):
        s = self.s
        g = s.flip_adjacency[np.ix_(s.ground, s.ground)]
        _, labels = connected_components(g, directed=False)
        self.assertEqual(sorted(np.bincount(labels)), [1] * 6 + [40])
        a = s.flip_adjacency[np.ix_(s.target, s.target)]
        for orbit, expected in [("62", (2, 12)), ("53", (4, 30)), ("44", (4, 36))]:
            mask = s.orbits[orbit][s.target]
            np.testing.assert_array_equal(np.diag(a @ a)[mask], expected[0])
            np.testing.assert_array_equal(np.diag(np.linalg.matrix_power(a, 4))[mask], expected[1])

    def test_global_thermodynamic_mixtures_uniform_inside_T(self):
        s = self.s
        rng = np.random.default_rng(19)
        ps = [classical_gibbs(s, b, k) for b, k in rng.uniform([0, 0], [8, 1.3], (30, 2))]
        p = mixture(ps, rng.random(30))
        np.testing.assert_allclose(conditional(p, s.target), 1 / 40, atol=1e-15)

    def test_projected_curve_and_full_emulation(self):
        s = self.s
        for x in (.05, .3, .6, 1, 1.5, 3):
            p = projected_gibbs(s, x)
            stat = summarize(s, p)
            self.assertGreater(stat["R1"], 0)
            self.assertGreater(stat["R2"], 0)
            paired = emulating_pairwise(s, stat["R1"], stat["R2"])
            np.testing.assert_allclose(conditional(p, s.target), conditional(paired, s.target), atol=1e-15)
            self.assertAlmostEqual(summarize(s, paired)["W"], 0, places=12)
        p = projected_gibbs(s, 20)
        masses = [p[s.orbits[o]].sum() / p[s.target].sum() for o in ("62", "53", "44")]
        np.testing.assert_allclose(masses, [1 / 6, 1 / 2, 1 / 3], atol=2e-7)

    def test_small_x_coefficients(self):
        stat = summarize(self.s, projected_gibbs(self.s, .05))
        self.assertAlmostEqual(stat["R1"] / .05**2, .998, delta=.001)
        self.assertAlmostEqual(stat["R2"] / .05**4, .249, delta=.001)
        self.assertAlmostEqual(stat["W"], -6.243e-6, delta=2e-9)

    def test_all_pairwise_null_and_asymmetric_curvature_identity(self):
        s = self.s
        rng = np.random.default_rng(23)
        np.testing.assert_allclose(s.w_coefficients @ s.pair_features, 0, atol=1e-15)
        for _ in range(40):
            p = from_log_weights(s.pair_features @ rng.normal(size=36))
            self.assertAlmostEqual(summarize(s, p)["W"], 0, places=12)
            p = rng.dirichlet(np.ones(256))
            a = summarize(s, p)
            self.assertAlmostEqual(a["W"], a["I_L"] - a["I_L_pair"], places=12)

    def test_sector_reweighting_changes_W_not_shape(self):
        s = self.s
        for x, w, il, locked in [(.6, -.110737711289, -1.066108368565, -.955370657276),
                                 (1, -.661923350232, -2.67804421103, -2.016120860799)]:
            p = projected_gibbs(s, x)
            a = summarize(s, p)
            np.testing.assert_allclose([a["W"], a["I_L"], a["I_L_pair"]], [w, il, locked], atol=1e-11)
            q = p.copy()
            q[s.orbits["F"] | s.orbits["A"]] *= np.exp(-a["W"] / 2)
            self.assertAlmostEqual(summarize(s, q)["W"], 0, places=11)
            np.testing.assert_allclose(conditional(p, s.target), conditional(q, s.target))

    def test_full_Hamiltonian_sign_reversals(self):
        s = self.s
        for beta, gamma, r1, r2 in [(6, .05, .066960445934, -.000594251582),
                                    (1.5, .2, .034271601409, -.003366866985),
                                    (.75, .4, .01701779, -.00294671),
                                    (6, .1, .221721342697, .008966457062),
                                    (2, .3, .134890881608, -.003085422149),
                                    (1.5, .4, .107514994679, -.005644048661),
                                    (2.5, .4, .28145501057, .019645044765)]:
            a = summarize(s, quantum_gibbs(s, beta, gamma))
            np.testing.assert_allclose([a["R1"], a["R2"]], [r1, r2], atol=5e-9)
        p = quantum_gibbs(s, 10, .1)
        masses = [p[s.orbits[o]].sum() / p[s.target].sum() for o in ("62", "53", "44")]
        np.testing.assert_allclose(masses, [.28749, .46550, .24702], atol=5e-6)

    def test_diagonalization_against_matrix_exponential(self):
        s = self.s
        h = np.diag(s.energy(.7)) - .2 * s.flip_adjacency
        direct = np.diag(expm(-1.2 * h)).copy(); direct /= direct.sum()
        np.testing.assert_allclose(quantum_gibbs(s, 1.2, .2, .7), direct, atol=2e-15)

    def test_single_susceptibility_and_physical_scale(self):
        s = self.s
        a = summarize(s, susceptibility_gibbs(s, 1, -.02))
        np.testing.assert_allclose([a["R1"], a["R2"]], [.06, -.02], atol=1e-14)
        for chi in (-.04, -.01, .02):
            a = summarize(s, susceptibility_gibbs(s, 2, chi))
            self.assertAlmostEqual(a["R2"] + a["R1"] / 3, 0, places=12)
        p1 = distribution(s, {"family": "susceptibility_transfer", "beta_per_scale": 1, "chi": -.02}, {"energy_scale": 1})
        p2 = distribution(s, {"family": "susceptibility_transfer", "beta_per_scale": 1, "chi": -.02}, {"energy_scale": .1})
        self.assertAlmostEqual(summarize(s, p2)["R1"] / summarize(s, p1)["R1"], .01, places=11)

    def test_uniform_field_ordering(self):
        for eta in (-1, -.1, 0, .1, 1):
            p = distribution(self.s, {"family": "uniform_field", "beta": 2, "eta": eta})
            stats = summarize(self.s, p)
            self.assertLessEqual(stats["R1"], 1e-15)
            self.assertLessEqual(stats["R2"], 1e-15)

    def test_variance_and_KL(self):
        s = self.s
        p = np.zeros(s.size); p[s.target] = 1 / 40
        np.testing.assert_allclose(np.diag(contrast_covariance(s, p, 1)), [5, 7.5])
        p = s.ground.astype(float) / 46
        self.assertAlmostEqual(w_variance_coefficient(s, p), 103.5)
        for x, expected in [(.3, 108.5221133), (.6, 127.7059737), (1, 201.8123311), (1.5, 547.3577076)]:
            self.assertAlmostEqual(w_variance_coefficient(s, projected_gibbs(s, x)), expected, places=6)
        q = classical_gibbs(s, 1.5)
        a = kl_decomposition(s, quantum_gibbs(s, 1.5, .2), q)
        self.assertAlmostEqual(a["total"], a["class_occupancy"] + a["within_class"])


class ReadoutPhysics(unittest.TestCase):
    def setUp(self):
        self.s = StateSpace()

    def test_symmetric_channel_against_hamming_matrix(self):
        s = self.s; p = quantum_gibbs(s, 2, .3)
        distance = (s.spins[:, None, :] != s.spins[None, :, :]).sum(2)
        epsilon = .02
        expected = (epsilon**distance * (1-epsilon)**(8-distance)) @ p
        np.testing.assert_allclose(ReadoutChannel.symmetric(8, epsilon).apply(p), expected, atol=1e-15)

    def test_asymmetric_channel_and_adjoint(self):
        rng = np.random.default_rng(55)
        channel = ReadoutChannel(rng.uniform(0, .04, 8), rng.uniform(0, .06, 8))
        p = rng.dirichlet(np.ones(256)); v = rng.normal(size=256)
        self.assertAlmostEqual(channel.apply(p) @ v, p @ channel.adjoint(v), places=13)
        fixed = np.zeros(256); fixed[0] = 1
        mean = channel.apply(fixed) @ self.s.spins
        np.testing.assert_allclose(mean, -1 + 2 * channel.minus)

    def test_gauge_readout_sign_swap(self):
        c = ReadoutChannel(np.full(8, .01), np.full(8, .03))
        g = np.array([-1, 1] * 4)
        transformed = c.logical_gauge(g)
        np.testing.assert_allclose(transformed.minus, np.where(g < 0, .03, .01))

    def test_exact_readout_derivatives(self):
        s = self.s
        for beta in (.5, 1, 1.5, 2, 6):
            p = classical_gibbs(s, beta)
            dp = s.flip_adjacency @ p - 8 * p
            ratios = [dp[s.orbits[o]].mean() / p[s.orbits[o]].mean() for o in ("62", "53", "44")]
            np.testing.assert_allclose(np.diff(ratios), leading_classical_bias(beta), atol=1e-14)
        stat = summarize(s, ReadoutChannel.symmetric(8, .01).apply(classical_gibbs(s, 1.5)))
        np.testing.assert_allclose([stat["R1"], stat["R2"]], [.017941803733, -.000172951693], atol=1e-12)

    def test_readout_invalidates_raw_pairwise_and_field_sign_tests(self):
        p = classical_gibbs(self.s, 1.5)
        self.assertAlmostEqual(summarize(self.s, p)["W"], 0, places=12)
        a = summarize(self.s, ReadoutChannel.symmetric(8, .01).apply(p))
        self.assertGreater(a["R1"], 0)
        self.assertGreater(abs(a["W"]), 1e-6)

    def test_budget_and_invalid_inputs(self):
        self.assertAlmostEqual(calibration_budget(.018, slope=1.8)["epsilon_half_width"], .001)
        for bad in (-.1, 1.1, np.nan):
            with self.assertRaises(ValueError):
                ReadoutChannel.symmetric(8, bad)


if __name__ == "__main__":
    unittest.main()
