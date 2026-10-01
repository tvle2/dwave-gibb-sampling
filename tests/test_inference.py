import csv
from pathlib import Path
import tempfile
import unittest
import numpy as np
from annni_selection.data import Dataset, save_dataset
from annni_selection.inference import fit_dataset, evaluate_frozen, channels_for, fit_pairwise
from annni_selection.io import read_json, write_json
from annni_selection.models import classical_gibbs, from_log_weights
from annni_selection.readout import ReadoutChannel, calibrate_controls
from annni_selection.simulation import synthetic_dataset, sample_synchronized
from annni_selection.states import StateSpace
from annni_selection.statistics import split_jobs, block_resample, score_jobs, holm, wild_cluster_omnibus
from annni_selection.study import freeze_transfer, evaluate_study


class Inference(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.space = StateSpace()
        self.config = {"seed": 4, "block_size": 1, "cv_folds": 2, "min_target_train": 10,
                       "min_state_train": 5, "readout": {"kind": "sensitivity", "epsilon_values": [.01], "provenance": "test scenario"},
                       "conditional_null": "classical", "families": {
                           "classical": {"candidates": [{"family": "classical", "beta": b} for b in (1, 1.5, 2)]},
                           "quantum": {"candidates": [{"family": "quantum", "beta": 1.5, "gamma": g} for g in (0, .2)]}}}

    def tearDown(self):
        self.tmp.cleanup()

    def dataset(self, name="data", scale=1):
        return synthetic_dataset(self.root / name, {"family": "classical", "beta": 1.5},
                                  jobs=12, reads_per_job=200, embeddings=2, epsilon=.01, seed=5,
                                  shared_fraction=.3, condition={"n": 8, "kappa": .5, "energy_scale": scale,
                                  "device": "synthetic", "graph_id": "synthetic", "anneal_time_us": .05})

    def test_no_test_leakage_into_fitting_or_eligibility(self):
        a = self.dataset()
        frozen_a = fit_dataset(a.path, self.config, self.root / "fit-a")
        bcounts = a.counts.copy()
        # Replace every B sample with one state, retaining sizes; A is untouched.
        for index in frozen_a["split"]["test"]:
            bcounts[index] = 0; bcounts[index, :, 0] = 200
        b = save_dataset(self.root / "changed-B", bcounts, a.physical, a.condition,
                         a.manifest["job_ids"], provenance={"boundary_status": "synthetic"})
        frozen_b = fit_dataset(b.path, self.config, self.root / "fit-b")
        self.assertEqual(frozen_a["fits_by_readout_draw"], frozen_b["fits_by_readout_draw"])
        self.assertEqual(frozen_a["training_counts_sha256"], frozen_b["training_counts_sha256"])
        with self.assertRaises(ValueError):
            evaluate_frozen(b.path, self.root / "fit-a", self.root / "wrong.json", 20)

    def test_frozen_evaluation_and_tamper_guard(self):
        data = self.dataset()
        fit_dataset(data.path, self.config, self.root / "fit")
        result = evaluate_frozen(data.path, self.root / "fit", self.root / "report.json", 29)
        self.assertEqual(len(result["embedding_observables"]), 2)
        self.assertIn("quantum minus classical", result["paired_comparisons"])
        p = self.root / "fit" / "predictions.npz"
        p.write_bytes(p.read_bytes() + b"modified")
        with self.assertRaises(ValueError):
            evaluate_frozen(data.path, self.root / "fit", self.root / "tampered.json", 20)

    def test_pairwise_optimization_recovers_nontrivial_distribution(self):
        rng = np.random.default_rng(91)
        truth = from_log_weights(self.space.pair_features @ rng.normal(0, .2, 36))
        channel = ReadoutChannel(np.full(8, .01), np.full(8, .02))
        frequency = channel.apply(truth)
        # Large deterministic expected counts isolate optimization from sampling error.
        counts = np.repeat((frequency * 1e7)[None, None, :], 4, axis=0)
        p, parameters = fit_pairwise(self.space, counts, [channel], l2=1e-8)
        np.testing.assert_allclose(p[0], truth, atol=1e-7)

    def test_calibration_is_separate_and_device_bound(self):
        source = self.root / "controls.csv"
        with source.open("w", newline="") as f:
            w = csv.writer(f); w.writerow(["block_id", "qubit", "true_spin", "trials", "errors"])
            for block in range(3):
                for qubit in range(16):
                    for spin in (-1, 1):
                        w.writerow([block, qubit, spin, 1000, 10 if spin == -1 else 20])
        cal = calibrate_controls(source, self.root / "cal", preparation_status="verified", provenance="test controls",
                                 context={"device": "synthetic", "graph_id": "synthetic"}, draws=3)
        self.assertEqual(cal.rates.shape, (3, 16, 2))
        self.assertAlmostEqual(cal.rates[0, 0, 0], 30.5 / 3001)
        data = self.dataset()
        channels, _ = channels_for(data, {"kind": "calibration", "path": str(cal.path)})
        self.assertEqual(len(channels), 3)
        data.condition["graph_id"] = "wrong"
        with self.assertRaises(ValueError):
            channels_for(data, {"kind": "calibration", "path": str(cal.path)})

    def test_whole_condition_transfer_no_recipient_fitting(self):
        a = self.dataset("source", 1)
        b = self.dataset("recipient", .1)
        study = {"seed": 1, "block_size": 1, "min_target_train": 10, "min_state_train": 5,
                 "conditional_null": "classical", "primary_ids": ["a", "b", "missing"],
                 "datasets": [{"id": "a", "path": str(a.path), "role": "development", "readout": self.config["readout"]},
                              {"id": "b", "path": str(b.path), "role": "holdout", "readout": self.config["readout"]}],
                 "families": {"classical": {"candidates": [{"family": "classical", "beta": v} for v in (1, 1.5, 2)]}}}
        result = freeze_transfer(study, self.root / "study")
        frozen = read_json(self.root / "study" / "b" / "frozen.json")
        self.assertEqual(frozen["split"]["train"], [])
        self.assertEqual(frozen["split"]["test"], list(range(12)))
        report = evaluate_study(self.root / "study", self.root / "reports", repeats=19)
        self.assertEqual(report["pvalues_with_missing_as_one"][-1], 1.0)


class Statistics(unittest.TestCase):
    def test_block_split_and_resampling_keep_whole_groups(self):
        split = split_jobs(24, block_size=3, seed=2)
        for block in range(8):
            members = set(range(block * 3, block * 3 + 3))
            self.assertTrue(members <= set(split["train"]) or members <= set(split["test"]))
        sample = block_resample(split["test"], split["block_ids"], np.random.default_rng(9))
        for i in range(0, len(sample), 3):
            self.assertEqual(len(set(np.array(split["block_ids"])[sample[i:i + 3]])), 1)

    def test_equal_job_embedding_score_and_missing_occupancy(self):
        counts = np.array([[[90, 10], [1, 9]], [[1, 1], [30, 70]]])
        p = np.array([[.8, .2], [.3, .7]])
        expected = np.array([(.9 * np.log(.8) + .1 * np.log(.2) + .1 * np.log(.3) + .9 * np.log(.7)) / 2,
                             (.5 * np.log(.8) + .5 * np.log(.2) + .3 * np.log(.3) + .7 * np.log(.7)) / 2])
        np.testing.assert_allclose(score_jobs(counts, p), expected)
        counts[0, 0] = 0
        self.assertAlmostEqual(score_jobs(counts, p)[0], .1 * np.log(.3) + .9 * np.log(.7))

    def test_zero_probability_is_not_silently_floored(self):
        score = score_jobs(np.array([[[1, 0]], [[0, 1]]]), np.array([[1., 0.]]))
        self.assertEqual(score[0], 0)
        self.assertEqual(score[1], -np.inf)

    def test_holm(self):
        np.testing.assert_allclose(holm([.01, .04, .03])["adjusted"], [.03, .06, .06])

    def test_shared_draws_keep_marginals_and_copy_correlation(self):
        p = np.array([[.2, .8]] * 3)
        states = sample_synchronized(p, 10000, np.random.default_rng(3), shared_fraction=1)
        np.testing.assert_array_equal(states[:, 0], states[:, 1])
        self.assertAlmostEqual((states[:, 0] == 0).mean(), .2, delta=.015)

    def test_exact_null_residual_is_not_rejected(self):
        counts = np.tile(np.array([[[20, 80], [60, 40]]]), (10, 1, 1))
        p = np.array([[.2, .8], [.6, .4]])
        result = wild_cluster_omnibus(counts, p, np.arange(10), np.arange(10),
                                     np.array([True, True]), np.array([True, True]), repeats=99)
        self.assertEqual(result["pvalue"], 1)


if __name__ == "__main__":
    unittest.main()
