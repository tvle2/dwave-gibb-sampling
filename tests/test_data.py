import json
from pathlib import Path
import tempfile
import unittest
import zlib
import types
from unittest.mock import patch
import numpy as np
from annni_selection.data import decode_manifest, Dataset, physical_to_logical
from annni_selection.io import write_json, write_sample_rows, iter_sample_rows
from annni_selection.acquisition import acquire_jobs, validation_plan, execute_validation_plan, shim_update
from annni_selection.readout import ReadoutChannel
from annni_selection.states import StateSpace
from annni_selection.catalog import inventory


class DataAndAcquisition(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.physical = np.arange(16).reshape(2, 8)
        self.embedding = self.root / "synthetic_graph_8.txt"
        self.embedding.write_text(repr([{i: int(q) for i, q in enumerate(m)} for m in self.physical]))

    def tearDown(self):
        self.tmp.cleanup()

    def test_stream_reader_formats_and_malformed_archives(self):
        rows = [[1, -1] * 8, [-1, 1] * 8]
        for name, compressed in [("a.json", False), ("a.json.zip", True)]:
            p = self.root / name
            raw = json.dumps(rows).encode()
            p.write_bytes(zlib.compress(raw) if compressed else raw)
            self.assertEqual(list(iter_sample_rows(p)), rows)
        for index, bad in enumerate((b"[[1]]", b"[[1],]", b"[[1]", b"[[1]]extra")):
            p = self.root / f"bad-{index}"
            p.write_bytes(zlib.compress(bad)[:-1] if index == 0 else bad)
            with self.assertRaises(ValueError):
                list(iter_sample_rows(p))

    def test_packed_and_dictionary_rows(self):
        physical = np.array([[10, 30, 50, 70, 90, 110, 130, 150]])
        z = [1, -1] * 4
        np.testing.assert_array_equal(physical_to_logical(z, physical, physical[0]), [z])
        np.testing.assert_array_equal(physical_to_logical(dict(zip(map(str, physical[0]), z)), physical), [z])
        with self.assertRaises(ValueError):
            physical_to_logical(z, physical)
        with self.assertRaises(ValueError):
            physical_to_logical([.9] * 8, physical, physical[0])

    def test_legacy_decode_retains_synchronization_and_inferred_status(self):
        rng = np.random.default_rng(10)
        rows = rng.choice([-1, 1], (24, 16))
        archive = self.root / "old_solutions.json.zip"
        write_sample_rows(archive, rows)
        spec = {"condition": {"n": 8, "graph_id": "graph"}, "embedding_file": self.embedding.name,
                "legacy_file": archive.name, "num_jobs": 4, "reads_per_job": 6}
        manifest = self.root / "input.json"; write_json(manifest, spec)
        data = decode_manifest(manifest, self.root / "decoded")
        self.assertEqual(data.counts.shape, (4, 2, 256))
        np.testing.assert_array_equal(data.counts.sum(2), 6)
        np.testing.assert_array_equal(data.state_ids(), StateSpace().encode(rows.reshape(24, 2, 8)))
        self.assertEqual(data.manifest["provenance"]["boundary_status"], "inferred_fixed_size")

    def test_decode_rejects_extra_rows_and_stale_graph(self):
        archive = self.root / "rows.zlib"; write_sample_rows(archive, [[1] * 16] * 9)
        spec = {"condition": {"n": 8, "graph_id": "graph"}, "embedding_file": self.embedding.name,
                "legacy_file": archive.name, "num_jobs": 4, "reads_per_job": 2}
        manifest = self.root / "input.json"; write_json(manifest, spec)
        with self.assertRaises(ValueError):
            decode_manifest(manifest, self.root / "bad")
        self.assertFalse((self.root / "bad").exists())
        spec["condition"]["graph_id"] = "wrong"; write_json(manifest, spec)
        with self.assertRaises(ValueError):
            decode_manifest(manifest, self.root / "bad")

    def test_gauge_undo_and_inactive_qubits(self):
        row = [3] * 30
        row[:16] = [1] * 16
        gauge = np.array([[-1, 1] * 4] * 2)
        np.testing.assert_array_equal(physical_to_logical(row, self.physical, gauge=gauge), gauge)
        row[0] = 3
        with self.assertRaises(ValueError):
            physical_to_logical(row, self.physical)

    def test_acquisition_resumes_completed_jobs_without_resubmission(self):
        class Future:
            id = "real-like-job-id"
            def result(self):
                return {"solutions": [[1] * 16, [-1] * 16], "timing": {"qpu_access_time": 1234}}
        class Solver:
            id, graph_id, calls = "synthetic", "graph", 0
            def sample_ising(self, h, j, **params):
                self.calls += 1
                if params["answer_mode"] != "raw":
                    raise AssertionError("not raw")
                return Future()
        solver = Solver()
        metadata = {"embedding_file": str(self.embedding), "condition": {"n": 8, "graph_id": "graph"}}
        path = self.root / "acq"
        for _ in range(2):
            result = acquire_jobs({"num_reads": 2, "auto_scale": False}, path, {0: 0}, {(0, 1): -1}, solver, 3, metadata=metadata)
        self.assertEqual(solver.calls, 3)
        self.assertAlmostEqual(result["qpu_access_seconds"], .003702)
        data = decode_manifest(str(path) + "_acquisition.json", self.root / "decoded")
        self.assertEqual(data.counts.shape, (3, 2, 256))
        with self.assertRaises(ValueError):
            acquire_jobs({"num_reads": 3}, path, {0: 0}, {(0, 1): -1}, solver, 3, metadata=metadata)

    def test_dry_run_has_no_qpu_dependency(self):
        plan = validation_plan({"synthetic": "graph"}, times=[.05], blocks=2, seed=1)
        a = execute_validation_plan(plan, self.root, self.root / "never_created")
        self.assertEqual(a["status"], "dry_run")
        self.assertFalse((self.root / "never_created").exists())
        self.assertEqual(plan, validation_plan({"synthetic": "graph"}, times=[.05], blocks=2, seed=1))

    def test_invalid_paid_response_blocks_automatic_resubmission(self):
        class Future:
            id = "charged-but-invalid-answer"
            def result(self):
                return {"solutions": [[1] * 16], "timing": {}}
        class Solver:
            id, graph_id, calls = "synthetic", "graph", 0
            def sample_ising(self, *args, **kwargs):
                self.calls += 1
                return Future()
        solver = Solver(); base = self.root / "bad_response"
        for _ in range(2):
            with self.assertRaises(ValueError):
                acquire_jobs({"num_reads": 2}, base, {0: 0}, {(0, 1): -1}, solver, 1)
        self.assertEqual(solver.calls, 1)
        marker = json.loads((self.root / "bad_response.jobs/job-00000.failure.json").read_text())
        self.assertEqual(marker["qpu_job_id"], Future.id)

    def test_consolidation_keeps_one_array_and_recorded_job_offsets(self):
        class Future:
            id = "mock"
            def result(self):
                return {"solutions": [[1] * 16, [-1] * 16], "timing": {}}
        class Solver:
            id, graph_id, calls = "synthetic", "graph", 0
            def sample_ising(self, *args, **kwargs):
                self.calls += 1
                return Future()
        solver = Solver(); base = self.root / "one_setting"
        condition = {"n": 8, "device": "synthetic", "graph_id": "graph", "kappa": .5,
                     "energy_scale": 1.0, "anneal_time_us": .05}
        kwargs = dict(metadata={"embedding_file": str(self.embedding), "condition": condition}, consolidate=True)
        result = acquire_jobs({"num_reads": 2}, base, {0: 0}, {(0, 1): -1}, solver, 3, **kwargs)
        self.assertEqual(result["jobs"][-1]["row_stop"], 6)
        self.assertEqual(len(list(self.root.rglob("*.json.zlib"))), 1)
        data = decode_manifest(str(base) + "_acquisition.json", self.root / "consolidated")
        self.assertEqual(data.manifest["provenance"]["boundary_status"], "recorded")
        acquire_jobs({"num_reads": 2}, base, {0: 0}, {(0, 1): -1}, solver, 3, **kwargs)
        self.assertEqual(solver.calls, 3)
        catalog = inventory(self.root, self.root, self.root / "catalog")
        self.assertEqual(len(catalog["entries"]), 1)
        entry = catalog["entries"][0]
        self.assertEqual(entry["format"], "recorded")
        indexed = decode_manifest(self.root / "catalog" / entry["manifest"], self.root / "indexed")
        np.testing.assert_array_equal(indexed.counts, data.counts)

    def test_entire_plan_preflight_precedes_paid_submissions(self):
        class Solver:
            id, graph_id, calls = "synthetic", "graph", 0
            undirected_edges = [(i, j) for i in range(16) for j in range(i + 1, 16)]
            properties = {"num_qubits": 16, "j_range": [-2, 1], "fast_anneal_time_range": [.005, 2000]}
            parameters = {k: None for k in ["num_reads", "auto_scale", "annealing_time", "fast_anneal"]}
            def sample_ising(self, *args, **kwargs):
                self.calls += 1
                raise AssertionError("No job should be submitted")
        solver = Solver()
        class Client:
            @classmethod
            def from_config(cls): return cls()
            def get_solver(self, name): return solver
            def close(self): pass
        cloud = types.ModuleType("dwave.cloud"); cloud.Client = Client
        plan = validation_plan({"synthetic": "graph"}, times=[.05, 3000], scales=[1],
                                blocks=1, gauges=1, reads=2, paired_shims=False)
        with patch.dict('sys.modules', {'dwave': types.ModuleType('dwave'), 'dwave.cloud': cloud}):
            with self.assertRaises(ValueError):
                execute_validation_plan(plan, self.root, self.root / "no_jobs", execute=True)
        self.assertEqual(solver.calls, 0)

    def test_shimming_keeps_NN_and_NNN_means(self):
        s = StateSpace()
        rng = np.random.default_rng(3)
        p = rng.dirichlet(np.ones(256))
        before = np.zeros((2, 8))
        update = shim_update(s, p, ReadoutChannel.symmetric(8, .01), flux_offsets=np.zeros(8),
                             coupler_corrections=before, flux_response_sign=1,
                             flux_gain=.01, coupler_gain=.01, provenance="independent controls")
        np.testing.assert_allclose(np.array(update["coupler_corrections"]).mean(1), 0, atol=1e-18)


if __name__ == "__main__":
    unittest.main()
