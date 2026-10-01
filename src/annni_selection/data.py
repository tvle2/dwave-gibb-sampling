"""Physical samples -> synchronized logical state IDs and job x embedding counts."""
import ast
from pathlib import Path
import numpy as np
from .io import canonical_hash, file_hash, iter_sample_rows, read_json, write_json, output_directory
from .states import StateSpace


def load_embeddings(path, n=8):
    mappings = ast.literal_eval(Path(path).read_text())
    if not isinstance(mappings, list) or not mappings:
        raise ValueError("Embedding file must contain a nonempty list of mappings")
    if any(set(m) != set(range(n)) for m in mappings):
        raise ValueError("Each mapping must contain exactly the logical indices 0..N-1")
    physical = np.array([[m[i] for i in range(n)] for m in mappings])
    if not np.issubdtype(physical.dtype, np.integer) or np.any(physical < 0):
        raise ValueError("Physical qubit labels must be nonnegative integers")
    if len(np.unique(physical)) != physical.size:
        raise ValueError("Mappings must be injective and vertex-disjoint")
    return physical


def physical_to_logical(row, physical, variable_order=None, gauge=None):
    if isinstance(row, dict):
        row = {int(k): v for k, v in row.items()}
        try:
            z = np.array([[row[int(q)] for q in m] for m in physical])
        except KeyError as exc:
            raise ValueError(f"Sample missing active qubit {exc}") from exc
    else:
        row = np.asarray(row)
        if row.ndim != 1:
            raise ValueError("Physical sample row must be one-dimensional")
        if variable_order is None:
            columns = physical
        else:
            order = list(map(int, variable_order))
            if len(set(order)) != len(order) or len(order) != len(row):
                raise ValueError("Invalid packed variable order")
            indices = {q: i for i, q in enumerate(order)}
            try:
                columns = np.array([[indices[int(q)] for q in m] for m in physical])
            except KeyError as exc:
                raise ValueError("Packed order omits active qubit") from exc
        if columns.max() >= len(row):
            raise ValueError("Dense sample too short; supply variable_order for packed samples")
        z = row[columns]
    if not np.isin(z, [-1, 1]).all():
        raise ValueError("Active embedded qubits must all be -1 or +1 (not inactive 3)")
    if gauge is not None:
        gauge = np.asarray(gauge)
        if gauge.shape != physical.shape or not np.isin(gauge, [-1, 1]).all():
            raise ValueError("Gauge must have shape embedding x N and values +/-1")
        z = z * gauge
    return z.astype(np.int8)


class Dataset:
    def __init__(self, path, verify=True):
        self.path = Path(path).resolve()
        self.manifest = read_json(self.path / "manifest.json")
        if verify and file_hash(self.path / "counts.npz") != self.manifest["counts_sha256"]:
            raise ValueError("Counts fingerprint does not match manifest")
        with np.load(self.path / "counts.npz", allow_pickle=False) as a:
            self.counts = a["counts"]
            self.physical = a["physical"]
            self.offsets = a["offsets"]
        self.space = StateSpace(int(self.manifest["condition"].get("n", 8)))
        if self.counts.ndim != 3 or self.counts.shape[1:] != (len(self.physical), self.space.size):
            raise ValueError("Expected job x embedding x state counts")
        if not np.issubdtype(self.counts.dtype, np.integer) or np.any(self.counts < 0):
            raise ValueError("Counts must be nonnegative integers")
        if len(self.offsets) != len(self.counts) + 1 or self.offsets[0] != 0 or np.any(np.diff(self.offsets) <= 0):
            raise ValueError("Invalid job offsets")
        if not np.all(self.counts.sum(axis=2) == np.diff(self.offsets)[:, None]):
            raise ValueError("Synchronized embeddings must have equal reads per job")
        self.fingerprint = canonical_hash(self.manifest)

    @property
    def condition(self):
        return self.manifest["condition"]

    def state_ids(self):
        if file_hash(self.path / "state_ids.npy") != self.manifest.get("state_ids_sha256"):
            raise ValueError("Synchronized state IDs do not match the manifest")
        return np.load(self.path / "state_ids.npy", mmap_mode="r", allow_pickle=False)


def save_dataset(path, counts, physical, condition, job_ids, state_ids=None, provenance=None):
    counts = np.asarray(counts)
    if counts.ndim != 3 or not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
        raise ValueError("Expected nonnegative integer job x embedding x state counts")
    nread = counts.sum(axis=2)
    if not np.all(nread == nread[:, :1]) or np.any(nread <= 0):
        raise ValueError("Each job must retain all synchronized embeddings")
    if len(job_ids) != len(counts) or len(set(job_ids)) != len(job_ids):
        raise ValueError("Job IDs must be unique and match the counts")
    offsets = np.r_[0, np.cumsum(nread[:, 0])]
    with output_directory(path) as out:
        np.savez_compressed(out / "counts.npz", counts=counts, physical=physical, offsets=offsets)
        if state_ids is not None:
            if np.shape(state_ids) != (int(offsets[-1]), counts.shape[1]):
                raise ValueError("Incorrect synchronized state-ID shape")
            np.save(out / "state_ids.npy", state_ids)
        manifest = {"schema_version": 1, "condition": condition, "job_ids": list(job_ids),
                    "counts_sha256": file_hash(out / "counts.npz"),
                    "provenance": provenance or {}, "bit_convention": "logical0_msb; -1=0,+1=1"}
        if state_ids is not None:
            manifest["state_ids_sha256"] = file_hash(out / "state_ids.npy")
        write_json(out / "manifest.json", manifest)
    return Dataset(path)


def decode_manifest(manifest_path, output):
    """Explicit jobs are authoritative; legacy fixed-size boundaries are marked inferred.

    A manifest contains condition, embedding_file, and either jobs[{file,reads,id}]
    or legacy_file, reads_per_job, num_jobs. Paths are relative to the manifest.
    """
    manifest_path = Path(manifest_path).resolve()
    spec = read_json(manifest_path)
    condition = spec["condition"]
    space = StateSpace(int(condition.get("n", 8)))
    embedding_path = (manifest_path.parent / spec["embedding_file"]).resolve()
    physical = load_embeddings(embedding_path, space.n)
    try:
        embedding_graph = embedding_path.stem.rsplit("_", 2)[1]
    except IndexError:
        embedding_graph = None
    if condition.get("graph_id") and embedding_graph != condition["graph_id"]:
        raise ValueError("Embedding filename graph ID does not match acquisition condition")
    if "legacy_file" in spec:
        reads, jobs = int(spec["reads_per_job"]), int(spec["num_jobs"])
        job_specs = [{"file": spec["legacy_file"], "reads": reads, "id": f"inferred-{j}"} for j in range(jobs)]
        boundary_status = "inferred_fixed_size"
    else:
        job_specs = spec["jobs"]
        boundary_status = "recorded"
    if not job_specs or any(int(j["reads"]) <= 0 for j in job_specs):
        raise ValueError("Positive per-job read counts are required")
    ids = [str(j["id"]) for j in job_specs]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate job IDs")
    offsets = np.r_[0, np.cumsum([int(j["reads"]) for j in job_specs])]
    for j, job in enumerate(job_specs):
        if "row_start" in job and (job["row_start"] != offsets[j] or job["row_stop"] != offsets[j + 1]):
            raise ValueError("Recorded job offsets are inconsistent")
    common_file = spec.get("legacy_file", spec.get("sample_file"))
    counts = np.zeros((len(job_specs), len(physical), space.size), dtype=np.int64)
    with output_directory(output) as out:
        states = np.lib.format.open_memmap(out / "state_ids.npy", mode="w+",
                                          dtype=np.uint8 if space.n == 8 else np.uint16,
                                          shape=(int(offsets[-1]), len(physical)))
        source_hashes = {}
        stream = None
        for j, job in enumerate(job_specs):
            source = (manifest_path.parent / (common_file or job["file"])).resolve()
            if common_file is None or stream is None:
                if str(source) not in source_hashes:
                    source_hashes[str(source)] = file_hash(source)
                expected_hash = spec.get("sample_sha256") if common_file else job.get("sha256")
                if expected_hash and source_hashes[str(source)] != expected_hash:
                    raise ValueError("Source archive fingerprint mismatch")
                stream = iter_sample_rows(source)
            for row_index in range(int(offsets[j]), int(offsets[j + 1])):
                try:
                    row = next(stream)
                except StopIteration as exc:
                    raise ValueError(f"Missing rows in job {ids[j]}") from exc
                z = physical_to_logical(row, physical, job.get("variable_order", spec.get("variable_order")),
                                        job.get("gauge", spec.get("gauge")))
                encoded = space.encode(z)
                states[row_index] = encoded
                counts[j, np.arange(len(physical)), encoded] += 1
            if common_file is None or j == len(job_specs) - 1:
                try:
                    next(stream)
                except StopIteration:
                    pass
                else:
                    raise ValueError("More samples than the declared job boundaries")
        states.flush()
        del states
        np.savez_compressed(out / "counts.npz", counts=counts, physical=physical, offsets=offsets)
        write_json(out / "manifest.json", {
            "schema_version": 1, "condition": condition, "job_ids": ids,
            "counts_sha256": file_hash(out / "counts.npz"),
            "state_ids_sha256": file_hash(out / "state_ids.npy"),
            "bit_convention": "logical0_msb; -1=0,+1=1",
            "provenance": {"boundary_status": boundary_status, "input_manifest_sha256": file_hash(manifest_path),
                           "sources": source_hashes, "embedding_sha256": file_hash(embedding_path),
                           "gauge_by_job": [j.get("gauge", spec.get("gauge")) for j in job_specs]}})
    return Dataset(output)
