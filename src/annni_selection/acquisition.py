"""Resumable per-job acquisition and offline prospective design. No import-time QPU calls."""
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import numpy as np
from .io import canonical_hash, file_hash, read_json, write_json, write_sample_rows
from .data import load_embeddings

PRIMARY_TIMES_US = [.005, .01, .02, .05, .1, .2, .5, 1, 2, 5, 10, 20, 50, 100, 500, 2000]


def now():
    return datetime.now(timezone.utc).isoformat()


def logical_couplings(n=8, kappa=.5, energy_scale=1.0):
    result = {}
    for i in range(n):
        for distance, value in ((1, -energy_scale), (2, kappa * energy_scale)):
            edge = tuple(sorted((i, (i + distance) % n)))
            result[edge] = result.get(edge, 0.0) + value
    return result


def map_couplings(physical, kappa=.5, energy_scale=1.0, gauges=None):
    physical = np.asarray(physical)
    if physical.ndim != 2 or len(np.unique(physical)) != physical.size:
        raise ValueError("Mappings must be vertex-disjoint")
    gauges = np.ones_like(physical) if gauges is None else np.asarray(gauges)
    if gauges.shape != physical.shape or not np.isin(gauges, [-1, 1]).all():
        raise ValueError("Gauge array must match the embedding array")
    j = {}
    for mapping, gauge in zip(physical, gauges):
        for (a, b), value in logical_couplings(physical.shape[1], kappa, energy_scale).items():
            j[tuple(sorted((int(mapping[a]), int(mapping[b]))))] = value * int(gauge[a]) * int(gauge[b])
    return {int(q): 0.0 for q in physical.ravel()}, j


def acquire_jobs(params, path, h, j, solver, num_jobs, *, metadata=None, compression_level=1, consolidate=False):
    """One durable raw archive per job. A submission failure is never retried blindly."""
    params = dict(params)
    if params.get("answer_mode", "raw") != "raw":
        raise ValueError("Analysis requires answer_mode='raw'")
    params["answer_mode"] = "raw"
    if params.get("auto_scale", False):
        raise ValueError("auto_scale must be false for calibrated parameter comparisons")
    params["auto_scale"] = False
    if params.get("fast_anneal") and (any(v != 0 for v in h.values()) or "initial_state" in params):
        raise ValueError("Fast anneals cannot use programmed fields or standard reverse initialization")
    if num_jobs < 1 or int(params.get("num_reads", 0)) < 1:
        raise ValueError("Positive jobs and num_reads are required")
    base = Path(path).resolve()
    legacy = Path(str(base) + "_solutions.json.zip")
    if legacy.exists():
        return {"status": "skipped_existing_legacy", "path": str(legacy)}
    folder = Path(str(base) + ".jobs")
    folder.mkdir(parents=True, exist_ok=True)
    specification = {"params": params, "h": [[int(k), float(v)] for k, v in sorted(h.items())],
                     "J": [[int(a), int(b), float(v)] for (a, b), v in sorted(j.items())],
                     "num_jobs": int(num_jobs), "solver_id": str(getattr(solver, "id", "unknown")),
                     "graph_id": str(getattr(solver, "graph_id", "unknown")), "metadata": metadata or {}}
    settings_path = folder / "settings.json"
    if settings_path.exists():
        if canonical_hash(read_json(settings_path)) != canonical_hash(specification):
            raise ValueError("Existing acquisition has different settings; choose a new path")
    else:
        write_json(settings_path, specification)
    final_manifest_path = Path(str(base) + "_acquisition.json")
    if final_manifest_path.exists():
        completed = read_json(final_manifest_path)
        if completed.get("complete") and completed.get("sample_file"):
            sample = base.parent / completed["sample_file"]
            if not sample.exists() or file_hash(sample) != completed["sample_sha256"]:
                raise ValueError("Consolidated archive is missing or changed")
            return completed
    records = []
    for index in range(num_jobs):
        filename = folder / f"job-{index:05d}.json.zlib"
        record_path = folder / f"job-{index:05d}.metadata.json"
        if record_path.exists():
            record = read_json(record_path)
            if record.get("file") != str(filename.relative_to(base.parent)) or record.get("reads") != int(params["num_reads"]):
                raise ValueError("Recorded job path or size differs from the acquisition settings")
            if not filename.exists() or file_hash(filename) != record["sha256"]:
                raise ValueError("Completed job archive is missing or changed")
            records.append(record)
            continue
        if filename.exists():
            raise ValueError(f"Orphan job archive needs inspection before resume: {filename}")
        failure_path = folder / f"job-{index:05d}.failure.json"
        if failure_path.exists():
            raise ValueError(f"Inspect the previous submission before retrying; resolve and remove {failure_path}")
        submitted = now()
        # Persist intent before contacting the solver. A crash, bad response or
        # failed disk write must not silently submit the same paid job again.
        pending = {"index": index, "submitted_at": submitted, "qpu_job_id": None,
                   "state": "submission_in_progress_or_interrupted",
                   "action": "Inspect submission status before retry; an exception does not prove the QPU job was not charged."}
        write_json(failure_path, pending)
        future = None
        try:
            future = solver.sample_ising(h, j, **params)
            result = future.result()
            rows = result.get("solutions")
            if rows is None:
                rows = future.samples
            if len(rows) != int(params["num_reads"]):
                raise ValueError("Returned raw row count differs from num_reads")
            occurrences = result.get("num_occurrences")
            if occurrences is not None and not np.all(np.asarray(occurrences) == 1):
                raise ValueError("Histogram answers cannot preserve read order")
            write_sample_rows(filename, rows, compression_level)
            record = {"id": f"job-{index:05d}", "qpu_job_id": getattr(future, "id", None),
                      "index": index, "reads": len(rows), "file": str(filename.relative_to(base.parent)),
                      "sha256": file_hash(filename), "submitted_at": submitted, "completed_at": now(),
                      "timing": result.get("timing", {}), "params": params,
                      "gauge": (metadata or {}).get("gauge")}
            write_json(record_path, record)
        except Exception as exc:
            write_json(failure_path, {**pending, "state": "failed", "failed_at": now(),
                       "reason_type": type(exc).__name__, "qpu_job_id": getattr(future, "id", None)})
            raise
        failure_path.unlink()
        records.append(record)
        del rows, future, result
    total_seconds = sum(r["timing"].get("qpu_access_time", 0) for r in records) / 1e6
    Path(str(base) + "_QPU_time.txt").write_text(str(total_seconds) + "\n")
    manifest = {"schema_version": 1, "condition": (metadata or {}).get("condition", {}),
                "embedding_file": (metadata or {}).get("embedding_file"), "jobs": records,
                "settings_sha256": file_hash(settings_path), "complete": True,
                "qpu_access_seconds": total_seconds}
    if consolidate:
        from .io import iter_sample_rows
        final_samples = Path(str(base) + "_solutions.json.zlib")
        def all_rows():
            for record in records:
                yield from iter_sample_rows(base.parent / record["file"])
        write_sample_rows(final_samples, all_rows(), compression_level)
        manifest["sample_file"] = final_samples.name
        manifest["sample_sha256"] = file_hash(final_samples)
        offset = 0
        manifest["jobs"] = []
        for record in records:
            job = dict(record)
            job["row_start"], job["row_stop"] = offset, offset + job["reads"]
            offset = job["row_stop"]
            job["checkpoint_sha256"] = job.pop("sha256")
            job.pop("file")
            manifest["jobs"].append(job)
    write_json(final_manifest_path, manifest)
    if consolidate:
        # Only our verified checkpoints, after the complete array and manifest are durable.
        for record in records:
            (base.parent / record["file"]).unlink()
    return manifest


def validation_plan(devices, *, times=None, scales=(1.0, .1), kappas=(.5,),
                    blocks=8, gauges=2, reads=2000, seed=0, paired_shims=True):
    """One job per setting per randomized block. Calibration jobs remain separate."""
    if blocks < 1 or gauges < 1 or reads < 1:
        raise ValueError("Positive blocks, gauges and reads are required")
    rng = np.random.default_rng(seed)
    times = PRIMARY_TIMES_US if times is None else list(times)
    gauge_signs = [np.ones(8, dtype=int).tolist()] + [rng.choice([-1, 1], 8).tolist() for _ in range(gauges - 1)]
    jobs = []
    variants = ["unshimmed", "shimmed"] if paired_shims else ["unshimmed"]
    for device, graph_id in devices.items():
        for block in range(blocks):
            batch = []
            for kappa, scale, time, gauge_index, shim in product(kappas, scales, times, range(gauges), variants):
                if scale <= 0 or time <= 0:
                    raise ValueError("Scales and anneal times must be positive")
                batch.append({"role": "validation", "device": device, "graph_id": graph_id, "n": 8,
                              "kappa": kappa, "energy_scale": scale, "anneal_time_us": time,
                              "block": block, "gauge_index": gauge_index, "logical_gauge": gauge_signs[gauge_index],
                              "shim": shim, "reads": reads, "fast_anneal": True})
            jobs.extend(batch[i] for i in rng.permutation(len(batch)))
    return {"schema_version": 1, "seed": seed, "jobs": jobs,
            "execution_status": "offline plan; no QPU connection or submission",
            "prerequisites": ["Independent preparation/readback calibration with uncertainty and transfer checks",
                              "Frozen shims from separate calibration-only jobs if shimmed variants are requested",
                              "Current graph, coupler, time, flux-bias and parameter compatibility checks"],
            "planned_physical_reads": sum(j["reads"] for j in jobs)}


def shim_update(space, probabilities, channel, *, flux_offsets, coupler_corrections,
                flux_response_sign, flux_gain, coupler_gain, provenance):
    """One calibration-only update. Targets magnetization and within-class correlations.

    flux_response_sign is the independently measured sign of d<Z>/d(flux_bias).
    No orbit population, R1, R2 or W appears in this objective.
    """
    if not provenance or flux_response_sign not in [-1, 1] or min(flux_gain, coupler_gain) < 0:
        raise ValueError("Calibration provenance, measured flux response sign and nonnegative gains are required")
    from .models import normalize
    p = normalize(probabilities)
    a, b = 1 - channel.minus - channel.plus, channel.minus - channel.plus
    if np.any(abs(a) < .1):
        raise ValueError("Readout inversion is too ill-conditioned for this update")
    corrected = (space.spins - b) / a
    magnetization = p @ corrected
    flux = np.asarray(flux_offsets, float) - flux_gain * flux_response_sign * magnetization
    corrections = np.asarray(coupler_corrections, float).copy()
    if corrections.shape != (2, space.n) or flux.shape != (space.n,):
        raise ValueError("Expected N flux offsets and 2 by N NN/NNN corrections")
    correlations = []
    for r in (1, 2):
        corr = p @ (corrected * np.roll(corrected, -r, axis=1))
        residual = corr - corr.mean()
        corrections[r - 1] += coupler_gain * residual
        # Floating-point drift only; preserve the starting class mean.
        corrections[r - 1] -= corrections[r - 1].mean() - np.asarray(coupler_corrections)[r - 1].mean()
        correlations.append(corr)
    return {"flux_offsets": flux.tolist(), "coupler_corrections": corrections.tolist(),
            "magnetizations": magnetization.tolist(), "correlations": np.array(correlations).tolist(),
            "provenance": provenance, "status": "proposed calibration update; verify hardware limits before application"}


def execute_validation_plan(plan, embeddings_directory, output, *, execute=False, shim_files=None):
    """Dry-run by default; preflight every request before the first paid submission."""
    if not execute:
        return {"status": "dry_run", "jobs": len(plan["jobs"]), "physical_reads": plan["planned_physical_reads"]}
    from dwave.cloud import Client
    clients = {}
    mappings = {}

    def prepare(item):
        device = item["device"]
        if device not in clients:
            client = Client.from_config()
            try:
                solver = client.get_solver(device)
            except Exception:
                client.close()
                raise
            clients[device] = (client, solver)
        solver = clients[device][1]
        if str(solver.graph_id) != item["graph_id"]:
            raise ValueError("Plan graph ID is stale; regenerate/validate embeddings")
        embedding = Path(embeddings_directory).resolve() / f"{device}_{solver.graph_id}_{item['n']}.txt"
        if embedding not in mappings:
            mappings[embedding] = load_embeddings(embedding, item["n"])
        physical = mappings[embedding]
        gauge = np.repeat(np.array(item["logical_gauge"])[None, :], len(physical), axis=0)
        h, j = map_couplings(physical, item["kappa"], item["energy_scale"], gauge)
        original = j.copy()
        available = {tuple(sorted(edge)) for edge in solver.undirected_edges}
        if any(edge not in available for edge in j):
            raise ValueError("Embedding uses a missing hardware coupler")
        params = {"num_reads": item["reads"], "auto_scale": False,
                  "annealing_time": item["anneal_time_us"], "fast_anneal": True}
        shim_hash = None
        if item["shim"] == "shimmed":
            key = f"{device}:g{item['gauge_index']}:k{item['kappa']}:s{item['energy_scale']}"
            if not shim_files or key not in shim_files:
                raise ValueError(f"Missing independently frozen shim file: {key}")
            shim = read_json(shim_files[key]); shim_hash = file_hash(shim_files[key])
            if shim.get("graph_id") != item["graph_id"] or shim.get("device") != device or shim.get("role") != "frozen_shim":
                raise ValueError("Shim identity/provenance mismatch")
            if not shim.get("calibration_provenance"):
                raise ValueError("Missing independent shim calibration provenance")
            for a, b, delta in shim.get("coupler_deltas", []):
                edge = tuple(sorted((int(a), int(b))))
                if edge not in j or not np.isfinite(delta):
                    raise ValueError("Shim contains an invalid coupling")
                j[edge] += float(delta)
            for mapping, signs in zip(physical, gauge):
                for r in (1, 2):
                    changes = []
                    for a in range(item["n"]):
                        b = (a + r) % item["n"]
                        edge = tuple(sorted((int(mapping[a]), int(mapping[b]))))
                        changes.append((j[edge] - original[edge]) * signs[a] * signs[b])
                    if abs(np.mean(changes)) > 1e-10:
                        raise ValueError("Shim changes a programmed NN/NNN class mean")
            flux = np.asarray(shim["flux_biases"], float)
            bounds = shim.get("verified_flux_bias_bounds")
            if flux.shape != (solver.properties["num_qubits"],) or not np.isfinite(flux).all():
                raise ValueError("Flux offsets must be finite and cover the solver's qubit indices")
            if not bounds or len(bounds) != 2 or bounds[0] > bounds[1] or np.any((flux < bounds[0]) | (flux > bounds[1])):
                raise ValueError("Supply verified flux-bias bounds and keep all offsets within them")
            params["flux_biases"] = flux.tolist()
        elif item["shim"] != "unshimmed":
            raise ValueError("Unknown shim variant")
        limits = solver.properties.get("j_range")
        if not limits or any(not np.isfinite(value) or not limits[0] <= value <= limits[1] for value in j.values()):
            raise ValueError("Programmed couplings exceed known solver range")
        trange = solver.properties.get("fast_anneal_time_range")
        if not trange or not trange[0] <= item["anneal_time_us"] <= trange[1]:
            raise ValueError("Anneal time outside fast-protocol range")
        if any(p not in solver.parameters for p in params):
            raise ValueError("Plan requests an unsupported solver parameter")
        return solver, h, j, params, embedding, gauge, shim_hash

    try:
        for item in plan["jobs"]:
            prepare(item)  # Discard per-request coupler maps; do not accumulate GBs of them.
        results = []
        for index, item in enumerate(plan["jobs"]):
            solver, h, j, params, embedding, gauge, shim_hash = prepare(item)
            path = Path(output) / f"request-{index:06d}"
            acquire_jobs(params, path, h, j, solver, 1,
                         metadata={"condition": item, "embedding_file": str(embedding), "gauge": gauge.tolist(),
                                   "plan_fingerprint": canonical_hash(plan), "shim_sha256": shim_hash})
            results.append(str(path) + "_acquisition.json")
        return {"status": "completed", "acquisition_manifests": results}
    finally:
        for client, _ in clients.values():
            client.close()
