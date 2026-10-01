"""Inventory existing data without loading samples; emit explicit decoder manifests."""
from collections import defaultdict
from pathlib import Path
import re
from .acquisition import PRIMARY_TIMES_US
from .io import read_json, write_json, canonical_hash, output_directory

LEGACY = re.compile(r"(?P<device>.+)_ANNNI_N(?P<n>\d+)_J2(?P<kappa>[\d.eE+-]+)_es(?P<scale>[\d.eE+-]+)_at(?P<time>[\d.eE+-]+)_solutions\.json\.(?:zip|zlib)$")


def condition_id(c):
    return f"{c['device']}_k{c['kappa']:g}_s{c['energy_scale']:g}_t{c['anneal_time_us']:g}"


def primary_condition(c):
    return (c["device"] in {"Advantage_system6", "Advantage2_system1"} and c["n"] == 8
            and c["kappa"] == .5 and c["energy_scale"] in (1.0, .1) and c["anneal_time_us"] in PRIMARY_TIMES_US)


def inventory(root, embeddings_directory, output):
    root, embeddings_directory = Path(root).resolve(), Path(embeddings_directory).resolve()
    items, acquired = [], defaultdict(list)
    recorded_arrays = set()
    for manifest_path in root.rglob("*_acquisition.json"):
        spec = read_json(manifest_path)
        if spec.get("complete") and spec.get("sample_file"):
            recorded_arrays.add((manifest_path.parent / spec["sample_file"]).resolve())
    for f in sorted(root.rglob("*_solutions.json.*")):
        if f.resolve() in recorded_arrays:
            continue
        match = LEGACY.fullmatch(f.name)
        if not match:
            continue
        m = match.groupdict()
        c = {"device": m["device"], "graph_id": f.parent.name, "n": int(m["n"]),
             "kappa": float(m["kappa"]), "energy_scale": float(m["scale"]),
             "anneal_time_us": float(m["time"]), "fast_anneal": True}
        embedding = embeddings_directory / f"{c['device']}_{c['graph_id']}_{c['n']}.txt"
        spec = {"condition": c, "embedding_file": str(embedding), "legacy_file": str(f),
                "reads_per_job": 2000 if c["anneal_time_us"] <= 10 else 250,
                "num_jobs": 40 if c["anneal_time_us"] <= 10 else 160,
                "boundary_status": "inferred from archived source-code settings; verify provenance before inference"}
        items.append((condition_id(c) + "_" + c["graph_id"], spec, "legacy"))
    for f in sorted(root.rglob("*_acquisition.json")):
        spec = read_json(f)
        if not spec.get("complete") or not spec.get("embedding_file") or not spec.get("condition"):
            continue
        c = spec["condition"]
        stratum = {k: c.get(k) for k in ("device", "graph_id", "n", "kappa", "energy_scale", "anneal_time_us", "gauge_index", "shim")}
        key = canonical_hash(stratum)
        spec["embedding_file"] = str((f.parent / spec["embedding_file"]).resolve())
        if spec.get("sample_file"):
            spec["sample_file"] = str((f.parent / spec["sample_file"]).resolve())
            # Repeated consolidated runs of one condition remain distinct; do
            # not overwrite a manifest with another run's source file.
            source_id = canonical_hash(str(f.resolve()))[:10]
            items.append((condition_id(c) + "_" + key[:10] + "_" + source_id, spec, "recorded"))
            continue
        for job in spec["jobs"]:
            job = dict(job)
            job["file"] = str((f.parent / job["file"]).resolve())
            job["id"] = str(f.resolve()) + ":" + job["id"]
            job["acquisition_block"] = c.get("block")
            acquired[key].append((c, spec["embedding_file"], job))
    for key, group in acquired.items():
        group.sort(key=lambda x: (x[2].get("submitted_at", ""), x[2]["id"]))
        c = dict(group[0][0])
        c.pop("block", None)
        spec = {"condition": c, "embedding_file": group[0][1], "jobs": [x[2] for x in group]}
        items.append((condition_id(c) + "_" + key[:10], spec, "recorded"))
    expected = [condition_id({"device": d, "kappa": .5, "energy_scale": s, "anneal_time_us": t})
                for d in ("Advantage_system6", "Advantage2_system1") for s in (1.0, .1) for t in PRIMARY_TIMES_US]
    found = {condition_id(spec["condition"]) for _, spec, _ in items}
    records = []
    with output_directory(output) as out:
        for identifier, spec, kind in items:
            write_json(out / (identifier + ".json"), spec)
            records.append({"id": identifier, "condition_id": condition_id(spec["condition"]),
                            "manifest": identifier + ".json", "format": kind,
                            "embedding_exists": Path(spec["embedding_file"]).is_file(),
                            "primary": primary_condition(spec["condition"])})
        result = {"entries": records, "primary_condition_ids": expected,
                  "missing_primary_conditions": [c for c in expected if c not in found],
                  "interpretation": "Metadata inventory only; decoder verifies row counts, active spins and archive hashes"}
        write_json(out / "catalog.json", result)
    return result
