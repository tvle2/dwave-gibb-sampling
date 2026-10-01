"""Shared-parameter prediction across entire held-out conditions and fixed families."""
from pathlib import Path
import numpy as np
from .data import Dataset
from .inference import channels_for, _observed, _scope, evaluate_frozen
from .io import canonical_hash, file_hash, write_json, read_json, output_directory
from .models import distribution
from .statistics import split_jobs, score_jobs, score_mean, holm


def freeze_transfer(study, output):
    """Select ONE physical candidate per family/draw on development conditions only.

    The scope here is a declared finite grid, not unconstrained per-recipient fits.
    Flexible fitted pairwise/sector stress models belong in per-condition analysis.
    """
    entries = study["datasets"]
    ids = [e["id"] for e in entries]
    if len(ids) != len(set(ids)) or any(Path(i).name != i or i in {".", ".."} for i in ids):
        raise ValueError("Study dataset IDs must be distinct safe directory names")
    if not any(e["role"] == "development" for e in entries) or not any(e["role"] == "holdout" for e in entries):
        raise ValueError("Transfer needs development and whole-condition holdout datasets")
    if any(e["role"] not in {"development", "holdout"} for e in entries):
        raise ValueError("Unknown dataset role")
    datasets = [Dataset(e["path"]) for e in entries]
    if len({d.fingerprint for d in datasets}) != len(datasets):
        raise ValueError("A dataset cannot appear in both development and holdout roles")
    channels_and_meta = [channels_for(d, e["readout"]) for d, e in zip(datasets, entries)]
    draw_counts = [len(c[0]) for c in channels_and_meta]
    draws = max(draw_counts)
    if any(n not in {1, draws} for n in draw_counts):
        raise ValueError("Calibration ensembles must have equal draw counts, or a single fixed channel")
    splits = [split_jobs(len(d.counts), study.get("block_size", 1), study.get("seed", 0))
              if e["role"] == "development" else
              {"train": [], "test": list(range(len(d.counts))),
               "block_ids": (np.arange(len(d.counts)) // study.get("block_size", 1)).tolist(),
               "block_size": study.get("block_size", 1), "seed": study.get("seed", 0)}
              for d, e in zip(datasets, entries)]
    names = list(study["families"])
    selected, profiles = [], []
    for draw in range(draws):
        chosen, profile = {}, {}
        for name, family in study["families"].items():
            best, best_value, candidates_profile = None, -np.inf, []
            for candidate in family["candidates"]:
                if candidate["family"] in {"pairwise_fit", "mixture_fit", "sector_fit"}:
                    raise ValueError("Transfer accepts fixed physical candidates; recipient-specific fitting is forbidden")
                scores = []
                for data, entry, split, (channels, _) in zip(datasets, entries, splits, channels_and_meta):
                    if entry["role"] != "development":
                        continue
                    p = distribution(data.space, candidate, data.condition)
                    latent = np.repeat(p[None, :], len(data.physical), axis=0)
                    observed = _observed(latent, channels[min(draw, len(channels) - 1)])
                    scores.append(score_mean(score_jobs(data.counts[split["train"]], observed,
                                                        _scope(data.space, family.get("fit_scope", "full")))))
                value = float(np.mean(scores))
                candidates_profile.append({"candidate": candidate, "development_score": value if np.isfinite(value) else None})
                if value > best_value:
                    best, best_value = candidate, value
            if best is None:
                raise ValueError(f"No supported transfer candidate for {name}")
            chosen[name], profile[name] = best, candidates_profile
        selected.append(chosen)
        profiles.append(profile)
    with output_directory(output) as out:
        for data, entry, split, (channels, metadata) in zip(datasets, entries, splits, channels_and_meta):
            directory = out / entry["id"]
            directory.mkdir()
            latent, observed = [], []
            for draw in range(draws):
                ps = [np.repeat(distribution(data.space, selected[draw][name], data.condition)[None, :],
                                len(data.physical), axis=0) for name in names]
                latent.append(ps)
                observed.append([_observed(p, channels[min(draw, len(channels) - 1)]) for p in ps])
            latent, observed = np.array(latent), np.array(observed)
            if split["train"]:
                eligibility_counts = data.counts[split["train"]].sum(0)
            else:
                # Only planned read totals and frozen predictions, not held-out outcomes.
                read_totals = data.counts[split["test"]].sum((0, 2))
                eligibility_counts = observed.min(axis=(0, 1)) * read_totals[:, None]
            target_eligible = eligibility_counts[:, data.space.target].sum(1) >= study.get("min_target_train", 100)
            w_eligible = (eligibility_counts[:, data.space.ground] >= study.get("min_state_train", 100)).all(1)
            np.savez_compressed(directory / "predictions.npz", observed=observed, latent=latent,
                                target_eligible=target_eligible, w_eligible=w_eligible)
            config = {"conditional_null": study.get("conditional_null"), "study_fingerprint": canonical_hash(study)}
            write_json(directory / "frozen.json", {"schema_version": 1, "dataset_fingerprint": data.fingerprint,
                "condition": data.condition, "job_ids": data.manifest["job_ids"], "split": split,
                "config": config, "model_names": names, "readout": metadata,
                "predictions_sha256": file_hash(directory / "predictions.npz"),
                "selected_by_draw": selected, "role": entry["role"],
                "interpretation": "One candidate per family selected on development conditions; holdout conditions never fitted"})
        write_json(out / "study.json", {"specification": study, "selected_by_draw": selected,
                   "profiles": profiles, "dataset_fingerprints": dict(zip(ids, [d.fingerprint for d in datasets]))})
    return read_json(Path(output) / "study.json")


def evaluate_study(frozen_directory, output, repeats=500, seed=0):
    directory = Path(frozen_directory)
    study = read_json(directory / "study.json")["specification"]
    reports = {}
    with output_directory(output) as out:
        for entry in study["datasets"]:
            reports[entry["id"]] = evaluate_frozen(entry["path"], directory / entry["id"],
                                                  out / (entry["id"] + ".json"), repeats, seed)
        primary = study.get("primary_ids", [e["id"] for e in study["datasets"]])
        if len(primary) != len(set(primary)):
            raise ValueError("Primary family has duplicate condition IDs")
        pvalues = []
        for name in primary:
            omnibus = reports.get(name, {}).get("conditional_omnibus")
            p = omnibus.get("worst_grid_pvalue") if omnibus else None
            pvalues.append(1.0 if p is None else p)
        result = {"primary_ids": primary, "pvalues_with_missing_as_one": pvalues,
                  "holm": holm(pvalues), "status": "exploratory; Holm does not repair an unvalidated nuisance/dependence model"}
        write_json(out / "family.json", result)
    return result
