"""Development-only fitting and immutable held-out predictions.

Readout is supplied from independent controls or an EXPLICIT sensitivity grid.
It is never optimized against within-T selection. Curvature is reported per
embedding; the resulting observations need not obey a latent pairwise identity.
"""
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from .data import Dataset
from .io import canonical_hash, file_hash, read_json, write_json, output_directory
from .models import distribution, from_log_weights, projected_gibbs
from .readout import Calibration, ReadoutChannel
from .observables import summarize
from .statistics import (split_jobs, grouped_folds, score_jobs, score_mean,
                         paired_score_interval, observable_intervals, wild_cluster_omnibus)


def homogeneous_gauge(dataset):
    gauges = dataset.manifest.get("provenance", {}).get("gauge_by_job", [None])
    keys = {canonical_hash(g) for g in gauges}
    if len(keys) > 1:
        raise ValueError("Analyze each gauge as a separate stratum; mixed gauges cannot use one static channel or W null")
    return gauges[0] if gauges else None


def channels_for(dataset, readout_spec):
    gauge = homogeneous_gauge(dataset)
    if readout_spec["kind"] == "calibration":
        calibration = Calibration(readout_spec["path"])
        context = calibration.manifest.get("context", {})
        if any(context.get(key) != dataset.condition.get(key) for key in ("device", "graph_id")):
            raise ValueError("Readout calibration device/graph does not match the dataset")
        draws = min(int(readout_spec.get("draws", len(calibration.rates))), len(calibration.rates))
        if draws < 1:
            raise ValueError("At least one calibration draw is required")
        channels = [[calibration.channel(q, d, None if gauge is None else gauge[e])
                     for e, q in enumerate(dataset.physical)] for d in range(draws)]
        metadata = {"kind": "independent_controls", "fingerprint": calibration.fingerprint,
                    "manifest": calibration.manifest, "draws": draws}
    elif readout_spec["kind"] == "sensitivity":
        eps = readout_spec["epsilon_values"]
        if not eps or not readout_spec.get("provenance"):
            raise ValueError("Sensitivity grid needs rates and explicit provenance")
        channels = [[ReadoutChannel.symmetric(dataset.space.n, x) for _ in dataset.physical] for x in eps]
        metadata = {**readout_spec, "interpretation": "assumed symmetric readout scenarios, not independent calibration"}
    else:
        raise ValueError("Choose independent calibration or explicit readout sensitivity; no implicit perfect-readout assumption")
    return channels, metadata


def _scope(space, name):
    if name == "full":
        return None
    if name == "T":
        return space.target
    if name == "G":
        return space.ground
    if name == "62_53":
        return space.orbits["62"] | space.orbits["53"]
    raise ValueError(f"Unknown fitting scope {name}")


def _observed(latent, channels):
    return np.array([channel.apply(p) for channel, p in zip(channels, latent)])


def fit_pairwise(space, counts, channels, l2=0.001):
    """Regularized all-fields/all-pairs stress test, fitted separately per embedding."""
    if l2 <= 0:
        raise ValueError("Positive regularization is required for finite sparse-data fits")
    features = space.pair_features
    means = (counts / counts.sum(2, keepdims=True)).mean(0)
    latent, parameters, diagnostics = [], [], []
    for frequency, channel in zip(means, channels):
        def objective(theta):
            p = from_log_weights(features @ theta)
            q = channel.apply(p)
            positive = frequency > 0
            if np.any(q[positive] <= 0):
                return 1e100, np.zeros_like(theta)
            loss = -float(frequency[positive] @ np.log(q[positive])) + l2 * float(theta @ theta) / 2
            ratio = np.divide(frequency, q, out=np.zeros_like(q), where=q > 0)
            back = channel.adjoint(ratio)
            gradient = -features.T @ (p * (back - p @ back)) + l2 * theta
            return loss, gradient
        fit = minimize(objective, np.zeros(features.shape[1]), jac=True, method="L-BFGS-B",
                       options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-7})
        if not fit.success:
            raise RuntimeError(f"Pairwise fit did not converge: {fit.message}")
        parameters.append(fit.x.tolist())
        latent.append(from_log_weights(features @ fit.x))
        diagnostics.append({"iterations": int(fit.nit), "objective": float(fit.fun)})
    return np.array(latent), {"family": "pairwise_per_embedding", "l2": l2,
                               "theta": parameters, "optimizer": diagnostics}


def _fit_mixture(space, specs, counts, channels, condition, mask=None):
    components = np.array([distribution(space, spec, condition) for spec in specs])
    observed = np.array([_observed(np.repeat(p[None, :], len(channels), axis=0), channels) for p in components])
    def loss(weights):
        p = np.einsum("k,kes->es", weights, observed)
        score = score_mean(score_jobs(counts, p, mask))
        return -score if np.isfinite(score) else 1e100
    initial = np.full(len(specs), 1 / len(specs))
    result = minimize(loss, initial, method="SLSQP", bounds=[(0, 1)] * len(specs),
                      constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                      options={"maxiter": 500, "ftol": 1e-12})
    if not result.success:
        raise RuntimeError(f"Mixture optimization failed: {result.message}")
    weights = np.maximum(result.x, 0)
    weights /= weights.sum()
    p = weights @ components
    return np.repeat(p[None, :], len(channels), axis=0), {"family": "mixture", "components": specs, "weights": weights.tolist()}


def _fit_candidate(dataset, candidate, counts, channels, scope):
    space, condition = dataset.space, dataset.condition
    family = candidate["family"]
    if family == "pairwise_fit":
        if scope != "full":
            raise ValueError("Pairwise stress fits use full distributions, not only target contrasts")
        return fit_pairwise(space, counts, channels, candidate.get("l2", .001))
    if family == "mixture_fit":
        return _fit_mixture(space, candidate["components"], counts, channels, condition, _scope(space, scope))
    if family == "sector_fit":
        if scope not in {"full", "G"}:
            raise ValueError("Free sector weights cannot be estimated only from T")
        specs = [{"family": "projected", "x": candidate["x"], "sector_weights": weights}
                 for weights in ([1, 0, 0], [0, 1, 0], [0, 0, 1])]
        return _fit_mixture(space, specs, counts, channels, condition, _scope(space, scope))
    p = distribution(space, candidate, condition)
    return np.repeat(p[None, :], len(channels), axis=0), dict(candidate)


def fit_dataset(dataset_path, config, output):
    dataset = Dataset(dataset_path)
    space, counts = dataset.space, dataset.counts
    space.require_eight()
    config = dict(config)
    if not config.get("families") or not config.get("readout"):
        raise ValueError("Specify candidate families and an independent readout input")
    split = split_jobs(len(counts), config.get("block_size", 1), config.get("seed", 0))
    train = split["train"]
    train_counts = counts[train].sum(0)
    target_eligible = train_counts[:, space.target].sum(1) >= config.get("min_target_train", 100)
    # I_L and its locked value need all G states, including the 53 orbit.
    w_eligible = (train_counts[:, space.ground] >= config.get("min_state_train", 100)).all(1)
    channels, readout_metadata = channels_for(dataset, config["readout"])
    names = list(config["families"])
    all_latent, all_observed, all_fits = [], [], []
    for channel_set in channels:
        latent_draw, observed_draw, fits = [], [], {}
        for name in names:
            family_config = config["families"][name]
            candidates = family_config["candidates"]
            if not candidates:
                raise ValueError(f"No candidates for {name}")
            scope = family_config.get("fit_scope", "full")
            mask = _scope(space, scope)
            profile = []
            best_score, best = -np.inf, None
            for candidate in candidates:
                scores = []
                # All tuning, including l2 and free sector weights, uses grouped CV on A only.
                for a, b in grouped_folds(train, split["block_ids"], config.get("cv_folds", 3)):
                    latent, _ = _fit_candidate(dataset, candidate, counts[a], channel_set, scope)
                    observed = _observed(latent, channel_set)
                    scores.extend(score_jobs(counts[b], observed, mask,
                                             target_eligible if scope in {"T", "62_53"} else None))
                value = score_mean(scores)
                profile.append({"candidate": candidate, "grouped_cv_score": float(value) if np.isfinite(value) else None})
                if value > best_score:
                    best_score, best = value, candidate
            if best is None:
                raise ValueError(f"No candidate in {name} supports the development observations in scope {scope}")
            latent, fitted = _fit_candidate(dataset, best, counts[train], channel_set, scope)
            latent_draw.append(latent)
            observed_draw.append(_observed(latent, channel_set))
            fits[name] = {"fit_scope": scope, "selected": fitted, "profile": profile}
        all_latent.append(latent_draw)
        all_observed.append(observed_draw)
        all_fits.append(fits)
    with output_directory(output) as out:
        np.savez_compressed(out / "predictions.npz", observed=np.array(all_observed), latent=np.array(all_latent),
                            target_eligible=target_eligible, w_eligible=w_eligible)
        write_json(out / "frozen.json", {
            "schema_version": 1, "dataset_fingerprint": dataset.fingerprint,
            "condition": dataset.condition, "job_ids": dataset.manifest["job_ids"],
            "config": config, "config_fingerprint": canonical_hash(config), "split": split,
            "model_names": names, "fits_by_readout_draw": all_fits, "readout": readout_metadata,
            "predictions_sha256": file_hash(out / "predictions.npz"),
            "training_counts_sha256": canonical_hash(counts[train]),
            "interpretation": "Frozen validation analysis; no test job was used for fitting or eligibility"})
    return read_json(Path(output) / "frozen.json")


def evaluate_frozen(dataset_path, frozen_path, output, repeats=500, seed=0):
    dataset = Dataset(dataset_path)
    frozen_path = Path(frozen_path)
    frozen = read_json(frozen_path / "frozen.json")
    if dataset.fingerprint != frozen["dataset_fingerprint"]:
        raise ValueError("Frozen prediction targets a different dataset")
    if file_hash(frozen_path / "predictions.npz") != frozen["predictions_sha256"]:
        raise ValueError("Frozen predictions changed")
    with np.load(frozen_path / "predictions.npz", allow_pickle=False) as data:
        observed, latent = data["observed"], data["latent"]
        eligible, w_eligible = data["target_eligible"], data["w_eligible"]
    indices, blocks = frozen["split"]["test"], frozen["split"]["block_ids"]
    counts, space = dataset.counts, dataset.space
    names = frozen["model_names"]
    scores = {}
    for m, name in enumerate(names):
        scores[name] = {}
        for scope in ("full", "T", "G"):
            values = [score_mean(score_jobs(counts[indices], p, _scope(space, scope), eligible if scope == "T" else None))
                      for p in observed[:, m]]
            scores[name][scope] = [v if np.isfinite(v) else None for v in values]
        scores[name]["observed_predictions"] = [summarize(space, p) for p in observed[0, m]]
        scores[name]["latent_predictions"] = [summarize(space, p) for p in latent[0, m]]
    comparisons = {}
    for a in range(len(names)):
        for b in range(a):
            comparisons[f"{names[a]} minus {names[b]}"] = {
                scope: paired_score_interval(counts, observed[:, a], observed[:, b], indices, blocks,
                                              mask=_scope(space, scope), eligible=eligible if scope == "T" else None,
                                              repeats=repeats, seed=seed)
                for scope in ("full", "T")}
    null_name = frozen["config"].get("conditional_null")
    omnibus = None
    if null_name:
        m = names.index(null_name)
        tests = [wild_cluster_omnibus(counts, p, indices, blocks, space.target, eligible,
                                      repeats=repeats, seed=seed) for p in observed[:, m]]
        ps = [t["pvalue"] for t in tests]
        omnibus = {"model": null_name, "by_readout_draw": tests,
                    "worst_grid_pvalue": max(ps) if all(p is not None for p in ps) else None,
                    "scope": "finite supplied nuisance scenarios; not an unrestricted global-mixture exclusion"}
    reports = observable_intervals(space, counts, indices, blocks, repeats, seed, w_eligible)
    report = {"schema_version": 1, "dataset_fingerprint": dataset.fingerprint,
              "frozen_fingerprint": canonical_hash(frozen), "condition": dataset.condition,
              "status": "exploratory until dependence and nuisance size calibration is validated",
              "scores": scores, "paired_comparisons": comparisons, "conditional_omnibus": omnibus,
              "embedding_observables": reports, "W_eligible_from_training": w_eligible.tolist(),
              "readout": frozen["readout"],
              "warnings": ["Observed W can violate a latent pairwise identity after readout or mixtures.",
                           "Quantum W attribution requires jointly validated sector predictions.",
                           "Readout sensitivity draws are not independent experimental replications."]}
    write_json(output, report)
    return report
