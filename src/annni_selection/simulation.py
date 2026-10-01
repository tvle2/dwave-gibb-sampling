"""Synthetic observations preserve simultaneous copies and optional temporal dependence."""
import numpy as np
from .data import save_dataset
from .models import distribution, normalize
from .readout import ReadoutChannel
from .statistics import split_jobs, wild_cluster_omnibus
from .states import StateSpace


def sample_synchronized(probabilities, reads, rng, shared_fraction=0.0, correlation_span=1):
    p = np.array([normalize(q) for q in probabilities])
    if not 0 <= shared_fraction <= 1 or correlation_span < 1 or reads < 1:
        raise ValueError("Invalid synthetic dependence settings")
    # Every marginal is exactly p_e. Shared uniforms retain cross-copy covariance;
    # piecewise-constant shared uniforms supply a controlled temporal stress test.
    u = rng.random((reads, len(p)))
    common = np.repeat(rng.random((reads + correlation_span - 1) // correlation_span), correlation_span)[:reads]
    use = rng.random(reads) < shared_fraction
    u[use] = common[use, None]
    states = np.column_stack([np.searchsorted(np.cumsum(q), u[:, e], side="right") for e, q in enumerate(p)])
    return np.minimum(states, p.shape[1] - 1).astype(np.uint8 if p.shape[1] <= 256 else np.uint16)


def synthetic_dataset(output, model, *, jobs=20, reads_per_job=1000, embeddings=4, epsilon=.001,
                      seed=0, shared_fraction=0.0, correlation_span=1, condition=None):
    if min(jobs, reads_per_job, embeddings) < 1:
        raise ValueError("Positive dimensions are required")
    space = StateSpace()
    condition = condition or {"n": 8, "kappa": .5, "energy_scale": 1.0,
                              "anneal_time_us": .05, "device": "synthetic", "graph_id": "synthetic"}
    latent = distribution(space, model, condition)
    observed = ReadoutChannel.symmetric(space.n, epsilon).apply(latent)
    probabilities = np.repeat(observed[None, :], embeddings, axis=0)
    rng = np.random.default_rng(seed)
    counts = np.zeros((jobs, embeddings, space.size), np.int64)
    states = []
    for j in range(jobs):
        sampled = sample_synchronized(probabilities, reads_per_job, rng, shared_fraction, correlation_span)
        states.append(sampled)
        for e in range(embeddings):
            counts[j, e] = np.bincount(sampled[:, e], minlength=space.size)
    physical = np.arange(embeddings * space.n).reshape(embeddings, space.n)
    return save_dataset(output, counts, physical, condition, [f"synthetic-{j}" for j in range(jobs)],
                        np.concatenate(states), {"boundary_status": "synthetic", "model": model, "epsilon": epsilon,
                        "seed": seed, "shared_fraction": shared_fraction, "correlation_span": correlation_span})


def null_size_study(model, *, repetitions=100, bootstrap_repeats=499, jobs=32, reads=1000,
                    embeddings=4, epsilon=.001, shared_fraction=0.0, correlation_span=1, seed=0):
    """Size check for the specified frozen null, not certification for unknown hardware noise."""
    if repetitions < 1 or bootstrap_repeats < 1 or jobs < 8:
        raise ValueError("Use positive repetitions and at least eight jobs for null size checks")
    space = StateSpace()
    p = ReadoutChannel.symmetric(8, epsilon).apply(distribution(space, model))
    probabilities = np.repeat(p[None, :], embeddings, axis=0)
    rng = np.random.default_rng(seed)
    split = split_jobs(jobs, 1, seed)
    pvalues = []
    for repetition in range(repetitions):
        counts = np.zeros((jobs, embeddings, space.size), np.int64)
        for j in range(jobs):
            states = sample_synchronized(probabilities, reads, rng, shared_fraction, correlation_span)
            for e in range(embeddings):
                counts[j, e] = np.bincount(states[:, e], minlength=space.size)
        result = wild_cluster_omnibus(counts, probabilities, split["test"], split["block_ids"],
                                      space.target, np.ones(embeddings, bool), repeats=bootstrap_repeats,
                                      seed=seed + repetition + 1)
        pvalues.append(result["pvalue"])
    if any(p is None for p in pvalues):
        raise ValueError("A simulated null test was ineligible; cannot report a rejection rate")
    k = sum(p is not None and p <= .05 for p in pvalues)
    from scipy.stats import beta
    interval = [0.0 if k == 0 else float(beta.ppf(.025, k, repetitions - k + 1)),
                1.0 if k == repetitions else float(beta.ppf(.975, k + 1, repetitions - k))]
    return {"repetitions": repetitions, "rejections_at_0.05": k, "rate": k / repetitions,
            "binomial_95_interval": interval, "pvalues": pvalues,
            "settings": {"model": model, "epsilon": epsilon, "shared_fraction": shared_fraction,
                         "correlation_span": correlation_span, "jobs": jobs, "reads": reads, "embeddings": embeddings},
            "interpretation": "size of this frozen-null stress scenario only; rerun complete fitting for selection calibration"}


def procedure_study(output, scenarios, analysis_config, *, repetitions=20, bootstrap_repeats=199,
                    controls_per_block=5000, control_blocks=8, seed=0):
    """Repeat synthetic acquisition, independent controls, A-only tuning and B validation.

    Reports score preferences and null-test size/power for the declared scenarios;
    it does not turn a predictive-score preference into physical mechanism exclusion.
    """
    import copy
    import csv
    from pathlib import Path
    from scipy.stats import beta as beta_distribution
    from .io import read_json, write_json
    from .readout import calibrate_controls
    from .inference import fit_dataset, evaluate_frozen
    if repetitions < 1 or control_blocks < 2 or controls_per_block < 1:
        raise ValueError("Positive repetitions/trials and at least two control blocks are required")
    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    summaries = []
    for scenario_index, scenario in enumerate(scenarios):
        identifier = scenario["id"]
        if Path(identifier).name != identifier or identifier in {".", ".."}:
            raise ValueError("Scenario IDs must be safe directory names")
        if any(s["id"] == identifier for s in summaries):
            raise ValueError("Duplicate scenario ID")
        pvalues, preferences = [], {}
        for repetition in range(repetitions):
            run_seed = seed + 100003 * scenario_index + repetition * 101
            directory = root / identifier / f"replicate-{repetition:04d}"
            directory.mkdir(parents=True)
            parameters = dict(scenario["simulation"])
            parameters["seed"] = run_seed
            data = synthetic_dataset(directory / "data", **parameters)
            epsilon = parameters.get("epsilon", .001)
            rng = np.random.default_rng(run_seed + 1)
            controls = directory / "controls.csv"
            with controls.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["block_id", "qubit", "true_spin", "trials", "errors"])
                for block in range(control_blocks):
                    for qubit in data.physical.ravel():
                        for spin in (-1, 1):
                            writer.writerow([block, qubit, spin, controls_per_block,
                                             rng.binomial(controls_per_block, epsilon)])
            calibrate_controls(controls, directory / "calibration", preparation_status="synthetic",
                               provenance="Independently simulated known-input controls",
                               context={k: data.condition[k] for k in ("device", "graph_id")}, draws=3, seed=run_seed + 2)
            config = copy.deepcopy(analysis_config)
            config["seed"] = run_seed + 3
            config["readout"] = {"kind": "calibration", "path": str(directory / "calibration"), "draws": 3}
            write_json(directory / "config.json", config)
            fit_dataset(data.path, config, directory / "frozen")
            result = evaluate_frozen(data.path, directory / "frozen", directory / "report.json",
                                     bootstrap_repeats, run_seed + 4)
            omnibus = result["conditional_omnibus"]
            pvalues.append(omnibus["worst_grid_pvalue"] if omnibus else None)
            for name, scopes in result["paired_comparisons"].items():
                interval = scopes["T"].get("interval")
                preferences.setdefault(name, []).append(None if interval is None else interval[0] > 0)
        valid = [p for p in pvalues if p is not None]
        k, n = sum(p <= .05 for p in valid), len(valid)
        ci = None if n == 0 else [0.0 if k == 0 else float(beta_distribution.ppf(.025, k, n-k+1)),
                                 1.0 if k == n else float(beta_distribution.ppf(.975, k+1, n-k))]
        summaries.append({"id": identifier, "scenario": scenario, "eligible_repetitions": n,
                          "omnibus_rejections": k, "rate": k/n if n else None,
                          "binomial_95_interval": ci, "pvalues": pvalues,
                          "conditional_score_interval_above_zero": preferences})
        write_json(root / "progress.json", {"completed_scenarios": summaries})
    result = {"scenarios": summaries, "repetitions": repetitions,
              "scope": "complete fitting procedure for these declared synthetic scenarios only"}
    write_json(root / "summary.json", result)
    return result
