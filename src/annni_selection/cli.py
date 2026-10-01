"""Command-line entry points; all commands are offline unless collect --execute."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
from .io import read_json, write_json, json_default


def _emit(value, path=None):
    if path:
        write_json(path, value)
        print(path)
    else:
        print(json.dumps(value, indent=2, default=json_default, allow_nan=False))


def demo(output, seed=12):
    from .simulation import synthetic_dataset
    from .readout import calibrate_controls
    from .inference import fit_dataset, evaluate_frozen
    from .verification import exact_report
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    synthetic_dataset(output / "data", {"family": "classical", "beta": 1.5}, jobs=20,
                      reads_per_job=1500, embeddings=3, epsilon=.01, shared_fraction=.2, seed=seed)
    rng = np.random.default_rng(seed + 1)
    controls = output / "synthetic_controls.csv"
    with controls.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["block_id", "qubit", "true_spin", "trials", "errors"])
        for block in range(12):
            for qubit in range(24):
                for spin in (-1, 1):
                    writer.writerow([block, qubit, spin, 10000, rng.binomial(10000, .01)])
    calibrate_controls(controls, output / "calibration", preparation_status="synthetic",
                       provenance="Independent synthetic known-input controls, separate RNG from target samples",
                       context={"device": "synthetic", "graph_id": "synthetic"}, draws=3, seed=seed)
    config = {"seed": seed, "block_size": 1, "cv_folds": 3, "min_target_train": 100,
              "min_state_train": 20, "readout": {"kind": "calibration", "path": str(output / "calibration"), "draws": 3},
              "conditional_null": "classical",
              "families": {
                  "classical": {"candidates": [{"family": "classical", "beta": b} for b in (1, 1.5, 2)]},
                  "quantum": {"candidates": [{"family": "quantum", "beta": b, "gamma": g} for b in (1.5, 2) for g in (0, .1, .2)]},
                  "susceptibility": {"candidates": [{"family": "susceptibility", "beta": 1.5, "chi": c} for c in (-.02, 0, .02)]},
                  "pairwise": {"candidates": [{"family": "pairwise_fit", "l2": r} for r in (.001, .01)]}}}
    write_json(output / "config.json", config)
    fit_dataset(output / "data", config, output / "frozen")
    evaluate_frozen(output / "data", output / "frozen", output / "report.json", repeats=199, seed=seed)
    write_json(output / "exact_checks.json", exact_report())
    return {"data": str(output / "data"), "frozen_predictions": str(output / "frozen"),
            "report": str(output / "report.json"), "purpose": "classical-plus-readout false-selection control; synthetic only"}


def parser():
    p = argparse.ArgumentParser(prog="annni", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("exact", help="Reproduce proposal enumeration and numerical tables")
    q.add_argument("--output")
    q.add_argument("--dictionary", action="store_true")
    q = sub.add_parser("inventory", help="Index archive files and create decoder manifests")
    q.add_argument("root"); q.add_argument("--embeddings", required=True); q.add_argument("--output", required=True)
    q = sub.add_parser("decode", help="Decode a manifest into synchronized logical samples")
    q.add_argument("manifest"); q.add_argument("--output", required=True)
    q = sub.add_parser("summarize", help="Descriptive per-embedding statistics; no pooling-based W claim")
    q.add_argument("dataset"); q.add_argument("--output", required=True)
    q = sub.add_parser("calibrate-readout", help="Independent known-input controls; never target contrasts")
    q.add_argument("csv"); q.add_argument("--output", required=True)
    q.add_argument("--preparation-status", choices=["verified", "bounded", "unresolved", "synthetic"], required=True)
    q.add_argument("--provenance", required=True); q.add_argument("--device", required=True); q.add_argument("--graph-id", required=True)
    q.add_argument("--method", choices=["block_bootstrap", "iid_jeffreys"], default="block_bootstrap")
    q.add_argument("--draws", type=int, default=32); q.add_argument("--seed", type=int, default=0)
    q.add_argument("--preparation-error-bound", type=float, default=0)
    q = sub.add_parser("fit", help="Fit and tune on A; save frozen predictions for B")
    q.add_argument("dataset"); q.add_argument("--config", required=True); q.add_argument("--output", required=True)
    q = sub.add_parser("evaluate", help="Evaluate an existing frozen artifact on B")
    q.add_argument("dataset"); q.add_argument("frozen"); q.add_argument("--output", required=True)
    q.add_argument("--repeats", type=int, default=500); q.add_argument("--seed", type=int, default=0)
    q = sub.add_parser("freeze-transfer", help="Fit shared candidates without recipient-condition data")
    q.add_argument("study"); q.add_argument("--output", required=True)
    q = sub.add_parser("evaluate-study", help="Held-out conditions and a fixed Holm family")
    q.add_argument("frozen"); q.add_argument("--output", required=True)
    q.add_argument("--repeats", type=int, default=500); q.add_argument("--seed", type=int, default=0)
    q = sub.add_parser("simulate", help="Generate a synthetic synchronized dataset")
    q.add_argument("spec"); q.add_argument("--output", required=True)
    q = sub.add_parser("null-size", help="Calibrate the multiplier test under a declared synthetic null")
    q.add_argument("spec"); q.add_argument("--output", required=True)
    q = sub.add_parser("procedure-study", help="Repeat independent controls, fitting and validation on synthetic scenarios")
    q.add_argument("spec"); q.add_argument("--output", required=True)
    q = sub.add_parser("evolve", help="Schedule-driven classical, coherent or phenomenological open dynamics")
    q.add_argument("spec"); q.add_argument("--output", required=True)
    q = sub.add_parser("plan", help="Randomized prospective validation plan, offline")
    q.add_argument("spec"); q.add_argument("--output", required=True)
    q = sub.add_parser("collect", help="Dry-run a plan; --execute explicitly enables paid QPU jobs")
    q.add_argument("plan"); q.add_argument("--embeddings", required=True); q.add_argument("--output", required=True)
    q.add_argument("--shim-files", help="JSON mapping of setting keys to frozen physical shim files")
    q.add_argument("--execute", action="store_true")
    q = sub.add_parser("shim-update", help="Propose a symmetry-only update from separate calibration counts")
    q.add_argument("spec"); q.add_argument("--output", required=True)
    q = sub.add_parser("plot", help="Export a scientific figure from an evaluated report")
    q.add_argument("report"); q.add_argument("--output", required=True)
    q = sub.add_parser("demo", help="Complete offline synthetic readout-confound demonstration")
    q.add_argument("--output", required=True); q.add_argument("--seed", type=int, default=12)
    return p


def _resolve_readout(spec, parent):
    if spec.get("kind") == "calibration":
        spec["path"] = str((parent / spec["path"]).resolve())
    return spec


def _resolve_models(config, parent):
    def resolve(candidate):
        if candidate["family"] == "dynamics":
            candidate["schedule"]["path"] = str((parent / candidate["schedule"]["path"]).resolve())
        for component in candidate.get("components", []):
            resolve(component)
    for family in config["families"].values():
        for candidate in family["candidates"]:
            resolve(candidate)


def run(args):
    from .states import StateSpace
    from .data import Dataset, decode_manifest
    if args.command == "exact":
        from .verification import exact_report
        _emit(StateSpace().dictionary() if args.dictionary else exact_report(), args.output)
    elif args.command == "inventory":
        from .catalog import inventory
        result = inventory(args.root, args.embeddings, args.output)
        _emit({"entries": len(result["entries"]), "missing_primary": len(result["missing_primary_conditions"]), "output": args.output})
    elif args.command == "decode":
        data = decode_manifest(args.manifest, args.output)
        _emit({"shape": data.counts.shape, "fingerprint": data.fingerprint, "output": args.output})
    elif args.command == "summarize":
        from .observables import summarize
        from .statistics import dependence_diagnostics
        data = Dataset(args.dataset)
        result = {"condition": data.condition, "fingerprint": data.fingerprint,
                  "embedding_summaries": [summarize(data.space, c) for c in data.counts.sum(0)],
                  "interpretation": "Descriptive observed distributions; static-pairwise null not guaranteed after job mixtures/readout"}
        if (data.path / "state_ids.npy").exists():
            result["dependence"] = dependence_diagnostics(data.space, data.state_ids(), data.offsets)
        _emit(result, args.output)
    elif args.command == "calibrate-readout":
        from .readout import calibrate_controls
        cal = calibrate_controls(args.csv, args.output, preparation_status=args.preparation_status,
              provenance=args.provenance, context={"device": args.device, "graph_id": args.graph_id},
              draws=args.draws, method=args.method, seed=args.seed, preparation_error_bound=args.preparation_error_bound)
        _emit({"fingerprint": cal.fingerprint, "output": args.output})
    elif args.command == "fit":
        from .inference import fit_dataset
        config = read_json(args.config)
        _resolve_readout(config["readout"], Path(args.config).resolve().parent)
        _resolve_models(config, Path(args.config).resolve().parent)
        fit_dataset(args.dataset, config, args.output)
        print(args.output)
    elif args.command == "evaluate":
        from .inference import evaluate_frozen
        evaluate_frozen(args.dataset, args.frozen, args.output, args.repeats, args.seed)
        print(args.output)
    elif args.command == "freeze-transfer":
        from .study import freeze_transfer
        study = read_json(args.study); parent = Path(args.study).resolve().parent
        _resolve_models(study, parent)
        for entry in study["datasets"]:
            entry["path"] = str((parent / entry["path"]).resolve())
            _resolve_readout(entry["readout"], parent)
        freeze_transfer(study, args.output); print(args.output)
    elif args.command == "evaluate-study":
        from .study import evaluate_study
        evaluate_study(args.frozen, args.output, args.repeats, args.seed); print(args.output)
    elif args.command == "simulate":
        from .simulation import synthetic_dataset
        synthetic_dataset(args.output, **read_json(args.spec)); print(args.output)
    elif args.command == "null-size":
        from .simulation import null_size_study
        _emit(null_size_study(**read_json(args.spec)), args.output)
    elif args.command == "procedure-study":
        from .simulation import procedure_study
        procedure_study(args.output, **read_json(args.spec)); print(args.output)
    elif args.command == "evolve":
        from .dynamics import Schedule, classical_evolve, quantum_evolve, convergence_check, noise_ensemble
        from .observables import summarize
        spec = read_json(args.spec); sch = spec.pop("schedule"); kind = spec.pop("kind")
        schedule = Schedule.from_csv(Path(args.spec).resolve().parent / sch["path"], sch["protocol"], sch["provenance"])
        if spec.pop("target_protocol", schedule.protocol) != schedule.protocol:
            raise ValueError("Schedule protocol does not match the target acquisition")
        fn = classical_evolve if kind == "classical" else quantum_evolve
        if kind not in {"classical", "coherent", "thermal_reset"}:
            raise ValueError("Unknown dynamical model")
        if kind == "coherent" and (spec.get("reset_rate", 0) or spec.get("dephasing_rate", 0)):
            raise ValueError("Coherent model cannot specify dissipative rates")
        if "amplitudes" in spec:
            spec["amplitudes"] = {int(k): v for k, v in spec["amplitudes"].items()}
            spec["exponents"] = {int(k): v for k, v in spec["exponents"].items()}
        fields = spec.pop("field_samples", None)
        steps = spec.pop("steps", 100)
        if fields is None:
            p, check = convergence_check(fn, steps=steps, space=StateSpace(), schedule=schedule, **spec)
        else:
            def ensemble(**kw):
                return noise_ensemble(fn, fields, **kw)
            p, check = convergence_check(ensemble, steps=steps, space=StateSpace(), schedule=schedule, **spec)
        _emit({"probabilities": p, "observables": summarize(StateSpace(), p), "convergence": check,
               "model": kind, "schedule_provenance": schedule.provenance,
               "scope": "thermal-reset bath is a phenomenological sensitivity model"}, args.output)
    elif args.command == "plan":
        from .acquisition import validation_plan
        _emit(validation_plan(**read_json(args.spec)), args.output)
    elif args.command == "collect":
        from .acquisition import execute_validation_plan
        shims = read_json(args.shim_files) if args.shim_files else None
        _emit(execute_validation_plan(read_json(args.plan), args.embeddings, args.output,
                                      execute=args.execute, shim_files=shims))
    elif args.command == "shim-update":
        from .acquisition import shim_update
        from .readout import ReadoutChannel
        spec = read_json(args.spec)
        if spec.pop("role", None) != "shim_calibration":
            raise ValueError("Shim input must be marked as separate shim_calibration data")
        rates = spec.pop("readout")
        _emit(shim_update(StateSpace(), channel=ReadoutChannel(**rates), **spec), args.output)
    elif args.command == "plot":
        from .reporting import plot_report
        plot_report(args.report, args.output); print(args.output)
    elif args.command == "demo":
        _emit(demo(args.output, args.seed))


def main():
    try:
        run(parser().parse_args())
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"annni: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
