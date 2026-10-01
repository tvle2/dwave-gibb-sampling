# ANNNI state selection on programmable quantum annealers

An offline analysis pipeline for the eight-spin periodic ANNNI experiment,
with explicit classical alternatives, full quantum Gibbs predictions,
independent readout calibration and held-out validation. Existing legacy
archives are supported. Importing the package or running an analysis command
does not connect to D-Wave.

The implementation follows the revised proposal's distinction between a
model consistency test and mechanism discrimination. Agreement inside the
40-state manifold is insufficient to exclude general pairwise interactions.
Observed contrast signs and observed `W` also require an observation model.

## Install and verify

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
annni exact --output results/exact_checks.json
annni demo --output results/demo
```

The demo generates **synthetic classical Gibbs samples with 1% readout error**,
independent synthetic calibration controls, and synchronized parallel copies.
It fits on development jobs A and evaluates frozen predictions on jobs B.
The example's 1% error rate is a stress scenario, not a measured device rate.
Run `annni --help` for the complete command list. Setting
`OPENBLAS_NUM_THREADS=1` can reduce overhead for these small matrices.

Core dependencies are NumPy and SciPy. Plotting and QPU access are optional:

```bash
python -m pip install -e '.[plots]'
annni plot results/demo/report.json --output results/demo/observables.png
```

## Start with the existing archive

```bash
annni inventory /Volumes/ThinhLe/data --embeddings embeddings --output results/catalog
```

`results/catalog/catalog.json` lists the settings, missing members of the
64-condition primary family, and decoder manifest filenames. The inventory
does not read every large sample file. Choose an actual manifest it produced:

```bash
annni decode results/catalog/SETTING.json --output results/SETTING
annni summarize results/SETTING --output results/SETTING_summary.json
annni fit results/SETTING --config configs/analysis_exploratory.json --output results/SETTING_fit
annni evaluate results/SETTING results/SETTING_fit --output results/SETTING_report.json
```

Replace `SETTING` with the inventory's exact name. The decoder recognizes
zlib-compressed `.json.zip` files by their bytes; they are not assumed to be
ZIP containers. It streams physical rows, rejects invalid active spins,
checks mappings and graph IDs, and keeps synchronized logical state IDs.
Legacy job boundaries reconstructed from fixed read counts are explicitly
marked **inferred**, not observed job IDs or verified physical timestamps.

Numerical grids, readout ranges, block sizes and occupancy thresholds in
`configs/` are **development examples**, not a preregistration or calibrated
nuisance envelope. Set them using controls/development data, then freeze
them. Fit outputs record fingerprints, selected parameters, grouped CV
profiles, splits and eligibility. Existing output directories are not
overwritten. Evaluate does not refit any model.

## Independent readout calibration

Prepare separate controls with both known spin signs on every physical qubit.
The CSV columns are:

```text
block_id,qubit,true_spin,trials,errors
0,42,-1,10000,8
0,42,1,10000,12
```

Include every used qubit and multiple independently acquired blocks. Controls
must not be counts conditioned on the target manifold. Preparation/evolution
errors must be separately assessed; a prescribed input alone does not establish
perfect preparation.

```bash
annni calibrate-readout controls.csv --output results/readout_calibration \
  --device Advantage_system6 --graph-id 01dae5a273 \
  --preparation-status verified --provenance 'Independent controls; record epoch and protocol here'
annni fit results/SETTING --config configs/analysis_calibrated.json --output results/SETTING_calibrated_fit
```

Use `verified` only with supporting evidence. `bounded` requires
`--preparation-error-bound`; `unresolved` retains that limitation. The block
bootstrap keeps all physical qubits together, and rates are bound to the
declared device/graph. Inference propagates supplied calibration scenarios
without fitting error rates to target contrasts. They form a **sensitivity
ensemble**, not a claim of a complete Bayesian posterior or a worst-case bound
over all asymmetric/correlated channels.

## Models and tests

| Component | Implementation |
|---|---|
| Exact dictionary | 256 states; `(C1,C2)` classes, domain walls, full symmetry orbits, flip graphs |
| Classical alternatives | Gibbs, uniform field, susceptibility `J + chi J^2`, normalized mixtures, all-fields/all-pairs fits |
| Quantum predictions | Full Hamiltonian; controlled projected limit; descriptive free sector weights |
| Observation | Asymmetric independent physical-qubit flips, gauge transformation, full-state leakage before conditioning |
| Statistics | Arithmetic `R1/R2`, log-mean `I_L`, pairwise-required curvature, `W`, eligibility, shared bootstrap covariance |
| Validation | A-only grouped CV; frozen B predictions; equal job/embedding scores; paired block resampling |
| Transfer | Shared candidates selected on development conditions; no recipient-condition fitting; fixed Holm family |
| Dynamics | Single/block heat-bath master equation, coherent evolution, CP thermal-reset/dephasing sensitivity, slow-field ensembles |
| Acquisition | Randomized plans, gauges, calibration-only shim updates, metadata, hashes and resumable raw jobs |

`R2 > 0` is a projected result, not a full-Hamiltonian theorem. The
susceptibility comparison uses `D_chi = R2 + R1/3` at `kappa=0.5`, with
readout applied to each prediction. `W` is computed separately for every
embedding. It is never treated as a single-static-model invariant of a pooled
embedding histogram. Zero counts produce undefined log invariants, not
undisclosed pseudocounts.

Each candidate can choose `fit_scope`: `full`, `G`, `T`, or `62_53`.
`projected_consistency` fits using only the 6+2 and 5+3 outcomes and predicts
the withheld 4+4 contrast. It remains a consistency test. `sector_fit` learns
sector weights on A and freezes them for B; it is a descriptive model, not
independent evidence for quantum dynamics.

## Scale transfer and synthetic validation

Edit `configs/transfer_template.json` with real dataset paths, independent
readout inputs and physically supported candidate parameters:

```bash
annni freeze-transfer configs/transfer_template.json --output results/transfer
annni evaluate-study results/transfer --output results/transfer_reports
annni null-size configs/null_size.json --output results/null_size.json
annni procedure-study configs/procedure_study.json --output results/procedure_study
```

`susceptibility_transfer` holds `beta_per_scale` and `chi` fixed, giving the
quadratic programmed-scale law at a common physical sampling point.
`physical_snapshot` uses calibrated A, B and temperature, so beta and Gamma
change with scale. These are hypotheses to validate, not assumptions that all
anneals freeze at the same point. Fit separate device calibrations rather
than assume identical bath parameters across processors.

`null-size` tests a specified frozen-null scenario. `procedure-study` repeats
the **whole** synthetic workflow, including new independent calibration data,
development-only fitting and held-out evaluation. Its size/power results apply
only to the supplied truth, nuisance and dependence scenarios. Do not treat a
small synthetic run as validation of the unknown hardware error model.

Omnibus outputs are deliberately labeled exploratory until their finite-sample
size and nuisance/dependence coverage are established. Missing primary tests
count as p=1 in the declared Holm family; the family is not silently shrunk.
A predictive-score preference or a rejected static model is not a universal
quantum witness.

## Dynamics

```bash
annni evolve configs/dynamics_classical.json --output results/classical_dynamics.json
annni evolve configs/dynamics_coherent.json --output results/coherent_dynamics.json
annni evolve configs/dynamics_open_sensitivity.json --output results/open_sensitivity.json
```

These shipped schedules are **synthetic**, not device calibrations. Supply CSV
columns `fraction,A_GHz,B_GHz`, where fraction is elapsed-time fraction after
the protocol's control-coordinate mapping. Energies are energy/h in GHz,
temperature is mK, time is microseconds, and stochastic rates are per
microsecond. Standard schedules must not stand in for fast schedules.
Every run compares successive step refinements. A `dynamics` candidate can
also be used directly in fitting/transfer with shared rates, a calibrated
schedule and an explicit `max_refinement_tv` tolerance; it is then passed
through the same independent readout channel as the static candidates.

The thermal-reset bath is a **phenomenological completely positive model**.
Its combination with dephasing and slow-field disorder provides environmental
sensitivity checks; it does not implement or validate a microscopic hybrid-noise
or strong-coupling theory. Physical bath applicability, actual schedule
calibration, correlated readout and calibration transfer require experimental
inputs. See [the implementation notes](docs/implementation.md).

## Future acquisition

The existing sampler retains one final compressed sample array per setting,
now named `_solutions.json.zlib`, and a sidecar with actual job boundaries,
IDs, timestamps, settings and hashes. Individual jobs are checkpointed while
acquiring, then streamed into the final array; verified checkpoints are removed
only after the final array and manifest are durable. Existing legacy files are
left intact. Compression level 1 avoids the former whole-setting memory build.

```bash
python -m pip install -e '.[qpu]'
python src/sample_dwave.py --output /Volumes/ThinhLe/data
# Only when ready to spend QPU time:
python src/sample_dwave.py --output /Volumes/ThinhLe/data --execute
```

The first command is a dry run. The archival sweep/order is retained. For the
new randomized validation campaign:

```bash
annni plan configs/validation_plan.json --output results/validation_plan.json
annni collect results/validation_plan.json --embeddings embeddings --output data/validation
```

`collect` is also a dry run until `--execute` is supplied. It validates every
request before submitting any. Shimmed variants require independently frozen,
gauge-specific physical corrections and verified flux-bias bounds. Stale
graphs, missing shims, changed NN/NNN class means, unsupported parameters and
out-of-range couplings stop preflight. Failed submissions are not retried
blindly because failure does not prove the job was not charged.

Known-state readback controls use separate supported preparation protocols;
programmed fields and standard reverse initialization cannot be combined with
the forward fast-anneal protocol. See D-Wave's
[solver parameters](https://docs.dwavequantum.com/en/latest/quantum_research/solver_parameters.html).
No QPU jobs were run to develop or test this implementation.
