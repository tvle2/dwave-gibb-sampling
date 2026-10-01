# Implementation and scientific scope

This implementation starts from repository commit
`98dc5355a6b6febcbf50eeb5a135251deebb6a05` and implements the revised
eight-spin state-selection proposal. It provides reproducible analysis and
acquisition software; it does not supply missing QPU samples, calibration
measurements, or a mechanism conclusion.

## Conventions

The classical energy is `E = -C1 + kappa*C2`, with
`Cr = sum_i z_i*z_(i+r)` and `z_i` in {-1,+1}. State index is
`sum_i ((z_i+1)/2)*2**(N-1-i)`: logical spin 0 is the most significant bit,
bit 0 means -1 and bit 1 means +1. Do not mix this with an implicit Qiskit
basis convention. N=8 contrasts are explicitly rejected for other ring sizes.

The quantum Hamiltonian is `diag(E) - Gamma*sum(Xi)`; `beta` is dimensionless.
The physical convention is `H/h = -A_GHz/2*sum(Xi) + lambda*B_GHz/2*E`.
Unitary propagation uses `2*pi*1000` to convert GHz and microseconds.
`kBT/h` in GHz is `0.020836619123... * temperature_mK`.
Stochastic rate parameters are inverse microseconds.

Distance-four correlations count each antipodal pair twice. The susceptibility
routine builds the full symmetric programmed matrix before taking `J + chi*J@J`
and evaluates unordered interactions only once. This also includes the NN/NNN
renormalizations; they cancel from the within-T contrast relation.

## Proposal-to-code mapping

| Proposal content | Code and behavior |
|---|---|
| Equations 1–7: matching and mixtures | `states.py`, `classical_gibbs`, `mixture`; normalize full components before mixing/conditioning |
| Equations 8–15: transverse-field models | `quantum_gibbs`, `projected_gibbs`; stable thermal eigenweights, complete spectrum |
| Equations 16–19 and 18a–b: pairwise constraint | `summarize`; separate arithmetic means and means of logs; correlated `I_L/I_L_pair/W` resamples |
| Equation 20: exact emulation | `emulating_pairwise`; tested on all 40 conditional probabilities |
| Equation 20a: common-mode field | `uniform_field`; evaluate full observed distribution instead of a raw sign witness |
| Equations 21a–b: sector population | `sector_fit`; weights fitted on development data only, descriptive interpretation |
| Equation 22: KL identity | `kl_decomposition`; exact probabilities; no uncorrected empirical entropy mechanism claim |
| Equations 23–24c: observation | `readout.py`; physical-qubit asymmetric channel, independent controls, exact leading readout bias and precision budget |
| Equation 25: score | `score_jobs`; equal embedding/job weights, identical missing-occupancy handling |
| Equation 26: classical dynamics | `classical_generator`/`classical_evolve`; contiguous single/block moves and instantaneous detailed balance |
| Equation 27 and dynamics | `dynamics.py`; explicit units, schedules, coherent propagation and CP bath sensitivity |
| Equations 27a–c: crosstalk transfer | `susceptibility_gibbs`, `susceptibility_transfer`, `freeze_transfer` |
| Equation 27d: planning | `w_variance_coefficient`, `contrast_covariance`; not a substitute for cluster uncertainty |
| Calibration campaign | `validation_plan`, `shim_update`, `execute_validation_plan`, resumable `acquire_jobs` |
| Appendices | `annni exact` and numerical regression tests |

## Data and reproducibility

A decoded dataset contains:

- `counts.npz`: integer counts `[job, embedding, state]`, physical qubit maps,
  and cumulative job read offsets.
- `state_ids.npy`: memory-mappable state IDs `[physical read, embedding]`.
  Parallel copies from one read always stay in the same row.
- `manifest.json`: condition, job identifiers, source and output fingerprints,
  gauge metadata and boundary provenance.

Raw decoder manifests either list recorded jobs with paths/counts/IDs or
declare a legacy file with `reads_per_job` and `num_jobs`. A consolidated new
archive uses `sample_file` and recorded job offsets. Decode rejects mismatched
hashes, extra/missing rows, stale graph IDs, overlapping embeddings, missing
physical variables and active-spin values other than ±1. Dense SAPI rows may
contain inactive values outside the embeddings. Packed arrays require an
explicit `variable_order`; mappings with physical-label keys are also accepted.

For an old archive, divisibility by a job size cannot prove acquisition
boundaries. The file is marked `inferred_fixed_size` even when row counts match.
Timestamps cannot be reconstructed from the numerical samples. Dependence
diagnostics report index lags, not invented physical correlation times.

`inventory` groups new one-job requests by device, graph, kappa, scale, time,
gauge and shim variant, preserving acquisition order. Analyze different gauges
as separate strata: a single pooled channel or static-W identity is not valid
for their arbitrary mixture. Time/scale/device collections remain distinct
conditions. The standard primary condition list contains 64 entries; additional
gauge or shim analyses do not silently redefine that family.

## Fit and validation artifacts

`fit` allocates contiguous job blocks to A/B before fitting. All optimization,
candidate selection, regularization tuning and count eligibility use A.
Grouped CV never splits a block. The frozen artifact contains both latent and
observed predictions `[readout scenario, model, embedding, state]`. Each
readout scenario is fixed across a complete physical-data resample; rates are
not independently redrawn for every target shot.

`evaluate` checks dataset/prediction fingerprints and uses only B. It reports
embedding-resolved observables, their shared bootstrap covariance, full/ground/
target scores and paired whole-block score comparisons. Missing orbit counts
make log observables undefined. W eligibility is frozen using all ground-state
counts needed for the curvature decomposition; this can be stricter than the
four-orbit formula for W alone. No empty-count smoothing is hidden in tests.

The multiplier omnibus is an exploratory cluster procedure. It retains
simultaneous embeddings and within-block dependence, but assumes independent
blocks. The finite supplied readout/model grid is not a confidence set over all
classical mixtures, correlated errors or kinetic processes. Run `null-size`
and `procedure-study` over a defensible nuisance/dependence envelope before
claiming size control. The latter resimulates controls and repeats tuning.
Its result is still scenario-specific. Holm cannot fix invalid marginal tests.

The paired score interval combines block resampling with a supplied calibration
sensitivity ensemble. It is not a fully marginalized posterior interval and
does not include unspecified preparation or model discrepancy. Epoch, pattern,
hardware and protocol transfer of controls require separate assessment.

`freeze-transfer` selects shared fixed candidates using only development
conditions. For a recipient condition all jobs are held out and no physical
parameter, mixture weight or sector population is fitted there. Its eligibility
uses frozen predictions and read totals, not held-out state outcomes. Per-device
calibrations should be supplied separately; identical parameters across QPU
generations are not assumed by the software.

## Dynamics and remaining physical inputs

Classical propagation uses a sparse column generator and exponential actions;
its single/block flip rates satisfy detailed balance at a fixed schedule point.
Coherent propagation uses midpoint Hamiltonian exponentials. Quantum bath
sensitivity composes unitary evolution, local Z dephasing, and a thermal reset
map, all completely positive. Slow-field ensembles average normalized final
distributions. These are useful testable models, not a microscopic derivation
of the hardware bath. A device-validated hybrid-noise/strong-coupling model and
spin-vector benchmark are not claimed to be implemented.

For quantitative dynamics, provide the **fast-protocol** A/B calibration after
mapping controls to physical elapsed time, temperature uncertainty, meaningful
rate bounds, and finer integration until the refinement diagnostic is adequate.
A small difference between two steps is a diagnostic rather than a certified
error bound. The inference `dynamics` adapter requires an explicit tolerance
and fails if refinement exceeds it. Synthetic schedules are refused as hardware
calibration. Slow longitudinal fields in the model are normalized Ising fields;
they do not automatically represent the distinct schedule dependence of
physical flux-bias offsets. That dependence must be supplied by an appropriate
calibrated physical model, for example a separately generated `tabulated`
prediction with recorded provenance and condition constraints.

The code does not manufacture calibration information from an archived target
histogram. It does not establish that equilibrium beta/Gamma correspond to an
actual freezeout time, that W is conserved during readout evolution, or that a
positive contrast proves quantum dynamics.

## Acquisition and shimming

The legacy sweep stays in its original acquisition order. It is dry-run by
default; `--execute` opts into paid sampling. Each completed raw job is
checkpointed and hashed. For the original sweep the final result is one
streamed `.json.zlib` array per setting plus job metadata. Old `.json.zip`
archives are not changed. Resume verifies settings and completed files.
An ambiguous submission failure leaves a marker requiring inspection before
retry. Orphan archives also stop rather than trigger blind resubmission.

Fresh plans randomize/interleave time, scale, gauge and shim variant within
blocks. Preflight validates every setting before any paid job, without storing
one full coupler map for every request. `shim_update` uses only magnetization
and bond correlations from separate controls. Its flux response sign must be
measured; it is not guessed. NN and NNN classes are updated separately and
retain their means. It proposes an update; it does not autonomously run an
adaptive calibration campaign or choose a stopping threshold from target data.

A physical shim file contains `role: frozen_shim`, `device`, `graph_id`,
`calibration_provenance`, `coupler_deltas` as `[qubit_a,qubit_b,delta_J]`,
`flux_biases` covering the solver's qubit indices, and independently verified
`verified_flux_bias_bounds`. A mapping file passed to `collect --shim-files`
maps keys such as `Advantage_system6:g0:k0.5:s1.0` to those files. Use the exact
numeric rendering in the generated plan (e.g. `s1` if the plan encodes an
integer). Corrections are physical and gauge-specific. The executor checks
NN/NNN class means, current topology, parameter support and available ranges;
it never clips couplers silently.

## Verification

The offline suite checks exact enumeration, flip walks, arbitrary-field/all-pair
identities, asymmetric curvature decompositions, pairwise emulation, the
full-Hamiltonian counterexamples, readout derivatives/channels, independent
calibration, detailed balance, unitary exponentials, quantum-state positivity,
archive formats, gauges, resume behavior and split/fingerprint integrity.
An adversarial test replaces all B outcomes and verifies that A-only fits are
unchanged. The synthetic demo exercises the complete command-line workflow.

Primary implementation references:
[D-Wave parameters](https://docs.dwavequantum.com/en/latest/quantum_research/solver_parameters.html),
[D-Wave susceptibility](https://docs.dwavequantum.com/en/latest/ocean/api_ref_system/generated/dwave.system.temperatures.background_susceptibility_ising.html),
[SciPy exponential action](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.expm_multiply.html).
