# Run log

Append new entries; do not rewrite prior entries. Distinguish code validation, timing calibration, and scientific convergence.

## 2026-08-21 — isolated refactor and local Phase 0 preparation

- Git branch: `codex/mps-mft-phase0-refactor`.
- Scope: all new implementation files are under `ladder_mps_mft/`; pre-existing `plot_ladder_mf_observables.jl` and `analysis/` changes were not modified.
- Julia environment resolved with Julia 1.12.7, ITensors 0.9.15, ITensorMPS 0.3.25, HDF5 0.17.2, and CSV 0.10.15; `Manifest.toml` was generated.
- Unit/integration suite: 57 tests passed. Coverage includes exact E_p selection, geometry kernels, deterministic seeds, on- and off-fixed-point variational constants, period-1 through period-4 handling, mixing, diagnostics primitives, immutable HDF5 restart fields, recursive accepted-only selection, and strict variational branch ranking.
- Local four-site end-to-end SCF smoke: solver, density search, MPO construction, checkpoint write/read, and final artifact path executed. It stopped at `maximum_iterations` by construction and is not scientific evidence.
- Local four-site pairing-field identity smoke: Hamiltonian identity error `-4.440892098500626e-16` and reconstructed/direct variational-energy difference `-4.440892098500626e-16`; the one-sweep effective eigenvalue differed from its final expectation by `-1.0199283164702422e-6`. This motivated using the direct bare-Hamiltonian energy for ranking and retaining the eigenvalue only as a consistency diagnostic.
- Phase 0 plan command passed. The configured worst-case reservation is `1.511718750` Perlmutter CPU node-hours under a `3.0` node-hour cap.
- Tiny Phase 0 script fixture: immutable seed creation, SHA verification, HDF5 MPS reload, two repeated metric solves, TOML output, MaxRSS parsing, numerical gates, and recommendation generation all executed. This L=2 fixture is an integration test only; its timing and energy are not benchmark evidence.
- Tiny diagnostic smoke: charge/spin grids, rung-cut entanglement, diagnostic HDF5, full sign-resolved rung/leg pair matrices, and one fixed-number sector DMRG all executed. The complete six-sector production gap bundle was not run locally.
- Perlmutter jobs submitted: none.
- Validation boundary: local tests validate APIs, algebraic bookkeeping, and serialization. They do not validate Perlmutter performance, production convergence, phase ordering, or physical conclusions.

Next action: sync the branch to Perlmutter, run `bash slurm/phase0_calibrate_cpu.sh plan`, review account/QOS/time/memory settings, then explicitly run `submit` if approved.

## 2026-08-21 — physical period-two correction

- Trigger: review against Bollmark, Kohler, and Kantian, Phys. Rev. B 111, 125141 (2025), Sec. III and Eq. (14), showed that the previous fixed-point-only acceptance rule was physically wrong for the transverse repulsive density channel. The paper identifies a stable two-cycle of the MF iteration as the CDW solution and uses the difference between its two density profiles as the CDW order parameter.
- Replaced mixer-dependent cycle classification with an initial unmixed raw-map probe. The default probes period two for 20 iterations before Anderson mixing can act. Every phase must recur across three links, and every applied field must equal the preceding raw measured field.
- An unmixed orbit is accepted only after phase-resolved density, same-phase energy recurrence, Hamiltonian identity, effective-eigenvalue consistency, and explicit `accepted_periods` gates. A recurrence seen after linear or Anderson mixing is only a `periodic_candidate`.
- Schema v3 stores every accepted orbit phase's MPS, fields, correlations, density, chemical potential, update mode, and energy decomposition. It also stores orbit-averaged canonical energy, phase-energy spread, and the central-bulk density contrast. Phase diagnostics are generated separately.
- Changed the transverse functional away from period one to use the applied partner-phase field. Fixed points are unchanged; periodic energies are averaged across phases rather than across fields.
- Regression suite: 80 tests passed, including all-phase period-2/3/4 recurrence, rejection of a one-phase false positive, raw-versus-Anderson acceptance, explicit Anderson cancellation of an ideal two-cycle, orbit-energy averaging, central density contrast, schema-v3 phase-MPS round trips, invalid-orbit selection rejection, and mixed fixed/periodic ranking.
- Local L=2 DMRG/HDF5 smoke passed through the revised solver and terminated at `maximum_iterations` as configured. It is API validation only, not evidence for a physical orbit.
- Perlmutter jobs submitted: none. The Phase 0 CPU timing payload remains separate from SCF/orbit physics.

## 2026-08-21 — Phase 0 first submission process failure

- Run ID: `20260821_phase0_cpu_v1`.
- Seed job `57392048` failed with exit code `1:0` after two seconds on `nid004121`; all eleven `afterok` benchmark jobs were consequently cancelled, and report job `57392070` failed because no serial baseline existed.
- Root cause: `run.env` attempted to assign `PHASE0_SCRIPT_VERSION` after the worker had declared that variable readonly. Julia never started; neither `metrics/seed.time` nor `seed_state.h5` was created. This is a launcher/process failure and supplies no timing, resource, convergence, or physics evidence.
- Fix: Phase 0 script v1.0.1 persists the submitted version as `PHASE0_RUN_SCRIPT_VERSION`, verifies it against the worker version after loading, and rejects legacy run environments with a clear message. Preserve the failed run directory and use a new run ID after synchronizing this fix.
- Validation: the focused shell environment round trip passed, the guarded plan remains `1.511718750` node-hours, and the full 83-test Julia suite passed.
