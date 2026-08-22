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

## 2026-08-22 — Phase 0 v2 audit and density-targeted correction

- Run ID: `20260821_phase0_cpu_v2`. Seed job `57393191`, all eleven backend
  jobs `57393193`--`57393215`, and report job `57393217` completed with exit
  code `0:0`. The synchronized `sacct.txt` is the resource/accounting source.
- Provenance is internally consistent across all eleven metrics: git commit
  `acc60f1725ce9647a57ca9256d6813e5c73e0d71`, implementation fingerprint
  `8a8920aa75996298b836f6e584bca6946f0f1ce1009212e53c25ab33b084d2fa`,
  config SHA-256
  `105a0e78f48324ee4d01942590d61a234bf63dab4aabc7239546dcaf79317e59`,
  E_p SHA-256
  `2209bd2ca3c1ad02c0e542d1a9d63ecf90fdfa49120ad9cc3af599a5b4bc1f0e`,
  and seed SHA-256
  `b97dc1f3b8e8943e742422d70d84fa05110f7a08e1283c3c7978bae4ca497f29`.
- Every candidate repeated its energy and density exactly at stored precision;
  cross-backend differences are at most approximately `1e-15` in energy per
  site and `1.3e-14` in density. This validates backend equivalence for the
  calculation that actually ran.
- Critical failure: the target density was `0.9375`, but the seed density was
  `0.5614556102812898` and the timing payload density was
  `0.56137567423918...`, an absolute miss of `0.3761243257608...`. The v2
  seed and payload called fixed-`mu=0` DMRG directly and bypassed
  `find_mu_for_density`. Because the anomalous fields conserve total `S_z` and
  fermion-number parity but not full particle number, the initial product-state
  density did not constrain the optimized state.
- Conditional wrong-workload result: `serial-t1` had median `71.666 s`, a
  `0.77%` repeat range, MaxRSS `1.368 GiB`, and a right-sized projection of
  `4 GiB`, two physical cores, and `3.1105e-4` node-hours per solve.
  `blocksparse-t4` was fastest in wall time (`51.318 s`) but `1.43x` the
  projected charge; `strided-t4` had a `24%` timing range. None of this ranking
  is promoted because density-search workloads can require multiple DMRGs.
- Actual v2 charge reconstructed from parent-job elapsed time and allocated
  CPUs is approximately `0.101751302083` node-hours for seed, matrix, and
  report. This is budget evidence, not production performance evidence.
- Decision: reject the v2 recommendation for resource selection and do not
  submit its chi=200 validation. Preserve all v2 artifacts unchanged.
- Correction: Phase 0 script v1.2.0 creates a density-targeted seed and times a
  complete `find_mu_for_density` call from that common seed for every repeat.
  Metric schema v3 stores target errors, converged chemical potentials, search
  statuses, and DMRG evaluation counts. The report now requires target-density,
  repeated search-path, chemical-potential, energy/density, model/config/code/
  seed provenance, exclusive topology, timing-stability, and MaxRSS gates.
  The separate chi=200 validation targets density independently at its own
  converged chemical potential.
- Configuration: Phase 0 density tolerance is `5e-4`; timing density search has
  up to 16 evaluations. The guarded worst-case reservation remains
  `1.511718750` node-hours under the `3.0` cap.
- Validation: Julia syntax parsing, shell syntax, `git diff --check`, and the
  guarded plan passed. The full Julia suite passed all 100 tests, including a
  pairing-seeded end-to-end density search and explicit rejection of a
  tampered wrong-density metric.
- Perlmutter jobs submitted by this correction: none. Next action after pushing
  and pulling the correction is a new immutable
  `20260822_phase0_cpu_v3` plan/submission; only a passing schema-v3 report may
  authorize the one chi=200 validation.

## 2026-08-22 — Phase 0 v3 failure and focused production benchmark

- Run ID `20260822_phase0_cpu_v3`, commit
  `10511f9a2cfe788c4f4c0436f7cad2ed60f1bdb0`, script v1.2.0. Seed job
  `57405642` failed with exit code `1:0` after `22:51`; every dependent
  benchmark was cancelled and the report failed because no serial metric
  existed.
- The seed exhausted 16 chemical-potential evaluations with status
  `maximum_mu_iterations`, ending at density `0.9843323116910832` for target
  `0.9375`. It produced neither `seed_state.h5` nor a candidate metric. Peak
  resident memory was about 1.29 GiB by `/usr/bin/time` and 1.34 GiB by Slurm,
  so this is a numerical search failure rather than memory, walltime, or
  scheduler failure.
- Phase 0's operational priority is the DMRG backend, not replacing the legacy
  density search. The complete v2 matrix is sufficient to shortlist
  `serial-t1` and `blocksparse-t4`; it remains numerical-equivalence evidence
  for its fixed-mu calculation despite the density mismatch.
- Script v1.3.0 now compares only those two finalists using the production-size
  `configs/phase0_validation.toml`: `L=64`, `chi=200`, six sweeps, fixed
  `mu=1.8`, and two repetitions from the same immutable seed. The timed region
  is exactly `run_dmrg_ground`; seed preparation, compilation, MPO construction,
  MPS copying, GC, density measurement, and chemical-potential search are
  excluded. Metric schema v4 records this contract explicitly.
- The default candidate limit is four hours at 32 GiB. The complete guarded
  reservation is `0.570312500` CPU node-hours under the 3.0 cap. The report
  ranks only candidates passing exact provenance/topology, energy/density
  equivalence, MaxRSS, and 10% timing-range gates.
- Pre-run CPU projection from v2, using `chi^2.5--chi^3` scaling and the
  six-versus-two-sweep ratio: `blocksparse-t4` 44--78 min per solve and
  `serial-t1` 62--109 min. These ranges are planning estimates that the focused
  run will replace.
- Legacy-GPU estimate: 35--60 s and `0.00243--0.00417` GPU node-hours for one
  six-sweep `chi=200` fixed-mu solve, extrapolated from the saved `chi=500` and
  `chi=1000` GPU sweep logs. It is not a matched measurement. CPU and GPU
  allocation pools are separate, and the CPU path conserves total `S_z` plus
  fermion-number parity while the legacy GPU path disables both QNs.
- Validation: Julia syntax parsing, shell syntax, `git diff --check`, and the
  guarded plan passed. The full Julia suite passed all 108 tests, including the
  fixed-mu seed/payload/report integration fixture and rejection of a tampered
  timing-region contract.
- Perlmutter jobs submitted by this implementation: none. Next action is the
  staged v4 seed preflight followed by the two-candidate matrix.
