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

## 2026-08-22 — completed-seed dependency compatibility fix

- The focused-run seed job `57426788` completed and produced its immutable
  warm-start artifact. The subsequent staged `submit-matrix` command failed
  before recording any candidate or report jobs with Slurm's `Job dependency
  problem` error.
- Root cause: after verifying the seed's `COMPLETED` accounting state and file,
  script v1.3.0 still attached `afterok:57426788` to each candidate. Slurm can
  reject a dependency on a completed job after that job ages out of the
  controller's active record, even while `sacct` retains its history.
- Script v1.3.1 omits the redundant dependency in the staged path after the
  completed-state and artifact checks. The one-shot path retains `afterok`
  while its seed is pending. No DMRG or scientific calculation changed.
- Existing v1.3.0 run environments are accepted only by v1.3.1. The completed
  seed from commit `38697d803a7a15218cd54b9df1507a41fa76587a` may be reused
  after exact model, config, E_p-registry, seed-file, and recorded seed-lineage
  checks; the new metrics retain both the seed and payload commits and
  implementation hashes.
- Validation: shell/Julia parsing, the guarded plan, and `git diff --check`
  passed. A mock Slurm submission verified that completed-seed candidates have
  no seed dependency while the report retains `afterany` on both candidates.
  The full Julia suite passed all 113 tests, including v1.3.0 run-environment
  compatibility and expanded seed-lineage checks.

## 2026-08-22 — Phase 1 refactored GPU pivot

- Decision: stop CPU production calibration after the observed roughly 100x
  wall-time disadvantage, but retain the refactored scientific workflow. Only
  the site-index representation, MPO/MPS storage device, and DMRG execution are
  moved to dense CUDA; the legacy SCF implementation and self-resubmitting
  wrapper remain unused and unmodified.
- GPU runtime: a weak CUDA extension plus a pinned `gpu/Manifest.toml` resolves
  CUDA.jl 5.9.5 with ITensors 0.9.15 and ITensorMPS 0.3.25. Production GPU
  configs explicitly disable `S_z` and fermion-parity QNs and record this in
  numerical fingerprints and HDF5 provenance. GPU MPS artifacts are copied to
  CPU before HDF5 writes and moved back to GPU on resume.
- Phase 1 design: nine independent pairing/SDW/CDW branches at `L=64`, `U=8`,
  `V=-0.2`, `t0=1.1`, `t_perp=0.1`, `density=0.9375`, and `chi=200`, with the
  refactored unmixed recurrence probe and common-functional acceptance gates.
  A 30-minute GPU smoke/precompile job must complete before the nine 12-hour
  jobs can be submitted. Continuations are explicit and limited to four
  segments per branch by default.
- `E_p`: exact lookup remains preferred. Opt-in interpolation is linear only in
  signed `E_p` between bracketing `t0` rows at identical `(L,U,V,density)`;
  extrapolation and sign-changing brackets are rejected. At the representative
  point it yields `-0.18452309659153343` and
  `t_perp^2/|E_p|=0.05419375777188662`. Mode, endpoints, endpoint chi values,
  weight, registry hash, and effective coupling are saved and fingerprinted.
- Budget: the guarded launcher starts a shared append-only reservation ledger
  at zero additional usage relative to the user-reported 277-node-hour
  baseline. It enforces a hard 400-additional-node-hour ceiling, conservatively
  summing CPU and GPU requested upper bounds without reclaiming early finishes.
  The smoke plus first matrix reserves 27.125 node-hours; four segments for
  every initial branch would reserve 108.125. Optional legacy CPU `E_p` jobs
  are available only through the same ledger at 12 node-hours each.
- Validation boundary: all Julia files parse, the launcher passes `bash -n`,
  the read-only plan reports the expected charges, `git diff --check` passes,
  and the full CPU suite passes all 143 tests including exact/interpolated
  `E_p`, sign/extrapolation rejection, GPU config validation, and unchanged
  CPU DMRG/checkpoint behavior. The SDW seed was corrected from an accidental
  `(0,pi)` leg-only alternation to the intended `(pi,pi)` rung-and-leg Néel
  pattern, with the `(pi,0)` CDW pattern tested separately. The final mocked
  Slurm workflow submitted one smoke plus nine branches, reserved exactly
  `27.125` node-hours, and rejected a `27.1` cap before allocating another job;
  matrix selection and reservation are held under one lock. A forced campaign
  preparation failure allocated zero jobs, and nested final/checkpoint lookup
  passed for status and continuation. CUDA package resolution and extension
  import were checked locally without a device. No Perlmutter job was
  submitted; the staged smoke job remains the required GPU runtime and HDF5
  round-trip proof.

## 2026-08-23 — Phase 1 v1 CUDA-library collision and v2 isolation fix

- Failed run: immutable run ID `20260822_phase1_gpu_v1`, launcher v1.0.0,
  submission commit `09d53c3262ef59e9bccc78240f7daabe7b71770c`. Smoke job
  `57451731` completed, then all nine segment-one branch jobs failed:
  frustrated pairing/SDW/CDW `57452337`/`57452338`/`57452339`, unfrustrated
  pairing/SDW/CDW `57452342`/`57452343`/`57452344`, and square pairing/SDW/CDW
  `57452345`/`57452346`/`57452347`.
- Failure classification: infrastructure/runtime, before a scientific MF
  iteration. Every branch log has the same three CUDA.jl system-library
  warnings followed by signal 11 in the first production DMRG
  eigendecomposition. The stack combines CUDA 13.2
  `/opt/nvidia/hpc_sdk/Linux_x86_64/26.5` `libcublasLt.so.13` with CUDA.jl artifact
  `libcublas.so` and `libcusolver.so`, crashing in `cublasLtLegacyGemmSSS`
  called from `cusolverDnSsyevd`. The v1 launcher explicitly loaded
  `cudatoolkit`, contrary to the selected CUDA.jl artifact-runtime model.
- The completed tiny smoke was a false negative: its log contained the same
  warnings, but the L=2, maxdim=4 DMRG happened not to segfault. Its artifact
  SHA-256 is
  `99bc4fdd8a520e38ef8f0fc3645bf1d6302f8bd7e9da5c8c928d4b68a2f8f49f`;
  it is environment evidence only, not a scientific result. The prepared GPU
  manifest SHA-256 is
  `6e3a094e02d1141e994c8904ea67df930d45c4402a64bc41cb613b55771fdb0d`
  and the branch manifest SHA-256 is
  `18369520757a976da0717bf154266f259999dad57036dc57b2e78c3091710b50`.
- Recoverability audit: `results/` contains zero files across all nine
  branches, so there is no `state.h5`, checkpoint, accepted fixed point, or
  periodic orbit to continue or compare. No phase conclusion can be drawn.
  Preserve v1 unchanged. Launcher v1.0.1 permits read-only status inspection
  of v1 but refuses further submissions from it.
- Correction: launcher v1.0.1 no longer loads `cudatoolkit`; it unloads the
  module, removes inherited NVIDIA-HPC-SDK and local-toolkit paths from
  `LD_LIBRARY_PATH`, and clears CUDA-root variables. The CUDA extension now
  requires the pinned artifact toolkit and treats any loaded non-artifact CUDA
  runtime library as fatal. Both smoke and production entry points run a
  256-by-256 Float32 GPU GEMM plus Hermitian eigendecomposition before DMRG;
  the smoke stores the preflight extrema and dimension.
- Budget: v1 retains its conservative `27.125` requested-node-hour ledger
  reservation; early failures are not reclaimed. A fresh smoke would raise the
  total to `27.250`, and its nine-job matrix would raise it to `54.250`, leaving
  `345.750` under the 400-additional-node-hour project cap. Actual Slurm charge
  is not inferable from the synced artifacts because `sacct` elapsed/TRES rows
  were not included.
- Validation: all Julia sources parse, `bash -n` and `git diff --check` pass,
  and the full local CPU suite passes all 150 tests. This includes mocked Slurm
  staging/budget tests, exact sanitization of the observed HPC-SDK path, and
  unchanged DMRG, density-search, mixing, recurrence, variational-energy,
  checkpoint, and selection coverage. The CUDA extension imports locally
  without a device. No local check can validate GPU execution; the corrected
  Perlmutter smoke is the required runtime proof.
- Perlmutter jobs submitted by this correction: none. Next action after pushing
  and pulling is a fresh immutable `20260823_phase1_gpu_v2` smoke. Inspect its
  log and HDF5 preflight before submitting the nine branches.

## 2026-08-24 — Phase 1 v2 audit and Float64 recurrence recovery

- Run `20260823_phase1_gpu_v2` passed the corrected artifact-runtime smoke job
  `57498404` and completed all nine segment-one branch jobs: frustrated
  pairing/SDW/CDW `57500137`/`57500138`/`57500139`, unfrustrated
  pairing/SDW/CDW `57500140`/`57500141`/`57500142`, and square
  pairing/SDW/CDW `57500143`/`57500144`/`57500145`. Every production log
  recorded the 256-dimensional preflight without the v1 system-library warning.
- Scientific outcome: zero of nine states is accepted. There are three raw-map
  period-two candidates (frustrated pairing/CDW and unfrustrated pairing), five
  mixer-dependent period-two candidates, and one stagnated square SDW state.
  Scheduler `COMPLETED` therefore means process completion, not converged
  physics.
- Precision root cause: all nine saved MPS tensors are Float32. The refactored
  CUDA extension used the opinionated `CUDA.cu` adaptor, which silently converts
  Float64 arrays to Float32. Hamiltonian-identity errors are
  `1.52e-5`--`2.77e-5` per site and effective-eigenvalue errors are
  `1.14e-5`--`3.34e-5` per site, so all nine fail the configured `1e-9` and
  `1e-6` gates. These states must not be rescued by loosening acceptance.
- Recurrence-control root cause: `cycle_action=stop` truncated raw candidates
  at the earliest recurrence and also terminated mixer-dependent candidates
  before the unmixed probe required by their own diagnostic. Frustrated
  pairing/CDW stopped after nine raw updates while their residuals were growing.
  They are transient candidates, not validated physical period-two orbits.
- Screening only: frustrated branches retain `max|alpha|=0.0110`--`0.0112`
  with very weak bulk spin-Hartree variation. The unfrustrated pairing seed
  retains a distinct `max|alpha|=0.0193` state approximately `0.5759` total
  energy above the nearly degenerate non-paired SDW/CDW candidates. All square
  branches have `max|alpha|<5e-6`; its pairing and SDW energies are unresolved,
  while the CDW candidate is approximately `0.0591` total higher than pairing.
  These comparisons are hypotheses only because every state is unaccepted.
- A central-half one-point Fourier screen finds charge weight near
  `|q_x|=pi/8` and spin weight near `q_x=pi+/-pi/16`, `q_y=pi`, in several
  square/unfrustrated branches at hole density `1/16`. This stripe-like relation
  is grid- and seed-dependent and is not a connected structure factor; require
  accepted states, correlation-based peaks, bulk-window checks, and `L`/chi
  scaling before reporting it as physics.
- Resource evidence: stored MF-iteration time sums to `40.406` GPU-hours, about
  `10.102` one-of-four-GPU node-hours before compilation/scheduler overhead.
  Exact actual charge still requires `sacct`; the append-only ledger remains at
  the user-reported `54.25` conservative reserved node-hours.
- Correction: runtime tensor scalar type is explicit, fingerprinted, and stored
  in provenance. Phase 1 configs use Float64. MPS/MPO tensors are converted
  tensor-by-tensor and transferred with NDTensors' type-preserving CUDA adaptor;
  Float32 parent checkpoints are promoted. Both smoke and production preflights
  use Float64, and `submit-matrix` reads the smoke MPS storage and refuses any
  non-Float64 artifact.
- Cycle correction: the initial raw probe now runs its full 20 updates unless a
  solution passes all gates. An exhausted unaccepted initial recurrence is
  archived before Anderson mixing. A mixer-dependent recurrence is archived
  and automatically followed by one fresh raw-map probe; a failed controlled
  probe stops for inspection rather than being damped or re-probed indefinitely.
- Launcher v1.1.0 adds `submit-recovery SOURCE_RUN NEW_RUN`. It selects and
  hashes all nine immutable v2 states, verifies their model fingerprints, and
  records parent status, numerical fingerprint, and Float32 dtype. A local
  end-to-end mock-Slurm submission against all nine synchronized HDF5 states
  produced valid Float64 recovery configs, one `0.125` ledger reservation, and
  one mock smoke job without changing v2.
- Validation: all Julia files parse, `bash -n`, `git diff --check`, and the
  reproducible nine-state audit pass. The full local Julia suite passes all 159
  tests, including tensor-by-tensor Float32-to-Float64 promotion, recurrence
  action policy, Float64 smoke-artifact validation, budget locking, DMRG,
  checkpoint, and branch-selection paths. The local GPU overlay is not
  instantiated, so actual CUDA transfer remains intentionally gated on the new
  Perlmutter smoke.
- Perlmutter jobs submitted by this change: none. Next action is
  `submit-recovery 20260823_phase1_gpu_v2 20260824_phase1_gpu_v3_float64`.
  Its smoke reserves `0.125` node-hours; the nine-job matrix adds `27.0`, taking
  the conservative ledger to `81.375` and leaving `318.625` under the
  400-additional-node-hour cap.

## 2026-08-24 — Complete MF histories, legacy field inheritance, and staged Float64 controls

- Root cause of the missing Phase 1 v2 profile history: `IterationRecord`
  retained the full applied and measured fields in memory, but checkpoint
  schema v4 serialized only scalar history arrays and the terminal field
  snapshots. This was a refactor storage omission; the solver had computed the
  missing data, but v2 cannot reconstruct it after the fact.
- Checkpoint schema v5 now stores the exact segment seed under
  `fields/initial` and both complete maps under `history/fields/applied` and
  `history/fields/measured`. Each component (`alpha`, `beta`, and `mu_cdw`) has
  MF-history index as its final dimension, aligned with `history/iteration`.
  `read_field_history` validates and exposes this contract.
- `plot_phase1_mf_observables.jl` uses the true schema-v5 history by default,
  supports either measured or applied fields, and reads the embedded exact
  seed. Immutable v2 artifacts retain the explicit seed/best/final-orbit
  saved-snapshot fallback; no continuous history is fabricated for them.
- A distinct SHA-guarded `inherit_from`/`inherit_sha256` lineage mode restores
  the legacy field-only behavior. It accepts legacy top-level
  `alpha`/`beta`/`mu_cdw`/`mu` or refactored `fields/restart`, applies the
  legacy zero-`mu_cdw` fallback, warns across transverse geometries, and always
  creates fresh site indices and a fresh product MPS. It is mutually exclusive
  with MPS-reusing `parent_checkpoint` and same-model `resume_checkpoint`.
  `scripts/prepare_field_inherit.jl` validates shapes and generates the pinned
  config without mutating the source artifact.
- Launcher v1.2.0 adds `prepare-recovery SOURCE_RUN NEW_RUN`. It generates and
  validates all nine Float64 recovery controls without Slurm submission or a
  budget reservation; `submit NEW_RUN` then submits only the gated smoke.
  `submit-recovery` remains the one-command equivalent.
- Local validation against the nine synchronized v2 states generated nine
  Float64 configs whose manifest records every Float32 parent and SHA, while
  leaving `jobs.tsv` at its header only. The legacy helper also successfully
  read the actual synchronized unfrustrated legacy state with SHA-256
  `a3a1954517313a1953037f38e21c6b51c91cba2377e5d11a8cfd3c3eb7ce5022`.
- Rendering validation passed for both one immutable v2 saved-snapshot figure
  and temporary schema-v5 complete-history/exact-seed figures. Julia parsing,
  `bash -n`, `git diff --check`, and the full local suite pass; the suite covers
  182 assertions, including schema-v5 history shapes/values, refactored and
  legacy field inheritance, SHA rejection, fresh-MPS initialization, launcher
  staging, DMRG, recurrence, variational energy, and strict selection.
- Perlmutter jobs submitted by this change: none. The conservative project
  ledger therefore remains `54.25` node-hours. Prepare
  `20260824_phase1_gpu_v3_float64_history`, inspect its manifest, and submit its
  `0.125`-node-hour smoke; after the Float64/runtime gates pass, the nine-job
  matrix adds `27.0`, for `81.375` total reserved and `318.625` remaining under
  the 400-node-hour cap.

## 2026-08-24 — Scratch-first full states and automatic stateless mirrors

- Storage regression: the refactor wrote every MPS-bearing Phase 1 state,
  rolling checkpoint, and orbit artifact directly below the CFS checkout. The
  completed v2 campaign alone occupies approximately 3.6 GiB locally. This
  violated both the legacy `copy_data.jl`/`stateless_data` design principle and
  NERSC's production-I/O guidance.
- Launcher v1.3.0 separates campaign control from numerical payload. Full
  branch and guarded `E_p` artifacts are written below
  `$PSCRATCH/MPS-MFT/ladder_mps_mft/phase1_gpu/RUN_ID`; configs, manifests,
  logs, ledger entries, and MPS-free HDF5 mirrors remain on CFS. GPU and CPU
  jobs request the `scratch,cfs` filesystem licenses.
- The recursive compactor omits top-level and orbit-member `psi` groups plus
  pair-binding `psi_N_*` sectors. It preserves complete schema-v5 applied and
  measured MF histories, exact seeds, fields, correlations, energies,
  diagnostics, attributes, and provenance. Each copy records the full path,
  SHA-256, size, omitted paths, and a non-restartable marker; a tree manifest
  records full and compact hashes and sizes.
- Selection, plotting, status, and the campaign audit accept the stateless
  files. Field-only `inherit_from` remains valid because all fields and the
  chemical potential are present. Parent/resume and orbit-MPS readers reject a
  stateless artifact with an explicit pointer to its full source. Continuation
  and recovery resolve full scratch states from `run.env` or
  `full_storage_path.txt`.
- Existing campaign migration is staged and non-destructive until the final
  operator step: NERSC Globus moves a quiescent CFS `results` tree to scratch;
  hashes are checked; `stateless_results` is built and verified; and the old
  CFS path can be retained as a scratch symlink so recorded absolute parent
  paths still resolve. The explicit pending-delete CFS directory is removed
  only after verification. Active v3 files must not be migrated while jobs are
  writing them.
- Scratch is not a backup and files unaccessed for eight weeks can be purged.
  Accepted full states and restart checkpoints that must survive should be
  archived to HPSS; compact CFS/local files suffice for analysis but not DMRG
  restart or MPS-level diagnostics.
- Local validation so far: Julia syntax parsing, `bash -n`, `git diff --check`,
  and a synthetic nested-orbit HDF5 compaction passed; the synthetic copy
  shrank from 20,496 to 8,360 bytes while retaining non-MPS data. Full suite
  status is recorded below after completion. Perlmutter jobs submitted by this
  storage change: none.
- Full local validation completed with all 206 assertions passing. The 59
  checkpoint/selection assertions include recursive orbit-MPS removal,
  schema-v5 field-history preservation, field-only inheritance from a compact
  state, `psi_N_*` removal, compact-tree manifests, and explicit rejection of
  stateless checkpoint/orbit-MPS reads.
- A real v2 square-SDW `state.h5` also passed the standalone full/compact hash
  verifier. Omitting its MPS reduced the file from 80,675,908 bytes (about
  77 MiB) to 1,637,619 bytes (about 1.6 MiB), while retaining the analysis
  datasets. This is a representative state-file reduction, not a projection of
  the total schema-v5 campaign size.
- After adding worker-only compatibility for already-queued launcher-v1.2 jobs,
  the complete suite was rerun and all 207 assertions passed. A queued v3 job
  may therefore start from launcher v1.3.0 without being rejected; it retains
  its prepared v1.2 CFS output path and does not adopt scratch storage midway
  through the immutable campaign.
- Added `slurm/migrate_phase1_to_scratch.sh` as the single guarded operator
  command for completed pre-v1.3 campaigns. It fixes the observed NERSC helper
  failure by creating mode-700 `~/.globus` before authentication, submits and
  waits for the documented `dtn`-to-`perlmutter` Globus transfer, checks every
  scratch file against a quiescent-source SHA-256 inventory, builds and verifies
  the stateless CFS mirror, and installs the compatibility symlink. Destructive
  CFS cleanup requires the explicit `--prune-cfs` flag and occurs only after all
  transfer and mirror gates pass.
- Validation after adding the one-command migrator: `bash -n` and
  `git diff --check` pass; the complete local Julia suite passes all 212
  assertions. No Globus transfer, Perlmutter job, or CFS deletion was performed
  by this local validation.
- The first live v2 migration submitted Globus task
  `d2d6989e-a050-11f1-b669-0afff7074b21`, but both NERSC status helpers returned
  empty output and the polling loop could not advance. The transfer itself is
  Globus-managed and was not cancelled. The migrator now supports
  `--resume-transfer TRANSFER_ID`, prefers the current plural status helper, and
  independently completes its gate only when the scratch file count and every
  SHA-256 match the quiescent CFS inventory; blank helper output is reported as
  `UNKNOWN` rather than as empty lines.
- Operator correction: CFS and Perlmutter scratch are both mounted and this
  campaign is only a few GiB, so Globus added unnecessary authentication and
  status-helper failure modes. The migrator now uses a direct `cp -a` into a
  scratch staging directory, verifies exact file count and SHA-256, and only
  then removes the original through the existing explicit cleanup gate. This is
  the safe cross-filesystem equivalent of `mv`, whose implementation would also
  be copy-then-delete. The already-submitted v2 Globus task is detected from its
  log and reused only if its scratch tree passes the same verification; no
  concurrent local copy is started.
- Validation after the direct-copy correction: `bash -n`, the help path, and
  `git diff --check` pass; the complete local Julia suite passes all 213
  assertions. No local validation moved or deleted any campaign artifact.
- The next live compaction exposed a second pre-existing truncated v2 artifact,
  `frustrated__cdw_s1/.../orbit_period_02_iter_0010.h5`, after the first bad
  `checkpoint_latest.h5` was removed. Hash equality had correctly established
  byte-for-byte CFS/scratch agreement but could not establish HDF5 readability.
  The migrator now offers explicit `--prune-corrupt-auxiliary`: it scans all
  HDF5 files before hashing, removes unreadable checkpoints/orbit snapshots and
  named `.corrupt-*` backups from both trees, and aborts without cleanup if any
  final `state.h5` is unreadable. Failed compaction staging directories are now
  removed automatically.
- Validation: the cleaner removed synthetic corrupt checkpoint/backup files
  from paired roots and preserved both roots when a synthetic final `state.h5`
  was corrupt. Julia parsing, `bash -n`, `git diff --check`, and the complete
  local suite pass; the suite contains 216 assertions. No campaign artifact was
  changed by local validation.

## 2026-08-25: Float64-history audit and cross-device handoff

- The completed v2 and v3 migrations are represented locally by compact,
  MPS-free `stateless_results` trees and `full_storage_path.txt` pointers to
  `/pscratch/sd/k/kwang98/MPS-MFT/ladder_mps_mft/phase1_gpu/RUN_ID`.
- Compact-only verification passed for all 42 v2 artifacts and all 50 v3
  artifacts. The manifests represent 3,512,377,436 and 7,385,759,171 full bytes
  respectively; the compact payloads are 113,990,381 and 666,136,203 bytes.
  These local checks verified compact hashes, sizes, stateless markers,
  full-artifact hash links, and MPS removal. They did not verify the full
  scratch sources because Perlmutter scratch is not mounted on the Mac.
- A fresh v3 audit generated at `2026-08-25T20:04:31.004` UTC found eight of
  nine states accepted, one raw-map period-two candidate, zero mixer-dependent
  candidates, and zero Hamiltonian-identity/effective-energy gate failures.
  Stored MF-iteration time totals 17.518 GPU-hours, or 4.379 one-of-four-GPU
  node-hours before scheduler, compilation, and non-iteration overhead.
- The accepted-state comparator authorized only same-geometry rankings. For
  cubic frustrated, CDW is below SDW and SC by 0.007368263458 and
  0.008447412380 total. For cubic unfrustrated, accepted CDW is below accepted
  SDW by 0.000312604756; the pairing state is excluded because it remains an
  unaccepted raw-map period-two candidate. For square, SC is below SDW and CDW
  by 0.001045316109 and 0.058892533574 total. These are finite-system seed
  comparisons at one point, not thermodynamic or cross-geometry phase claims.
- `docs/PHASE1_V3_AUDIT.md` records the reproducible numerical result and next
  scientific gates. `docs/DEVICE_HANDOFF_2026-08-25.md` separates GitHub code
  and context from ignored numerical data, gives a deterministic lightweight
  archive/Globus transfer path, and records the exact continuation order.
  `docs/NEW_DEVICE_CHAT_PROMPT.md` is the copy-ready first prompt for a new
  Codex desktop task.
- This handoff made documentation-only tracked changes. It did not submit or
  cancel a Perlmutter job, alter the project budget ledger, migrate/prune an
  artifact, modify an immutable state, or verify full scratch files.

## 2026-08-25: Phase-resolved chi=400 recurrence submission preparation

- The transferred lightweight payload passed fresh compact-only verification:
  42 v2 artifacts (`3,512,377,436` represented full bytes; `113,990,381`
  compact bytes) and 50 v3 artifacts (`7,385,759,171` represented full bytes;
  `666,136,203` compact bytes). Full scratch files were not mounted or checked.
- A fresh v3 audit at `2026-08-25T20:22:58.857` UTC reproduced eight accepted
  states and the one unfrustrated-pairing raw-map candidate in
  `audit-win-nextprep-20260825`. No energy comparison changed.
- The candidate has already executed an initial 20-update raw probe, one linear
  step, seven Anderson steps, and a second 20-update raw probe. In the final
  raw segment its period-two energy difference grows to `2.671975013e-3`
  total and its two-step measured-field residual grows to `1.225459946e-3`.
  The final chi=200 DMRG discarded-weight proxy is about `9.05e-6`. Another
  identical chi=200 raw extension was therefore rejected as redundant.
- The next control is three matched `cubic_unfrustrated` pairing branches at
  the same model point: v3 cycle members `001` and `002` as separate full-MPS
  parents, plus independent seed `pairing_s2`. All use chi=400, 16 sweeps,
  cutoff `1e-11`, DMRG energy tolerance `1e-9`, density tolerance `1e-4`, and
  exactly one 20-update raw period-one/two probe. `cycle_action=stop` and
  `max_iterations=probe_iterations+1` prevent Anderson from entering.
- `parent_orbit_phase` now selects `cycle_members/NNN` from one immutable full
  parent while recording the phase index in provenance. The phase MPS is the
  warm start and that phase's measured field is the next raw field. The v3 full
  source is hash-pinned as
  `ed6381ea7c3e2e654e1600566c827729d1bde894fa4f9a708543dc6627ac6df4`;
  preparation rechecks this full SHA, full/compact status and fingerprints,
  Float64 scalar type, raw phase lineage, and both phase MPS groups on
  Perlmutter.
- Launcher v1.4.0 adds read-only `plan-recurrence` and preparation-only
  `prepare-recurrence SOURCE_RUN NEW_RUN`. It generates exactly three configs
  and refuses missing/unmounted/hash-mismatched full parents. Submission remains
  the separate smoke/status/matrix sequence.
- The plan reserves `9.125` node-hours: `0.125` smoke plus three 12-hour,
  one-of-four-GPU segments at `3.0` each. The synced ledger snapshot is
  `87.375` reserved (SHA-256
  `b2757e54c732c5f0364c7f2f1eb6510f91b6dd72fe29d66040ed18c78dce8080`),
  projecting `96.500` reserved and `303.500` below the project cap. The fresh
  live Perlmutter ledger and accounting remain authoritative before submission.
- Local validation passed compact verification, the reproduced campaign audit,
  `bash -n`, and all 242 Julia assertions, including full orbit-phase parent
  loading, stateless rejection, SHA propagation, three-branch preparation, and
  no-ledger-write preparation. No Slurm job was submitted, no ledger row was
  changed, and no immutable or full HDF5 artifact was modified.

## 2026-08-26: Seed-aware conditional chi=400 staging

- The accidental standard campaign named literal `RUN_ID` was retained and
  audited rather than treated as the intended recurrence run. Its independent
  unfrustrated pairing seed collapsed to a nearly zero-pairing accepted fixed
  point, the accepted SDW seed was slightly lower within that campaign, and the
  CDW seed stagnated. Its implementation fingerprint differs from v3, so no
  cross-run energy ranking was authorized.
- The chi=400 successor is now two separately prepared campaigns. Stage A is
  unchanged: v3 orbit phases `001` and `002` remain separate full-MPS parents,
  accompanied by independent `pairing_s2`, with a 20-update raw probe and no
  Anderson entry. Its smoke plus first segments reserve at most `9.125`
  node-hours.
- Conditional Stage B is unavailable until the Stage A stateless results and
  hash-linked full scratch artifacts are present and verified. At least one
  phase-parent lineage and the independent `pairing_s2` lineage must each be an
  accepted pairing-bearing solution with `max|alpha| >= 1e-4`; every phase of
  an accepted orbit must clear the floor separately. The gate also requires
  cubic-unfrustrated geometry, Float64, the current implementation, and exact
  model, numerical, implementation, and `E_p`-registry fingerprints. It writes
  the source paths and hashes to `conditional_gate.tsv`.
- If and only if that gate passes,
  `prepare-recurrence-competitors RECURRENCE_RUN NEW_RUN` creates independent
  `sdw_s2` and `cdw_s2` chi=400 controls with random seeds `1203` and `1304`.
  They share Stage A's numerical fingerprint and raw recurrence policy. Their
  80-iteration execution ceiling allows Anderson only after a recurrence-free
  20-update raw probe; an unaccepted raw recurrence still stops and remains
  phase-resolved. Stage B's smoke plus first segments reserve at most `6.125`
  node-hours.
- The combined first-segment envelope is `15.250` node-hours. Against the
  synced `114.500` ledger snapshot, Stage A projects `123.625` reserved and
  `276.375` unreserved; both stages project `129.750` reserved and `270.250`
  unreserved under the 400-node-hour project cap. Their combined four-segment
  emergency ceiling is `60.250`, but no continuation is pre-authorized. The
  unused allowance is intentionally retained for higher-bond-dimension and
  scaling calculations. Live Perlmutter accounting remains authoritative.
- Launcher v1.5.0 separates preparation from submission. `submit RUN_ID` now
  requires an existing prepared campaign and cannot silently create the
  standard nine-branch matrix. `prepare-standard` is explicit, and literal
  placeholders including `RUN_ID` are rejected for submission actions.
- Local validation passed Julia syntax parsing, `bash -n`, the read-only
  `plan-recurrence` ledger calculation, and all 263 Julia assertions. The 62
  Phase 1 launcher assertions cover the conditional two-lineage gate,
  phase-by-phase orbit fields, full-artifact hashes, fingerprint equality,
  two-branch preparation, no-ledger-write preparation, placeholder rejection,
  no implicit campaign creation, and the hard-cap rejection path. No Slurm job
  was submitted or cancelled, no ledger row changed, no campaign was prepared
  against Perlmutter scratch, and no immutable HDF5 artifact was modified.

## 2026-08-27: chi=400 Stage A audit and stateless-transfer pruning control

- Compact-only verification passed all 14 Stage A branch artifacts. Their
  manifests represent `11,826,906,884` full bytes and `163,304,236` compact
  bytes. Full scratch files were not mounted or verified on Windows.
- A fresh audit in `audit-win-stagea-20260827` found zero accepted states, two
  raw-map period-two candidates, and one `time_limit` result. Both phase-parent
  candidates remain pairing-bearing at `max|alpha|` about `0.0187`, pass
  density and Hamiltonian/effective-energy consistency, and fail only the
  variational-energy recurrence gate with `dE/site=3.171e-5` and `3.370e-5`.
  Their phases remain separate and unranked.
- The independent `pairing_s2` branch reached nine raw records. Its relative
  residual fell to `7.877e-3` at record 6 and then grew to `3.923e-1` by the
  time limit, so its destination basin is unresolved. The conditional Stage B
  gate fails and the SDW/CDW controls must not be prepared.
- The next exact calculation is one explicit segment-002 continuation of only
  `unfrustrated__pairing_s2_chi400`. Its plan-only requested cost is `3.000`
  node-hours, projecting the synced ledger from `123.625` to `126.625` and
  leaving `273.375` under the 400-additional-node-hour cap. No further segment
  is pre-authorized.
- `scripts/prune_phase1_stateless_extras.py` adds a dry-run-first cleanup for
  redundant compact checkpoints and orbit snapshots while retaining final
  states, diagnostics, summaries, configs, and logs. The current four-run
  dry-run projects compact payload reduction from `1715.023` to `501.972` MiB,
  saving `1213.051` MiB. Applying on Perlmutter requires full-source hash
  verification; applying to an unmounted workstation mirror requires an
  explicit local-only boundary. No real campaign file was pruned in this
  audit.
- The cleanup utility is Python and documentation-only. The scientific
  implementation fingerprint remains
  `bf67d865fcb44e339dc44994fc11a0c703056a0356f3edf6f7432987251130f4`.
  A temporary-copy apply test passed the Julia stateless verifier both before
  and after pruning, and the complete local Julia suite passed all 263
  assertions after Git Bash was added to the Windows test process path. No job
  was submitted or cancelled, no ledger row was changed, and no immutable HDF5
  file was overwritten.

## 2026-08-27: stateless-pruner Python 3.6 compatibility correction

- The first Perlmutter invocation stopped at parse time because the default
  `python3` does not recognize `from __future__ import annotations`. Parse-time
  failure occurred before argument handling, verification, manifest backup, or
  deletion, so no campaign file could have changed.
- The optional future import and Python 3.7+/3.9+ annotation dependencies were
  removed. The utility now uses ordinary classes and `typing` generics that are
  compatible with Python 3.6.
- Python 3.6 grammar validation and a complete four-campaign dry run pass. A
  temporary-copy apply test also passed the Julia stateless verifier before and
  after pruning. The corrected script SHA-256 is
  `a9e16279a1af279360fefcfe404dc7b411ae35fe22a345898f104a6798dcc449`.
- No real campaign artifact was pruned, no full scratch artifact was modified,
  no job was submitted or cancelled, and no ledger row changed.

## 2026-08-28: local spatial phase-defect audit

- Added the read-only `scripts/audit_spatial_phase_defects.py` diagnostic and
  six synthetic Python tests. The audit reads compact schema-v5 applied and
  measured field histories, constructs charge/spin leg-parity and pairing
  form-factor profiles, measures a Hann-weighted central-75% finite-interval
  spectrum, demodulates the dominant wavevector, and identifies diagnostic
  amplitude-zero/phase-jump coincidences above an absolute `1e-6` signal floor.
  The resolved second spectral peak excludes the two bins adjacent to the
  primary peak.
- The spatial residual definition respects the recurrence contract. Ordinary
  fixed-point searches use the rung-resolved raw link `f(x)-x`; period-two
  candidates use the same-phase two-step field change `x_m-x_(m-2)`, because a
  one-step difference contains the physical orbit-phase contrast. These maps
  are diagnostic only and do not replace literal raw-map acceptance or cycle
  gates.
- Ran the audit locally on all three chi=400 Stage A states and all nine v3
  Float64-history states. The ignored output is
  `output/phase1_gpu/20260826_phase1_unfrustrated_pairing_recurrence_chi400/spatial-defect-audit-20260828`;
  it contains a Markdown report, 12 figures, source SHA-256 inventory, state
  and channel summaries, spectral histories, phase-slip candidates, and
  residual-component histories.
- Neither phase-parent branch has a final charge, spin, or pairing phase-slip
  candidate. Their final same-phase period-two changes are `1.891583e-3` and
  `1.928906e-3`, both peaked at rung 51 and about `90.46%`/`90.50%` Hartree by
  squared field residual. The hotspot is stationary through nearly the entire
  raw history rather than traversing the ladder.
- The independent `pairing_s2` final raw-link residual is `0.392273`, peaks at
  rung 49 after peaking at rung 62 in records 7--8, and is `95.49%` Hartree by
  squared residual. Its only resolved phase-slip diagnostic is in the charge
  envelope, moving from rung 15 at record 8 to rung 12 at record 9; the final
  charge candidate is 37 rungs from the residual peak. No spin or pairing
  phase-slip candidate is resolved. These histories therefore do not currently
  support a moving domain wall as the primary cause of Stage A nonconvergence;
  open-boundary and multi-wavevector alternatives remain diagnostic questions.
- Validation: all six Python tests pass, `git diff --check` passes, and the
  complete Julia suite passes all 263 assertions when launched from a Git Bash
  login environment. Git emitted sandbox-user safe-directory warnings in
  provenance subprocesses, but the associated 68 checkpoint/selection
  assertions passed. Every HDF5 input was opened read-only and compact-SHA
  hashed. Full Perlmutter scratch artifacts were not mounted or verified.
- This audit consumed zero node-hours. It submitted or cancelled no job,
  changed no ledger row, wrote no HDF5 file, changed no acceptance or energy
  ranking, and made no thermodynamic-phase claim.

## 2026-08-28: opt-in matched-mode independent seeding

- Added an opt-in `matched_mode` independent-seed protocol while preserving
  the historical random-pairing/staggered-Hartree path as the default
  `legacy` protocol. Exact regression tests confirm that explicitly selecting
  `legacy` reproduces the prior pairing, SDW, and CDW field arrays.
- A matched seed uses one declared finite-ladder cosine mode and phase,
  mean-controls nonzero modes, maps the profile into pairing, SDW, or CDW, and
  normalizes the complete stored field vector to
  `norm(alpha,beta,mu_cdw)/sqrt(2L) = initial_amplitude`. Pairing templates are
  explicit `onsite_s`, `rung_s`, `leg_s`, `extended_s`, or `d_wave`; Hartree
  leg parity is explicit or resolves to SDW-odd/CDW-even. Uniform leg-even CDW
  is rejected as redundant with chemical-potential targeting.
- Matched branch preparation assigns a common product-state random seed across
  SC, SDW, and CDW within a geometry. The guarded standard, recurrence, and
  conditional-control manifests record the seed protocol and a dedicated
  initial-seed fingerprint. Seed choices deliberately remain outside the
  numerical fingerprint so independently seeded converged states can still
  pass the established same-model/numerics/code/E_p comparison gates.
- `scripts/inspect_initial_seed.jl` provides a no-DMRG, lightweight TSV preview
  containing charge/spin leg-parity profiles and pairing-form-factor proxies.
  `docs/SEEDING.md` gives the exact formula and emphasizes that one matched
  mode controls roughness and source norm but does not remove wavevector or
  form-factor selection; a predeclared mode/phase bank is still required for
  basin-accessibility claims.
- Final local validation passed all 348 Julia assertions, including the legacy
  regression, matched-channel norm and structure, config/provenance round
  trips, lightweight inspection, matched branch generation, production
  common-seed preparation, recurrence manifests, and all existing Slurm,
  ledger, recurrence, variational, checkpoint, and strict-selection guards.
  All six spatial-audit Python tests also pass. The final scientific
  implementation fingerprint is
  `edb4d230260000b89def6b61c5a0ee861eaa8ed34464e937c7865e9c8de87593`.
- This implementation and validation were local and consumed zero node-hours.
  No job was prepared on Perlmutter, submitted, continued, or cancelled; no
  ledger row or immutable HDF5 artifact changed. Against the last synced
  `123.625` reserved snapshot, a three-branch first-segment pilot remains a
  plan-only `9.125` node-hour envelope (`132.750` projected reserved and
  `267.250` unreserved), subject to authoritative live accounting before any
  preparation or submission.

## 2026-08-28: matched-seed chi=400 pilot submission preparation

- Added the locked `configs/phase1_gpu_matched_seed_pilot_chi400.toml` and
  `scripts/prepare_phase1_matched_seed_pilot.jl`. The three independent
  cubic-unfrustrated branches use one common product-state random seed (`1404`),
  field norm `1e-3`, phase `0`, and respectively pairing mode `0` with
  `d_wave` form factor, SDW mode `58` with odd leg parity, and CDW mode `11`
  with even leg parity. They share the model and numerical fingerprints but
  retain distinct seed fingerprints.
- Every branch uses chi `400`, 16 sweeps, cutoff `1e-11`, DMRG energy tolerance
  `1e-9`, and exactly 20 unmixed raw-map updates. `max_iterations=21` and
  `cycle_action=stop` prohibit Anderson entry. The modes are a targeted
  convergence control based on observed finite-run profiles, not an unbiased
  wavevector survey or a thermodynamic-phase claim.
- Launcher v1.6.0 adds read-only `plan-matched-seed-pilot` and preparation-only
  `prepare-matched-seed-pilot NEW_RUN`. Preparation creates exactly three
  configs plus a seed-resolved manifest, requires independent starts, writes
  full MPS paths below scratch and stateless destinations below CFS, and does
  not call Slurm or modify the ledger. Smoke and matrix submission remain
  separate guarded actions.
- The first-segment plan is `0.125 + 3*3 = 9.125` requested node-hours; the
  four-segment `36.125` ceiling is informational and no continuation is
  pre-authorized. The synced ledger still ends at `123.625`, projecting
  `132.750` reserved and `267.250` unreserved. The user reports a queued
  segment-002 `pairing_s2` continuation that is absent from this synced
  `jobs.tsv` and ledger. If the live guarded ledger contains its expected
  `3.000` reservation, cancellation does not reclaim it and the authoritative
  projection is instead `135.750` reserved with `264.250` unreserved.
- Local validation passes shell syntax, `git diff --check`, all 375 Julia
  assertions (including 89 launcher assertions), and all six spatial-audit
  Python tests. The synthetic launcher test verifies exact branch settings,
  preparation without a ledger write, the `9.125` calculation, smoke gating,
  atomic matrix reservation, and hard-cap rejection.
- The resulting scientific implementation fingerprint is
  `6156bb036632935c691b0b88e2e372a92db91949012a5d74096984eccd067197`.
- No real campaign directory was prepared on Perlmutter, no job was submitted
  or cancelled, no ledger row changed, and no HDF5 artifact was written or
  overwritten. Live Perlmutter job state and accounting must be reconciled
  before syncing the scientific implementation or submitting the new smoke.

## 2026-08-29: matched-seed chi=400 pilot result audit

- Audited the locally synced
  `20260828_phase1_unfrustrated_matched_seed_chi400` campaign. The pairing,
  SDW, and CDW segment-001 jobs each consumed about `11.49` recorded branch
  wall-hours and ended at the configured `41,400`-second internal solver
  deadline, before the 12-hour Slurm limit, with `status=time_limit`,
  `accepted=false`, and `fundamental_period=0`. They completed 12, 9, and 12
  mean-field updates, respectively; none completed the configured 20-update
  raw recurrence probe.
- Pairing reached its minimum raw-map relative residual
  `5.847670840e-3` at update 6, then expanded and ended at
  `1.432878371e-1`. CDW similarly reached `7.456334661e-3` at update 6 and
  ended at `1.232033467e-1`. Their final density searches were interrupted at
  absolute density errors `2.665144918e-2` and `8.460002088e-3`, so the final
  records are diagnostics rather than candidate solutions.
- Matched SDW followed a different trajectory: after nonmonotonic early
  updates it contracted to `1.130148161e-3` at update 8 and ended at
  `1.176246562e-3` at update 9. Its final density error is
  `2.242561055e-4`. This is close enough to justify a continuation, but it is
  not accepted and does not yet distinguish a period-one fixed point from a
  longer raw recurrence.
- The closest pairing seed contrast is the prior chi=400 independent
  broadband `pairing_s2` state. At each of the first nine matched update
  indices the new residual is no larger. Its minimum is `25.76%` lower
  (`5.848e-3` versus `7.877e-3`), while the update-9 residual is only `6.95%`
  lower (`0.3650` versus `0.3923`). Both histories turn upward after update 6.
  The fairer seed therefore improves early efficiency modestly but does not
  resolve pairing convergence. This is one trajectory per protocol and the
  implementation fingerprints differ (`1d75bb1735b9...` versus
  `9e34457163a0...`), so it is not a replicated causal estimate or an energy
  comparison.
- Re-ran the recurrence-aware spatial audit over the matched campaign, the
  prior chi=400 recurrence campaign, and v3. Matched pairing and matched SDW
  have no resolved final phase-slip candidates. The clean CDW mode develops
  three persistent spin-envelope candidates near rungs 14, 34, and 52;
  candidate coverage is `0.83`, and the nearest is three rungs from the final
  residual peak. Broadband seed disorder is therefore not the sole source of
  spatial defects: the raw dynamics can nucleate or retain them from a single
  smooth input mode. The heuristic still cannot distinguish a mobile wall
  from open-boundary or multi-wavevector beating without continuation.
- All three new terminal profiles are spin-dominant by the one-point field
  diagnostic. Pairing and SDW end with dominant spin wavevector
  `q/pi=0.920635`; CDW ends at `0.952381`. They qualitatively approach the
  spin-rich unfrustrated manifold seen in prior runs, but are not demonstrated
  to be one fixed point because none is accepted and their residual/defect
  structures differ. One-point fields are not thermodynamic order parameters.
- Regenerated the established six mean-field figures below
  `output/phase1_gpu/20260828_phase1_unfrustrated_matched_seed_chi400/plots/mf_profiles`
  and created read-only comparison outputs below `analysis/matched_seed_comparison`
  and `analysis/spatial_defect_audit_matched_vs_prior`. The reusable extractor
  is `scripts/compare_phase1_matched_seed.py`; it deliberately omits energy
  ranking. No new state is eligible for canonical variational comparison, and
  no energy is compared across numerical or implementation fingerprints.
- Compact-only stateless verification passed independently for all three
  branch mirrors: four manifest artifacts per branch, matching compact hashes
  and sizes, explicit stateless markers, linked full hashes, and no MPS
  tensors. Combined compact bytes are `86,338,383`; recorded full scratch
  bytes are `5,617,104,580`. The full Perlmutter artifacts are not mounted and
  were not hash-verified locally (`full_artifacts_verified=false`).
- The synced conservative ledger now sums to `135.750` reserved node-hours and
  `264.250` unreserved under the 400-additional-node-hour cap. The recommended
  next calculation is one matched-SDW continuation segment under the same
  raw-map/no-Anderson policy, plan-only `3.000` node-hours, projecting
  `138.750` reserved and `261.250` unreserved subject to authoritative
  Perlmutter rechecking. This preserves the larger reserve for later bond-
  dimension and length convergence.
- This audit submitted or cancelled no job, changed no ledger row, modified no
  immutable HDF5 artifact, and made no thermodynamic-phase claim.

## 2026-08-29: chi=400 density-cost and SDW-node pairing audit

- Added the read-only local analyzer
  `scripts/analyze_phase1_density_and_node_lock.py`. It parses saved Slurm
  logs and compact schema-v5 histories, selects the last density-converged
  measurement rather than an interrupted terminal search, and writes bounded
  TSV outputs below the matched campaign's
  `analysis/density_and_node_lock` directory. It does not modify HDF5,
  scheduler state, or the budget ledger.
- The three matched chi=400 first segments performed much more nested work
  than their 9--12 displayed outer updates suggest: `233` separate
  density-targeted DMRG solves and `1,871` printed DMRG sweeps. Relative to
  the earlier unfrustrated independent chi=200 trio, logged DMRG sweep time
  per completed outer update is `9.99` times larger on aggregate. The
  chi=400 campaign simultaneously changes maxdim `200 -> 400`, the sweep cap
  `12 -> 16`, DMRG energy tolerance `1e-8 -> 1e-9`, and density tolerance
  `2e-4 -> 1e-4`, while retaining the same `41,400`-second internal deadline.
- The measured multiplication is two-stage: matched chi=400 used `7.06`
  chemical-potential evaluations per outer update versus `3.05` at chi=200,
  and each chemical-potential evaluation used about `4.3` times as much
  logged DMRG sweep time on aggregate. The current safeguarded search starts
  a fixed `0.05` chemical-potential bracket step after any initial density
  miss. Individual trial `(mu,density,sweeps,time)` values are not logged, so
  the saved evidence cannot yet separate fixed-step overshoot from a density
  plateau or noisy finite-DMRG compressibility.
- At the last density-valid records, matched SDW is at update 8 with relative
  field residual `1.130148161e-3`: its field and density gates pass, but its
  canonical variational-energy change is `1.225969556e-4` per site, above the
  `1e-7` gate. Pairing and CDW are selected at update 11 and fail both field
  and energy gates. All three therefore require additional raw-map evidence;
  none is accepted.
- Defined a descriptive node-lock statistic as the central-bulk Pearson
  correlation between `abs(pair_d)` and the negative demodulated
  `spin_odd`-envelope magnitude. At the last density-valid measurement,
  matched pairing and SDW give `0.923` and `0.942` with best lag zero; their
  pairing amplitude is respectively about `19.1` and `66.4` times larger in
  the lowest SDW-envelope quartile than in the highest. Matched CDW is a
  counterexample at `-0.106` and enrichment `0.85`.
- The node locking emerges under the unmixed raw map: matched SDW reaches
  `0.919` by update 3 and stays near `0.94`, while matched pairing rises from
  `0.210` at update 8 to `0.778` at update 9 and `0.937` at update 10.
  Anderson therefore did not create this texture. The v3 unfrustrated pairing
  candidate and both chi=400 phase-parent candidates also give approximately
  `0.91`, whereas accepted v3 unfrustrated SDW/CDW states have no resolved
  d-wave field. The original legacy spatial artifacts were not reprocessed in
  this scoped audit.
- The pattern is consistent with an amplitude-modulated, approximately
  constant-phase d-wave anomalous field concentrated near SDW envelope nodes.
  It is not by itself a sign-changing pair-density wave, a connected pairing
  correlator, or a thermodynamic d-wave-order claim. Open boundaries,
  finite-length beating, incomplete raw-map convergence, and moving domain
  walls remain alternative explanations.
- Validated and rendered the report `Why the chi=400 Runs Produced Few Outer
  Iterations—and What the SDW-Node Pairing Texture Means` with native runtime,
  convergence-gate, and node-lock figures. The latest synced ledger remains
  `135.750` reserved and `264.250` unreserved. A same-code matched-SDW
  continuation remains the first compute priority at a plan-only `3.000`
  node-hours; detailed per-mu logging and any adaptive-bracket change should
  be validated separately before changing the implementation fingerprint of
  a scientific continuation.
- No job was submitted or cancelled, no ledger row changed, no immutable HDF5
  file was modified, and no cross-geometry energy comparison was made.

## 2026-08-30: exploratory square seed/basin campaign preparation

- Audited two newly synced, independently initialized legacy square artifacts
  at `L=64,U=8,t0=1.4,t_perp=0.1,density=0.9375,chi=200`. Their SHA-256
  hashes are `761a14d5248507abc4bc7092f3960302dc915866ce5c602c77bfe787a3c05be`
  for `V=-0.2` and
  `3100916863023c01ead6ae3edd77beda60a7a2c1e2b9991a2d1dad27ee7b75b0`
  for `V=-0.4`. Neither records inherited, parent, or resume lineage.
- The legacy `V=-0.2` and `V=-0.4` files contain only four and three outer MF
  updates, end at absolute density errors `1.143e-3` and `1.226e-3`, and have
  stored d-wave proxies of magnitude `0.0795` and `0.0865`. Correlation-based
  dominant d-wave magnitudes are `0.0840` and `0.0885`. This supports an
  initialization-sensitive basin hypothesis but does not establish refactored
  convergence, a canonical energy ranking, or thermodynamic d-wave order.
- Added the locked exploratory base
  `configs/phase1_gpu_square_seed_pilot_chi200_loose.toml` and preparer
  `scripts/prepare_phase1_square_seed_pilot.jl` for the representative
  `t0=1.4,V=-0.4` point. The three independent starts use a common field norm
  `1e-3`, phase zero, and product-state random seed `1404`: d-wave pairing mode
  `0`, odd-leg SDW mode `51`, and even-leg CDW mode `5`. These post-hoc modes
  are targeted basin reconnaissance, not an unbiased wave-vector bank.
- The exploratory fingerprint uses chi `200`, 12 sweeps, cutoff `1e-10`, DMRG
  energy tolerance `1e-6`, inner and outer density tolerances `1e-3`, initial
  chemical-potential bracket step `0.01`, growth factor `3`, variational-energy
  tolerance `1e-6`, and at most 80 MF updates. The initial `mu=0.55` is shared
  without importing legacy fields or MPS data. The exact registry row
  `E_p=-0.24962435880865996` is mandatory and interpolation is disabled.
- The first 20 updates remain the unmixed physical map. An accepted raw
  period-one/two orbit can terminate there; otherwise an unaccepted raw
  recurrence is archived separately before Anderson acceleration, and a
  mixer-dependent recurrence receives its own fresh raw-map probe. Anderson
  fixed-point acceleration does not redefine the raw-map orbit physics.
- Launcher v1.7.0 adds read-only `plan-square-seed-pilot` and preparation-only
  `prepare-square-seed-pilot NEW_RUN`. Preparation emits three configs with
  one model fingerprint, one numerical fingerprint, three distinct seed
  fingerprints, full MPS destinations below scratch, and stateless CFS
  destinations. It refuses lineage, interpolation, unsafe run IDs, and
  overwrite of an existing run; it neither submits nor reserves.
- The live synced plan uses the `135.750` reserved / `264.250` unreserved
  ledger snapshot. One smoke plus three first segments costs a conservative
  `9.125` requested node-hours, projecting `144.875` reserved and `255.125`
  unreserved. Nine separately gated three-branch first segments for the later
  square `t0={1.0,1.2,1.4}` by `V={0,-0.2,-0.4}` grid would cost a plan-only
  `82.125`, project to `217.875`, and leave `182.125`. The other eight points
  are not prepared or authorized, and no continuation is pre-authorized.
- Any energy comparison is limited to accepted states sharing the square
  geometry and complete model/numerical/implementation/`E_p` fingerprints,
  evaluated with the canonical variational functional including
  double-counting terms. Loose-chi results are preliminary and cannot be
  ranked against legacy energies, another grid point, another transverse
  geometry, or tighter/higher-chi states. The accuracy ladder retains later
  tighter chi `200`, chi `400`, chi `800`, and length controls.
- Direct synthetic preparation passed, Bash syntax passed, the read-only plan
  reproduced every cost, and all 429 Julia assertions passed, including 143
  guarded Phase 1 launcher assertions. `git diff --check` is run separately at
  handoff. Detailed rationale and commands are in
  `docs/SQUARE_SEED_AND_GRID_PLAN_2026-08-30.md`.
- The resulting local scientific implementation fingerprint is
  `6052d87b4b821ffd8b6b16ee68434f1d029a033638d3d1cb8992f3fee715bb8e`.
- No Slurm job was submitted or cancelled, no ledger row changed, no campaign
  directory was prepared on Perlmutter, and no HDF5 artifact was modified or
  overwritten. Perlmutter scheduler state and accounting remain authoritative.

## 2026-08-30: six-branch harmonic-stripe and legacy-like seed revision

- This append supersedes the three-branch square plan above without rewriting
  its historical record. The supplied clean legacy square profile motivated a
  combined SDW/CDW construction rather than independent SDW and CDW sources.
  At `L=64`, the primary slow signed spin-envelope mode is `m=4`, giving the
  antiferromagnetic spin mode `n_s=59` and locked charge second harmonic
  `n_c=8`. The adjacent predeclared control is `m=5`, giving `(58,10)`.
- Added `stripe` and `stripe_pairing` matched seeds. Both use odd transverse
  spin parity, even charge parity, phase zero, and separately normalized
  `charge:spin=0.2`; the mixed branches add uniform d-wave pairing at
  `pairing:spin=1`. Pure stripe remains an `alpha=0` symmetry-subspace control,
  while stripe+pairing allows coexistence without forcing pairing to survive.
- Read-only inspection of the actual legacy fresh-run block corrected the
  recollection that both alpha and beta began randomly. It set `beta=0` and
  `mu_cdw=0`; for alpha it drew one Gaussian coefficient per relative rung
  offset and leg-pair class and copied it along all rungs. Added the separately
  labeled `legacy_pairing` matched seed with that center-of-mass-constant
  structure, a dedicated reproducible field RNG stream keyed by `1404`, and
  the common total norm `1e-3`. It is structurally legacy-like but deliberately
  does not reproduce the legacy per-coefficient amplitude convention.
- The representative square bank now has six independent starts: uniform
  d-wave, legacy-like mixed-relative-bond pairing, pure stripes `m=4,5`, and
  stripe+d-wave starts `m=4,5`. All share one product-state seed, model
  fingerprint `e93552440b67e1070d005f2b5ed7307fe8a0cc97b707b76916dce39aa0bc0482`,
  and numerical fingerprint
  `15a04c35cd48f193ecf95e305cd8cff5bb014030a9cbe27148c36836b730c6ce`;
  all six seed fingerprints are distinct.
- Lightweight construction checks reproduced total field norm `1e-3` for all
  six seeds. The legacy-like branch has `max|alpha|=5.03369689215e-4` and exact
  zero beta and Hartree fields. The pure stripe branches have exact zero alpha;
  the mixed branches have nonzero alpha and the declared stripe harmonics.
- Launcher v1.9.0 prepares six configs but still submits nothing during
  preparation. From the synced `135.750` reserved ledger, one smoke plus six
  first segments has a plan-only envelope `18.125`, projecting `153.875`
  reserved and `246.125` unreserved. The conditional representative-six plus
  eight later three-branch grid envelope is `91.125`, projecting `226.875`
  reserved and leaving `173.125`. Repeating all six branches at all nine points
  would cost `163.125` and leave only `101.125`; it is not recommended because
  higher-chi and length convergence remain mandatory.
- Direct synthetic preparation produced six branches, the read-only launcher
  plan reproduced every cost, `bash -n` passed, `git diff --check` reported no
  whitespace errors, and the complete local suite passed all `513` assertions
  (`177` configuration/seed and `173` guarded-launcher assertions included).
  The resulting implementation fingerprint is
  `8e19b55d67a1460938b1070d0e35a9d2075fe147409070a77d125b6ec663df15`.
- No Slurm job was submitted or cancelled, no ledger row changed, no
  Perlmutter campaign directory was prepared, no immutable HDF5 artifact was
  modified, no cross-geometry energy comparison was made, and no
  thermodynamic-phase claim was made. Live Perlmutter accounting remains the
  submission authority.

## 2026-08-31: square seed/basin pilot compact audit and figures

- Audited the synced campaign
  `20260830_phase1_square_t014_vm04_seed_chi200_loose`, submitted from commit
  `30b9a0df407afaf266c26f35b09bfe5ec962615a`. All six branches ended as
  accepted period-one fixed points after six stored MF records
  (`initial:1,unmixed_probe:5`). No Anderson update was used, no raw-map
  period-two candidate was detected, and no branch timed out.
- Ran compact-only stateless verification separately on all six result
  directories. All 30 manifest rows passed. The local compact artifacts total
  `122306340` bytes (`116.64 MiB`) versus `2935558986` recorded full bytes
  (`2.734 GiB`); the compact HDF5 files omit `psi`. The full scratch artifacts
  were not mounted or rehashed locally, so their manifest paths, sizes, and
  SHA-256 values remain provenance rather than a local full-artifact check.
- The six accepted states share model fingerprint
  `e93552440b67e1070d005f2b5ed7307fe8a0cc97b707b76916dce39aa0bc0482`,
  numerical fingerprint
  `15a04c35cd48f193ecf95e305cd8cff5bb014030a9cbe27148c36836b730c6ce`,
  implementation fingerprint
  `e56feef54bf8bf619f2d531af4e474c5532404ca49d2532aa606011f44242ca5`,
  Float64 tensors, and the exact registered
  `E_p=-0.24962435880865996` with source SHA-256
  `2209bd2ca3c1ad02c0e542d1a9d63ecf90fdfa49120ad9cc3af599a5b4bc1f0e`.
- The canonical selector therefore authorizes this within-campaign numerical
  order: legacy-like pairing `-149.4625379781576`; stripe `m=4`
  `-149.4625274080429`; stripe `m=5` `-149.4612599174197`; stripe+d `m=4`
  `-149.4600424777624`; stripe+d `m=5` `-149.4593696755100`; d-wave seed
  `-149.4593503931971`. The first two differ by only `8.258e-8` per physical
  site, below the campaign's `1e-6`-per-site stabilization scale. That scale is
  a stopping gate, not an error bar, and this loose chi-200 ordering is not a
  thermodynamic preference.
- Final relative raw-map residuals span `2.105e-5`--`2.054e-3`; final density
  errors span `4.504e-4`--`4.959e-4`. Each branch used the identical density
  search work sequence `3,6,1,1,1,1`, or 13 Hamiltonian evaluations total.
  Once the chemical potential was bracketed on update two, each later outer
  update reused it with one evaluation. The six stored branch wall times sum
  to `2.197` GPU-hours, equivalent to `0.549` one-of-four node-hours before
  scheduler/compile overhead; Perlmutter accounting remains authoritative.
- Cross-branch profiles show strong basin collapse at the pilot tolerance.
  Every start selects the same centered even-charge profile with dominant
  `q/pi=8/63`; the maximum relative RMS difference from the d-wave-seed charge
  profile is `1.065e-3`. After aligning the arbitrary global pairing sign, the
  d-wave proxy profiles differ by at most `1.267e-4` in relative RMS and the
  selected pairing channel is uniform `q=0`. The remaining seed memory is a
  weak odd-leg spin field: the stripe families retain `q/pi=59/63` (`m=4`) or
  `58/63` (`m=5`) with RMS `2.46e-5`--`3.99e-5`; pairing-only controls are
  below the spatial-audit `1e-6` signal floor.
- The spatial phase-defect audit found no branch satisfying its combined
  phase-slip persistence and residual co-localization heuristic. These short
  histories therefore do not support a moving domain wall as the primary
  convergence limitation, while open-boundary and multi-q caveats remain.
  The terminal stripe residuals are still `1.3e-3`--`2.1e-3`, so their
  acceptance under the deliberately loose `5e-3` relative gate should not be
  mistaken for tight scientific convergence.
- Relative to the separately audited legacy `V=-0.4,t0=1.4,chi=200` file, the
  refactored pilot stores six rather than three outer updates and reaches the
  density gate rather than ending at error `1.226e-3`. Legacy energy and field
  amplitude are excluded from the comparison because they do not share the
  canonical refactored functional and normalization/provenance contract.
- Generated the established six per-branch field/history figures and six seed
  figures below `analysis/mf_observables_20260831`, the six spatial diagnostic
  figures and tables below `analysis/spatial_phase_defects_20260831`, and the
  cross-branch convergence, terminal-profile, energy/channel figures plus
  ranked TSVs and report below `analysis/square_seed_comparison_20260831`.
- The recommended next calculation is a two-parent, common-fingerprint tighter
  chi-200 continuation of the legacy-like pairing minimum and near-degenerate
  stripe `m=4` state: 16 sweeps, density and inner-mu tolerance `1e-4`, cutoff
  `1e-11`, DMRG energy tolerance `1e-9`, variational-energy tolerance `1e-7`,
  relative field tolerance `1e-4`, and a 20-update raw-map-only probe before
  any Anderson acceleration. Its plan-only first-segment envelope is
  `0.125 + 2*3 = 6.125` requested node-hours, projecting `160.000` reserved
  and `240.000` unreserved. It was not prepared or submitted.
- The synced conservative ledger now contains `153.875` reserved additional
  node-hours and `246.125` unreserved under the 400-node-hour hard cap; this
  campaign contributed `18.125` requested node-hours. This audit submitted or
  cancelled no job, changed no ledger row, and modified no HDF5 artifact.

## 2026-08-31: six-parent square tight-five campaign prepared in code

- Reclassified the six loose square energies as scientifically unresolved.
  Their complete spread is `2.490e-5` per physical site, while the leading
  fixed-density interpolation scale `|mu| |n-n_target|` is approximately
  `2.5e-4` per site at the observed density errors. The exact stored numerical
  ordering remains provenance, but the supported result is qualitative basin
  collapse toward very similar charge and nonzero d-wave-pairing profiles.
- Added `docs/PHASE1_NUMERICAL_ERROR_BUDGET.md`. Stopping tolerances are not
  treated as error bars. The documented envelope separates density mismatch,
  recent SCF energy drift, Hamiltonian/effective-energy identities, a common
  frozen-correlation small-field scan, E_p sensitivity, DMRG/chi control, and
  finite-size control. Correlated systematic scales are not combined as an
  unjustified root-sum-square statistical error.
- Kept the physical mean-field map unthresholded. Hard zeroing near a floor can
  create zero/nonzero chatter and changes the numerical map. The new manifest
  instead declares post-processing floors `0,1e-6,1e-5,1e-4`; plots may use a
  declared floor and energy sensitivity must recompute the canonical
  transverse pair, exchange, density, and double-counting terms against stored
  correlations. The scan is not a new self-consistent solution.
- Added `scripts/prepare_phase1_square_tight5.jl` and launcher actions
  `plan-square-tight5` / `prepare-square-tight5`. Preparation requires all six
  accepted period-one compact states, resolves their six immutable full scratch
  parents, rehashes the full files, checks Float64/model/numerical/
  implementation/E_p provenance agreement, and writes fresh parent lineages.
  A stateless analysis copy can never be used as the restart parent.
- The common new contract retains `L=64`, square geometry, `U=8`, `V=-0.4`,
  `t0=1.4`, `t_perp=0.1`, density `0.9375`, and `chi=200`; it raises the DMRG
  work to 16 sweeps with cutoff `1e-11` and energy tolerance `1e-9`, tightens
  inner/outer density tolerances to `1e-4`, uses field gates `1e-7` absolute or
  `1e-4` relative, and uses the `1e-7`-per-site energy-change gate. Every
  branch has at most five new raw-map MF evaluations and cannot enter Anderson.
  Five records cannot validate period two under the eight-record complete-
  history contract; a two-cycle-looking result remains unresolved.
- The launcher version is now `1.10.0`. Tight-five branches request one of four
  GPUs for `03:00:00`, or `0.75` node-hours each. The plan-only envelope is
  `0.125 + 6*0.75 = 4.625` node-hours. From the synced `153.875` ledger it
  projects to `158.500` reserved and `241.500` unreserved, preserving most of
  the hard-cap allowance for higher bond dimension and length convergence.
  Live Perlmutter accounting remains authoritative.
- The complete local Julia suite passed `544/544` tests, including `204/204`
  guarded Phase 1 launcher tests with six synthetic hash-linked full/compact
  accepted parents. Bash syntax and the local plan also passed. No Slurm job
  was submitted or cancelled, no Perlmutter run directory was prepared, no
  ledger row changed, and no HDF5 artifact was modified. Actual preparation
  must occur on Perlmutter after syncing this code because scratch is not
  mounted locally.

## 2026-09-01: six-parent square tight-five compact audit

- Audited the synced campaign
  `output/phase1_gpu/20260831_phase1_square_t014_vm04_chi200_tight5` at code
  commit `4370a46d47acb874965fb8fdbe4eda6a0c26d5b7`. The common contract is
  square, `L=64`, `U=8`, `V=-0.4`, `t0=1.4`, `t_perp=0.1`, density
  `0.9375`, `chi=200`, Float64, at most five additional raw-map updates, and
  no possible Anderson entry. The DMRG, density, field, and energy settings
  match the 2026-08-31 preparation entry.
- Ran `scripts/verify_stateless_results.jl` without `--full` on all six result
  roots. All 24 compact artifacts passed marker, compact SHA-256, compact-size,
  recorded full-size, and recorded full-SHA metadata checks; no MPS tensors
  remain in the mirrors. The compact copies total `107,910,655` bytes
  (`102.912 MiB`) while their recorded immutable full artifacts total
  `2,921,150,977` bytes (`2.7205 GiB`). Scratch is not mounted locally, so
  `full_artifacts_verified=false`; the authoritative full-file hashes and
  Perlmutter accounting were not independently remeasured on this computer.
- Reproduced the campaign audit below `analysis/audit_20260901`. All six
  branches ended `maximum_iterations`, not a Slurm time limit: each contains
  one inherited initial record plus four new unmixed-probe records. There are
  `0/6` accepted states, zero raw-map candidates, zero mixer-dependent
  candidates, no validated recurrence period, and no stored canonical
  solution energy. Hamiltonian-identity and effective-energy checks pass for
  every record.
- Every terminal density error passes the `1e-4` gate (`6.35e-5`--`7.76e-5`).
  The d-wave and legacy-like pairing branches pass the final and trailing-two
  field gate, but all branches miss the `1e-7`-per-site energy-change gate by
  factors `2.50`--`2.79`. The four stripe-family branches also miss the
  relative-field gate by factors `1.06`--`1.78`. Both residuals and energy
  changes decrease monotonically over the available child records.
- Density fixing required five or six Hamiltonian evaluations only on the
  first child update and one on each later update, for nine or ten evaluations
  per branch. It is therefore not the limiting cause of the five-record stop.
  Stored branch wall time sums to `2.593` GPU-hours, or approximately `0.648`
  one-of-four GPU node-hours before scheduler and compilation effects; the
  conservative ledger continues to count requested ceilings rather than
  reclaiming early completion.
- The spatial audit below `analysis/spatial_phase_defects_20260901` finds a
  common terminal charge mode `q/pi=8/63` with RMS approximately `4.751e-4`
  and a common selected d-wave `q=0` component with RMS approximately
  `2.775e-3`. Across all six terminal states, the maximum pairwise relative
  RMS difference is only `0.03193%` for charge and `0.00412%` for d-wave.
  Thus the six initial seeds have collapsed to the same resolved charge and
  pairing profiles to substantially better precision than the stopping gates.
- The weak stripe-seed odd-leg spin component falls by factors `11.5`--`12.4`
  from parent to child, to RMS `1.98e-6`--`3.46e-6`; pairing controls are at
  `2e-8`--`4e-8`. With the declared derived-profile floor `1e-5`, every
  terminal spin profile is identically zero while charge and d-wave are
  unchanged. The remaining stripe residual is `97.8%`--`98.4%` Hartree-field
  residual and is consistent with decay of a numerically negligible seeded
  spin texture, not evidence for a distinct resolved SDW state.
- No branch satisfies the spatial audit's combined moving-phase-slip and
  residual-co-localization heuristic, so these data do not support a moving
  domain wall as the primary convergence limitation. Five complete records
  remain insufficient for the canonical period-two test, which requires
  eight; any apparent future two-cycle remains unresolved until that contract
  is met.
- The six child terminal current-iterate energies span `7.717e-6` per site,
  versus `2.490e-5` per site among the accepted loose parents. This reduced
  spread is useful evidence of basin collapse only. The child states are
  unaccepted and store no solution energy, and the parent and child numerical
  fingerprints differ, so no variational energy ranking is authorized either
  among the children or between parent and child. No thermodynamic order or
  phase is claimed from one-point finite-ladder fields.
- Generated the established 12 per-branch figures below
  `analysis/mf_observables_20260901`, spatial diagnostics below
  `analysis/spatial_phase_defects_20260901`, and reproducible cross-campaign
  tables, threshold scans, figures, and report below
  `analysis/tight5_comparison_20260901_final`.
- The exact recommended next calculation is one further same-six-parent
  raw-map segment with at most five additional MF updates and no Anderson.
  Its plan-only requested ceiling is `6*0.75 = 4.500` additional node-hours
  without repeating the completed smoke. From the synced ledger value of
  `158.500` reserved, this would project to `163.000` reserved and `237.000`
  unreserved under the 400-node-hour project hard cap, subject to an
  authoritative live Perlmutter recheck before submission. This keeps most of
  the allowance for later bond-dimension and length convergence.
- This audit submitted or cancelled no job, changed no budget-ledger row,
  migrated/pruned/deleted no data, and modified no HDF5 artifact. Only compact
  stateless content was inspected locally; full-artifact verification,
  authoritative charging, higher-chi/length convergence, and thermodynamic
  scientific conclusions remain outside this audit boundary.

## 2026-09-01: numerical re-audit, classifier repair, and square V=0 campaign preparation

- Before changing the classifier, added and ran the read-only
  `scripts/audit_scf_numerics.py` over all 45 local Phase 1 final-state paths.
  Thirty-six complete-history artifacts were auditable; nine v2 paths lack
  `history/fields/applied` in their local compact files and therefore remain
  explicitly unclassifiable under the new test. Fourteen auditable paths
  change classification. The tabular evidence and error list are under
  `analysis/numerics_reaudit_pre_fix_20260901`; no HDF5 field was edited.
- Five v3 stored fixed points fail the new slow-mode-extrapolated residual
  gate. The v3 unfrustrated-pairing period-two candidate and both Stage A
  phase-parent candidates have consecutive-step cosine approximately
  `+0.9999` and two-step/one-step ratio approximately `2`, so the repaired
  classifier calls them monotone drift rather than oscillatory period two.
  The two Stage A states were already unaccepted; this audit does not promote
  or rank them.
- Period-two recurrence now additionally requires step-vector cosine at most
  `-0.5` and two-step/one-step ratio at most `0.5`. Fixed-point acceptance
  estimates `lambda=dot(r_k,r_(k-1))/||r_(k-1)||^2` for aligned residuals and
  gates the extrapolated residual `r/(1-lambda)`; `lambda>=1` fails.
- The density solver now carries a positive `dn/dmu` estimate between SCF
  updates, tries a bounded Newton predictor before safeguarded bracketing, and
  uses configurable `1e-8,1e-9,0` noise for warm-started mu re-solves. The
  first solve for a changed MF Hamiltonian retains the normal DMRG schedule.
- The DMRG observer now stops on absolute sweep-energy change only after the
  final max-dimension/noise schedule is reached and records per-sweep energy,
  maximum discarded weight, and maximum link dimension. State schema v6
  stores this evidence per MF update; sector-gap schema v2 stores it per
  fixed-N/fixed-Sz sector. A real four-site DMRG smoke produced three finite
  sweep energies, nonzero discarded weights, and realized link dimensions
  `[4,4,4]`.
- Energy stabilization and authorized ranking now use the stored
  target-density correction `E + mu*(N_target-N)` while preserving the raw
  canonical energy and full double-counting decomposition. An older artifact
  must contain enough data to reconstruct this correction or is excluded from
  ranking.
- The ranking implementation fingerprint now hashes only `src/**/*.jl` plus
  the active CPU or GPU Manifest. Requested walltime and output verbosity are
  omitted from the numerical fingerprint. The broader source/config/launcher/
  test hash remains separately recorded as `tree_sha256`. This prevents a
  launcher-only edit from blocking future same-solver campaign comparison.
- Added the exact-registry square `V=0,t0=1.4` six-seed loose chi-200 config,
  generalized the square preparer across the locked `V=-0.4` and `V=0` points,
  and added launcher v1.11.0 actions `plan-square-v0-seed-pilot` and
  `prepare-square-v0-seed-pilot`. The six starts are uniform d-wave,
  legacy-like pairing, stripe `m=4,5`, and stripe+d-wave `m=4,5`; all share
  amplitude `1e-3`, phase zero, and product-state seed `1404`.
- The plan-only first-segment envelope is `0.125 + 6*3 = 18.125` node-hours.
  From the synced conservative ledger of `158.500`, submission would project
  to `176.625` reserved and `223.375` unreserved under the 400-additional-
  node-hour cap. The four-segment `72.125` ceiling is not pre-authorized, which
  preserves capacity for later bond-dimension and length convergence.
- The full local Julia suite passed `619/619` assertions, including the real
  DMRG observer callback, synthetic sector-gap schema v2 evidence, numerical
  classifier counterexamples, solver/tree fingerprint separation, and
  preparation of the V=0 six-branch campaign with an unchanged test ledger.
  Bash syntax and Python audit help also passed.
- No MPO optimization was added because the supplied evidence says MPO
  construction is not the bottleneck. The proposed `sacct` reconciliation,
  `hbm80g` constraint change, and CPU-pool routing were accounting/scheduler
  policy rather than solver numerics and were deliberately deferred from that
  numerical change.
- No Slurm job was submitted or cancelled, no Perlmutter run directory was
  prepared, no budget row changed, no data were migrated/pruned/deleted, and
  no immutable HDF5 artifact was overwritten. Perlmutter accounting and full
  scratch verification remain authoritative; higher-chi/length convergence
  and thermodynamic phase claims remain outside this preparation.

## 2026-09-01: elapsed-time budget reconciliation and GPU constraint policy

- Launcher v1.12.0 keeps the original requested reservations immutable and
  adds an independent append-only reconciliation ledger keyed by Slurm job ID.
  The explicit `reconcile [RUN_ID]` action reads finalized allocation rows from
  Perlmutter `sacct`, records `ElapsedRaw`, state/start/end, the effective node
  fraction, measured node-hours, and released ceiling, and is idempotent on
  repeated use. Pending/running or missing jobs retain their full requested
  reservation.
- The active 400-node-hour project total is now original requested ceilings
  minus recorded terminal-job releases. Present one-of-four-GPU and 64-of-128-
  CPU jobs both carry an effective fractional rate of `0.25` node-hours per
  elapsed wall hour. The measured amount is capped at its original reservation;
  Perlmutter scheduler and allocation accounting remain authoritative.
- The smoke and all branch configs below chi `1200` now request `gpu` instead
  of `gpu&hbm80g`; configs at chi `1200` and above retain the 80-GB-HBM guard.
  The selected threshold is frozen in each new run environment.
- The complete local Julia suite passed `633/633` assertions. Mock Slurm tests
  verified low-chi and high-chi constraint selection, exact elapsed-charge
  arithmetic, byte-for-byte preservation of the reservation ledger, active-cap
  reporting, and job-ID idempotence on repeated reconciliation.
- No real reconciliation was run locally, no existing reservation row changed,
  and no job was submitted or cancelled. CPU-pool routing beyond the already
  guarded `E_p` path remains a separate future scheduler change.

## 2026-09-01: explicit time-zero MF seed history

- State schema v7 now stores the exact initial field inside the history at
  `history/fields/seed` with `seed_iteration=0`, in addition to the established
  `fields/initial` copy. Applied and measured update arrays retain their
  existing one-through-N alignment; density, energy, residual, and DMRG data
  still start at update 1 because no such measurements exist for the seed.
- `read_field_history(...; include_seed=true)` and complete-history plots can
  prepend time zero. The plotting adapter now honors its existing
  `include_seed` option for schema-v5-and-newer histories and falls back to
  `fields/initial` for schema-v5/v6 files, so already-synced artifacts also gain
  a seed point when replotted without rewriting them.
- The complete local Julia suite passed `642/642` assertions, including schema
  v7 seed identity, time-zero reader alignment, and stateless-mirror retention.

## 2026-09-01: direct Phase 1 scientific submission

- At the user's explicit request, launcher v1.13.0 retires the standalone GPU
  smoke as a public submission gate. `submit RUN_ID` now atomically checks the
  project allowance and submits every still-pending scientific branch directly;
  `submit-matrix RUN_ID` is a backward-compatible alias for the same action.
- Every scientific branch retains the artifact-runtime isolation and dense
  GPU linear-algebra preflight before entering SCF, so removing the separate
  smoke does not remove the branch-level CUDA guard.
- Direct initial envelopes no longer include a `0.125` smoke reservation:
  standard nine-branch `27.000`, three-branch `9.000`, six-branch `18.000`,
  tight-five `4.500`, and recurrence Stage A plus conditional Stage B `15.000`
  node-hours. Continuations remain explicit and are not pre-authorized.
- The already-prepared square `V=0,t0=1.4` v1.12.0 campaign is intentionally
  direct-submission compatible with v1.13.0. An existing smoke row does not
  block or duplicate any branch; its prior reservation remains append-only
  until that terminal job is processed through `reconcile RUN_ID`.
- Bash syntax passed and the complete local Julia suite passed `642/642`
  assertions. The launcher test directly submitted a v1.12.0-prepared mock
  campaign, recorded exactly nine branch reservations with no smoke, enforced
  the hard cap, and reconciled finalized elapsed charges idempotently.
- No real job was submitted or cancelled and no Perlmutter ledger was changed
  locally. Live `sacct` measurements and Perlmutter accounting remain
  authoritative.

## 2026-09-01: direct-submission launcher hotfix

- The first real v1.13.0 direct attempt exposed two launcher regressions before
  any scientific branch was submitted: the branch selector assumed `config`
  was manifest column six, but the current square V=0 manifest places it after
  the newly recorded point coordinates; `status` also still called the
  accidentally removed `slurm_state` helper.
- Launcher v1.13.1 resolves manifest fields by the named `label` and `config`
  headers, validates every pending config and its DMRG max dimension before
  allocating any branch, and restores the read-only Slurm status helper.
  V1.12.0- and v1.13.0-prepared campaigns remain direct-submission compatible.
- Regression coverage now parses the actual extended square V=0 manifest and
  executes launcher `status` after mock direct submission. The failed real
  attempt recorded no branch job or branch reservation; the user's existing
  smoke remains a separate historical row subject to terminal reconciliation.
- Bash syntax passed and the complete local Julia suite passed `646/646`
  assertions, including `256/256` guarded Phase 1 launcher assertions.

## 2026-09-01: isolated-ladder backbone and hybrid Stage 1 pilot

- Added the six-sector, number- and `S_z`-conserving isolated-ladder backbone
  for the first `L=64`, `U=8`, `n=0.9375`, `V=0`, `t0=1.4` point. The site
  indices explicitly carry `Nf`, redundant `NfParity`, and `Sz`; a tested
  `removeqn(psi, "Nf")` transition therefore preserves parity and `Sz` for a
  later pairing-field response calculation.
- Each sector uses spatially distributed holes, a 15-sweep `chi<=200`
  pre-relaxation, persistent noise at or above `1e-8`, and warm starts through
  `chi=400,800,1200`. Absolute energy stopping begins only after the maximum
  bond dimension has been held for the configured minimum sweeps. Immutable
  MPS checkpoints are written after every completed stage and a missing final
  sector automatically resumes from the newest exact-config/exact-code stage.
- Sector and assembled schema v2 artifacts retain per-sweep energies,
  discarded weights, realized link dimensions, last-five-sweep spreads, every
  final MPS, and spin/charge/pair-binding estimates at every chi. Assembly
  streams one MPS at a time, rejects mixed configuration or implementation
  hashes, and marks whether all final sectors pass the scientific gates.
- Implemented Stage 1 of the hybrid search only. It diagonalizes complete
  connected charge and spin covariance matrices in both leg-parity sectors and
  complete real-space pairing covariance matrices within onsite, rung, and
  both leg-bond classes. Pair addition and removal Gram matrices are summed for
  the Hermitian `Delta+Delta^dagger` source. Cross-class pairing mixing is
  deliberately deferred to Stage 2 candidate orthonormalization.
- Replaced the original independent-MPO pairing loop with a cached MPS transfer
  sweep. On the tiny state, every optimized rung and leg matrix agreed with the
  direct fermionic MPO definition to `1e-10`; the Hermitian field covariances
  were positive semidefinite to numerical precision.
- The tiny six-sector/HDF5/Stage-1 integration run completed. Checkpoint-only
  recovery recreated a deliberately removed final sector without another DMRG
  stage. The Stage-1 artifact passed its PSD gate with minimum eigenvalue
  `-1.700029006457271e-16`; this tiny result is software evidence only, not a
  physical ladder conclusion.
- The finalized Windows Julia suite passed `442` assertions. Two Bash-only
  launcher executions were reported as platform skips; source-level launcher
  assertions passed. A separately discovered Git-for-Windows Bash then passed
  `bash -n`, the bare-pilot plan, and the required Phase 0 CPU plan. The bare
  plan reserves at most `17.015625` CPU node-hours under its `24.0` cap.
  Perlmutter-side validation and the real Stage-1 submission remain pending
  because this Windows session had no SSH key and non-interactive NERSC
  authentication was rejected. No real Slurm job was submitted, no allocation
  ledger changed, and Stage 2 was not started.

## 2026-09-01: Perlmutter operator-boundary correction

- The user clarified the permanent host boundary: Codex must never authenticate
  to or log in to NERSC/Perlmutter, transfer files to or from it, or operate its
  scheduler. The user always performs synchronization, live accounting/status
  checks, submission, continuation, and cancellation.
- Added this rule at the repository root and repeated it in the ladder
  subproject. Perlmutter commands in documentation are now explicitly labeled
  as user-run handoff commands. Codex prepares and validates locally and
  analyzes only artifacts that the user has synchronized back.
- The pending local SSH authentication prompt was terminated without login.
  No Codex in-app NERSC browser tab remained open. The attempted connection in
  the preceding entry was outside the intended workflow and must not be
  repeated.

## 2026-09-01: square V=0 CUDA preference failure and durable isolation fix

- Perlmutter jobs `57842419` and `57842420` from
  `20260901_phase1_square_t014_v000_seed_chi200_loose` failed before SCF in the
  branch-level dense CUDA preflight. Job `57842419` used `00:05:58`, mostly for
  first-use precompilation, and job `57842420` reused that cache and failed in
  `00:00:44`. Their logs report both a Cray `libmpi_gtl_cuda.so` preload missing
  `libcudart.so.13` and CUDA.jl selecting a local toolkit with no discoverable
  runtime. Neither job produced scientific MF iterations or a state eligible
  for analysis.
- The successful 2026-08-30 square six-seed campaign used the same CUDA 5.9.5,
  CUDA runtime wrapper, GPU manifest, and artifact-isolation function. The
  relevant launcher-policy change was the requested relaxation from
  `gpu&hbm80g` to `gpu` below chi 1200. The failed logs additionally demonstrate
  that a local-toolkit preference was effective at package precompile time even
  though no uncommented setting exists in the repository or the user's searched
  `~/.julia` preference files. The printed files under `~/.julia/packages/CUDA`
  are package documentation templates, not active preferences.
- Made the established artifact-only policy explicit in `gpu/Project.toml`:
  `CUDA_Runtime_jll.local = "false"`. Added the runtime wrapper as an extra so
  Julia's preference loader treats the setting as belonging to the active GPU
  environment and gives it precedence over higher load-path environments.
- The same project now selects artifact `MPICH_jll` and an empty MPI preload
  list. HDF5 is used only from a single Julia process in Phase 1; GPU-aware Cray
  MPI is not part of the solver and must not pull system CUDA runtime libraries
  into the artifact-only process.
- The four remaining submitted branches must stay held until this project file
  is synchronized and a one-allocation preflight succeeds. No Codex action was
  taken on Perlmutter: no job was submitted, held, released, requeued, or
  cancelled, no ledger row changed, and no HDF5 artifact was modified. Terminal
  elapsed-time reconciliation remains authoritative on Perlmutter.

## 2026-09-02: square V=0 CUDA artifact-version failure and explicit 13.0 pin

- Audited the locally synchronized rerun
  `20260901_phase1_square_t014_v000_seed_chi200_loose_cudafix`, commit
  `f92c15deb0afafb5ef3da74c6854262fc6177b98`. All six jobs (`57852352` through
  `57852357`) failed before the dense CUDA preflight and before any MF update;
  the CFS result tree contains no artifacts.
- The prior artifact/local-toolkit repair did take effect. The new logs no
  longer attempt to preload `libmpi_gtl_cuda.so`, and they no longer say a local
  toolkit was requested. Instead CUDA_Runtime_jll reports an inherited request
  for CUDA `13.2.0`, while its pinned wrapper provides toolkits only through
  `13.0.2`. Leaving only `local = "false"` allowed the higher-load-path `version`
  preference to remain merged into the active GPU environment.
- A locally synchronized accepted state from the successful 2026-08-30 square
  campaign records CUDA.jl `5.9.5`, artifact runtime `13.0.0`, driver `13.0.0`,
  A100-SXM4-80GB, and passed runtime isolation. The GPU project now explicitly
  pins `CUDA_Runtime_jll.version = "13.0"` in addition to `local = "false"`,
  reproducing that demonstrated artifact family rather than selecting an
  untested version.
- The synchronized reconciliation ledger records measured fractional GPU
  node-hours of `0.057847222` for the six failed rerun jobs and releases
  `17.942152778` of their `18.000000000` requested ceiling. Those Perlmutter
  measurements are authoritative. Across the preceding failed/cancelled
  campaign and this rerun, the synchronized measured charge is `0.093611111`
  node-hours; no scientific result was produced by either attempt.
- Future branch logs now print the selected CUDA runtime, driver, and toolkit
  source immediately after the existing dense preflight. Only lightweight
  project parsing, source assertions, and whitespace checks are required for
  this preference-only repair; the user explicitly declined another heavy
  local integration suite.
- Launcher v1.13.2 adds a zero-node-hour `check-gpu-preferences` action that
  reads Julia's effective merged preferences without importing CUDA, HDF5, or
  MPI. Direct GPU submissions and continuations run it before acquiring the
  budget lock or calling `sbatch`; they require artifact mode, CUDA 13.0,
  artifact `MPICH_jll`, and no MPI preloads.
- Codex performed no Perlmutter operation, changed no budget ledger, and wrote
  no HDF5 artifact. A third campaign must not be submitted until the effective
  preferences report artifact CUDA `13.0` and the preceding terminal campaign
  has been reconciled on Perlmutter.

## 2026-09-02: bare Stage 1 analysis and gated Stage 2 handoff

- Analyzed the user-synchronized bare-ladder run
  `20260901_bare_t014_v0_stage1` at `L=64`, `V=0`, `t0=1.4`, and `chi=1200`.
  All 33 compact manifest rows were present. All six final sectors passed the
  saved gates; the final spin gap is `0.1533613320`, charge gap
  `0.0108583011`, hole pair binding `-0.1465102212`, and particle pair binding
  `-0.1485427484`.
- The rung-pair decay exponent is `0.76136 +/- 0.02146` with strong fits, while
  the charge exponent is `1.21579 +/- 0.08304` with materially weaker fits.
  Equal-time covariance spectra are broad and contain boundary-dominated modes,
  so they are retained as an unbiased screen rather than interpreted as
  susceptibility eigenvalues. At `tp=0.1`, especially
  `tp/charge_gap=9.2095`, the MPS+MF result must remain exploratory.
- The six sector logs contain 351 sweeps and 27.037 summed DMRG hours. The ideal
  six-way sector-array critical path is 9.051 hours, set by the spin-excited
  sector. No synchronized `sacct` or `/usr/bin/time -v` data were available, so
  actual CPU utilization, charge, and peak RSS were not inferred. The compact
  mirror is 2.046 MiB versus 6.725 GiB represented by the full manifest.
- Added the reproducible report under
  `docs/reports/bare_stage1_t014_v0_20260902/`, including a self-contained HTML
  report, Markdown narrative, machine-readable artifact metadata, deterministic
  CSV extracts, source hashes, and explicit evidence exclusions.
- Implemented Stage 2 projected-response discovery in `src/BareStage2.jl` and
  `scripts/run_bare_stage2.jl`. Fourteen motivated/covariance names
  orthonormalize to twelve independent directions: nine normal and three
  pairing. The implementation adds a strict number-conserving zero-field
  reference, a parity- and `Sz`-conserving pairing reference obtained by
  removing only `Nf`, exact MPO-conjugate response measurements, all-geometry
  map reuse, measured reciprocity and cross-block gates, and a separately
  submitted three-mode `h`/`h/2` validation with Richardson assembly.
- Added `configs/bare_stage2_t014_v0.toml`, the guarded user-run Perlmutter
  launcher `slurm/bare_stage2_cpu.sh`, and `docs/BARE_STAGE2_CPU.md`. Discovery
  retains the calibrated four-thread block-sparse topology, parallelizes the 12
  independent probes, and enforces an 18.65625 CPU-node-hour reservation cap.
  Preparation and assembly request only 8 GiB because they do not load the
  large MPS tensors. Optional validation has a separate 9.609375-node-hour
  bound and cannot be
  submitted accidentally with discovery.
- Codex performed no Perlmutter login, transfer, scheduler query, submission,
  cancellation, or accounting change. The persistent checkout remains
  `$CFS/m4863/MPS-MFT/ladder_mps_mft`; all documented commands are operator
  handoffs for the user.

## 2026-09-03: square V=0, t0=1.4 six-seed compact analysis

- Audited the user-synchronized campaign
  `20260902_phase1_square_t014_v000_seed_chi200_loose_cuda130` at `L=64`,
  `U=8`, `V=0`, `t0=1.4`, `tp=0.1`, density `0.9375`, square geometry, and
  `chi=200`. All six compact Float64 states are accepted period-one fixed
  points under the configured loose gates, and all required model, numerical,
  implementation, pair-binding, and geometry fingerprints match.
- The final d-wave-proxy profiles agree after global sign alignment to within
  `0.0083%--0.199%` RMS of the d-wave-seeded reference, and their RMS
  amplitudes span only `0.086%`. All have uniform `q=0` leg pairing, and the
  spatial audit finds no pairing phase-slip candidate. Charge-profile
  differences are below `0.95%` by the same comparison.
- Pairing-only branches have spin below the `1e-6` analysis floor, but
  stripe-started branches retain `3.58e-4--1.07e-3` spin-profile RMS at their
  seeded wavevectors. In particular, pure `m=4` stopped after six raw updates
  with relative residual `4.108e-3`, just inside the loose `5e-3` gate.
  Therefore the evidence supports one common pairing-dominated family with
  residual stripe memory, not six demonstrated high-precision identical fixed
  points.
- The authorized target-density-corrected canonical energy spread is only
  `7.715e-5` total (`6.027e-7` per physical site). The apparent ordering is not
  resolved: every last solve reached `maxlinkdim=200` with discarded weight
  approximately `2.33e-4--2.35e-4`, density corrections exceed the inter-seed
  spread, and no bond-dimension or full error extrapolation exists. Raw
  canonical energies were not ranked.
- The reviewed legacy table inside the repository already labels its generic
  `V=0,t0=1.4` row d-wave-dominant, while the specific high-amplitude CDW/SDW
  endpoint inherited from `t0=1.0` is not available in-scope as an identifiable
  artifact. The six matched-norm `1e-3` seeds do not exclude that basin.
- Recommended gate: replay the exact legacy-parent terminal fields at the
  current `V=0,t0=1.4` endpoint without normalizing their amplitude, using the
  current solver and matching fingerprints. This is one 12-hour one-of-four
  GPU branch, a plan-only ceiling of `3.000` node-hours and no smoke. Only if a
  distinct accepted endpoint survives should a bidirectional `t0` continuation
  be run; four new coarse branches would add at most `12.000` requested
  node-hours. The local ledger lacks this newest campaign, so live Perlmutter
  reconciliation remains authoritative.
- Reproducible outputs are under the campaign's
  `analysis/audit_20260903/`, `analysis/spatial_phase_defects_20260903/`, and
  `analysis/analysis_20260903.md`. Compact artifacts and histories were
  verified locally; full scratch MPS tensors and live scheduler accounting were
  not. No HDF5 file or ledger was modified, and Codex performed no Perlmutter
  operation.

## 2026-09-03: frozen legacy-field one-shot DMRG prepared

- The user supplied the exact legacy endpoint
  `stateless_data/results_L_64_U_8.0_V_0.0_t0_1.4_t_p_0.1_geometry_square_chi_200_density_0.9375_gpu.h5`
  and clarified that one fresh DMRG is required, but no mean-field loop or
  chemical-potential search is needed. The source is read-only and has SHA-256
  `5d7529713df02b1495b58ae2e9298c0c4da25ea95daf6cb47c43351943c93722`.
- Offline inspection finds `completed=true`, no saved period-two flag, saved
  density `0.9374967904`, and a final saved-map residual of `1.34216e-4`
  absolute and `9.98144e-4` relative. The legacy `E=-281.9204407` is the
  effective-Hamiltonian eigenvalue and is not itself comparable with the
  current canonical energies. Reconstructing the current functional from the
  saved legacy correlations gives a provisional target-density energy
  `-84.6163371905`, which is `0.0257361851` below the accepted six-state
  minimum (`2.01064e-4` per site). This is a useful signal but not a formal
  comparison because it retains the legacy DMRG result, its Float32-saved
  effective energy, and its unmeasured consistency error; the new solve is
  required.
- Added `scripts/prepare_frozen_legacy_energy.jl` and
  `scripts/run_frozen_legacy_gpu.jl`. The prepared job imports the exact legacy
  `alpha`, `beta`, `mu_cdw`, and `mu`, starts from a fresh product MPS, performs
  one `chi=200` Float64-CUDA DMRG, and then measures the outgoing fields,
  one-step raw-map residual, full current variational energy including double
  counting and target-density correction, DMRG truncation evidence, and quick
  charge/spin/entanglement diagnostics.
- Launcher v1.14.0 adds `plan-frozen-legacy` and
  `prepare-frozen-legacy SOURCE_RUN LEGACY_H5 NEW_RUN`. Submission remains the
  existing direct `submit` action, creates no smoke job, reserves one of four
  GPUs for three hours (`0.750` node-hours), writes the full MPS to scratch,
  and mirrors a stateless result to CFS. The one-shot campaign cannot use the
  continuation action.
- The six current states are rehashed and required to be accepted period-one
  fixed points with matching model, numerical, implementation, E_p-registry,
  geometry, and Float64 fingerprints before preparation. Their formal ranking
  is kept separate. The new result is always saved as a selection-ineligible
  diagnostic because a single map evaluation is not an SCF acceptance test.
- The local synchronized ledger reports `40.229027778` effective node-hours but
  does not yet contain the newest six-job campaign. Conservatively adding its
  unreconciled `18.000` ceiling and this job's `0.750` request projects
  `58.979027778` effective node-hours and `341.020972222` remaining under the
  400-hour cap. Live Perlmutter reconciliation is authoritative.
- Local validation was intentionally lightweight. The legacy schema and all
  six compact reference classifications/fingerprints were checked, including
  the Linux-style implementation hash. This Windows host has no usable Julia
  runtime or Bash installation, so final Julia loading and launcher syntax
  validation must occur during the non-submitting Perlmutter preparation step.
  Codex ran no DMRG, CUDA initialization, transfer, scheduler command, or ledger
  mutation.

## 2026-09-03: frozen job 57886813 reporting failure isolated

- User-reported status shows Slurm job `57886813` as `FAILED`, but its compact
  Float64 state is present with status `frozen_field_evaluation` and finite
  canonical energy `-84.624511246252`. The DMRG therefore completed without
  reaching the internal deadline; this is not a CUDA or quantum-solve failure.
  The status helper does not display the target-density-corrected energy, so
  this number must not be compared directly with the ranked six-state values.
- The first failure is a deterministic reporting bug after `state.h5` and
  `diagnostics.h5` are written: `run_frozen_legacy_gpu.jl` requested a
  nonexistent `qy` field while assembling `frozen_dmrg_observables.tsv`; the
  diagnostics peak records expose that component as `ky`. The runner now uses
  `ky`. A second latent bug was repaired before it could be reached: the runner
  now derives `target_label` from the prepared output directory rather than
  referring to the preparation-only `TARGET_LABEL` constant.
- Added `scripts/finalize_frozen_legacy_result.jl` to rebuild
  `frozen_dmrg_observables.tsv`, `energy_comparison.tsv`, and `run_summary.md`
  from the compact state, compact diagnostics, and the six hashed accepted
  references. It modifies no HDF5 artifact, runs no DMRG, requests no
  allocation, and does not touch the ledger. The completed DMRG must not be
  resubmitted merely to repair reporting.
- The frozen diagnostic remains selection-ineligible because no SCF acceptance
  test was performed. Its target-density-corrected energy, density, raw-map
  mismatch, truncation evidence, and consistency errors must be read from the
  artifact before any conditional energetic comparison is reported.

## 2026-09-03: square V=0, t0=1.4 chi=400 two-lineage comparison prepared

- Prepared a two-branch bond-dimension/basin control at square `L=64`, `U=8`,
  `V=0`, `t0=1.4`, `tp=0.1`, and density `0.9375`. The lineages are the
  accepted pure-d-wave `m=0` branch and the frozen legacy-like diagnostic's
  measured outgoing map. Both load their exact full `chi=200` scratch MPS
  checkpoints and start fresh `chi=400` histories.
- The legacy-like parent deliberately uses `fields/restart`, which the frozen
  job stored from the measured current map, rather than the applied legacy
  tensor. Preparation rehashes the full parent and refuses it unless inactive
  onsite-beta restart entries are zero to `1e-12`; those unused legacy entries
  caused the misleading `0.90069` all-entry residual but do not enter the MPO
  or canonical functional.
- Added `configs/phase1_gpu_square_v0_chi400_tight_compare.toml` with 16 DMRG
  sweeps, `maxdim=400`, cutoff `1e-11`, DMRG energy tolerance `1e-9`, inner and
  outer density tolerances `1e-4`, field gates `1e-7` absolute or `1e-4`
  relative, target-corrected energy stability `1e-7` per site, two stable
  records, a 20-update raw-map probe, and up to 80 MF updates. Anderson remains
  downstream of the raw recurrence policy.
- Added `scripts/prepare_phase1_square_v0_chi400_compare.jl`. It validates both
  compact-to-full links, full hashes, source classifications, model/numerical/
  implementation/E_p/scalar provenance, physical restart fields, and common
  new numerical fingerprint before writing two immutable parent configs and a
  manifest. The two accepted new states may be ranked within `chi=400`; changes
  from `chi=200` are convergence diagnostics rather than a cross-fingerprint
  variational ranking.
- Launcher v1.15.0 adds read-only
  `plan-square-v0-chi400-compare` and preparation-only
  `prepare-square-v0-chi400-compare PAIRING_RUN FROZEN_RUN NEW_RUN`. Direct
  `submit` creates no smoke and requests two 12-hour one-of-four-GPU jobs. The
  first-segment ceiling is `6.000` node-hours; four segments for both lineages
  would be `24.000` node-hours and are not pre-authorized.
- The last synchronized reconciliation ledgers total `42.024097222` active
  node-hours. A first submission would project to `48.024097222` active and
  `351.975902778` unreserved under the 400-hour project cap. Live Perlmutter
  accounting remains authoritative and must be checked immediately before
  submission.
- The handoff and completion gates are documented in
  `docs/SQUARE_V0_T014_CHI400_COMPARISON_2026-09-03.md`. Codex performed no
  transfer, Perlmutter login, scheduler action, or ledger mutation. Local
  validation was limited to static source inspection and TOML parsing because
  this Windows host has no runnable Julia or Bash installation.

## 2026-09-03: bare Stage 2 response and square-SCF seed analysis

- Analyzed the synchronized Stage 2 pilot at `V=0`, `t0=1.4` from
  `output/bare_stage2/20260902_bare_t014_v0_stage2`. All 22 compact artifacts
  passed the local size/hash/stateless verifier; the full scratch artifacts
  were not locally available for re-verification.
- Wrote the portable technical report and reproducible extractor under
  `docs/reports/bare_stage2_t014_v0_20260903/`. The report compares the
  projected response spectrum, the geometry-dependent bare image `F(0)`, the
  six existing square `chi=200` seeds and accepted endpoints, and synchronized
  Slurm resource accounting.
- The square bare image has norm `0.0306655`, dominated by a common beta
  background and uniform Hartree offset; its nonuniform remainder has norm
  `0.00320857`. Its charge-even profile has cosine `0.7251` with the current
  `m=4` charge template but contains no resolved spin or pairing source.
- All six current square branches are accepted period-one fixed points with
  nearly identical total, beta, Hartree, and pairing-field scales. Their
  beta-plus-Hartree distance from `F(0)` is only `0.00213`--`0.00262`.
  Remaining spin-odd amplitudes retain seed dependence, so the recommended
  basin test is `F(0)` plus the same controlled symmetry-breaking increment,
  compared against the existing zero-background starts; `F(0)` alone is not a
  new symmetry-breaking direction.
- The retained response basis has maximum leakage `0.8327`, and the planned
  `h/2` validation has not run. Reported eigenvalues are therefore pilot
  subspace estimates. Measured Stage 2 charge was `3.87395` CPU node-hours.
- The report artifact passed schema and portable-package validation; only
  structural HTML verification was available because no local Chromium
  headless executable is installed. No DMRG, transfer, Perlmutter login, or
  scheduler action was performed during this analysis.

## 2026-09-03: square smooth-pairing five-point grid fill prepared

- Defined the five missing coordinates of the square `t0={1.0,1.2,1.4}` by
  `V={-0.4,-0.2,0.0}` grid: `(1.0,-0.4)`, `(1.0,-0.2)`, `(1.2,-0.4)`,
  `(1.2,-0.2)`, and `(1.2,0.0)`. Every point has an exact signed `E_p` row in
  `data/E_p_values.csv`; no interpolation or additional isolated-ladder job is
  required.
- Replaced the initially considered exact-zero field control with the user's
  requested previously tested smooth pairing control. It uses the
  `legacy_pairing` matched-mode seed at total norm `1e-3` and common RNG seed
  `1404`: coefficients vary across relative bond and leg-pair classes but are
  copied along all rungs. `beta` and `mu_cdw` begin at zero. This contains no
  center-of-mass spatial noise, while its nonzero `alpha` avoids the exact
  number-conserving normal-sector lock. It is not claimed to be fully
  symmetry-unbiased or an exhaustive basin search.
- Added `configs/phase1_gpu_square_grid_smooth_pairing_chi200_loose.toml` and
  `scripts/prepare_phase1_square_smooth_pairing_grid.jl`. Preparation verifies
  the five point contracts, exact pair-binding values, common loose numerical
  fingerprint, common initial-seed fingerprint, source/Manifest implementation
  fingerprint, no lineage, zero normal fields, matched seed norm, and exact
  center-of-mass uniformity before writing the five immutable configs and
  manifest.
- The campaign uses 12 sweeps, `chi=200`, cutoff `1e-10`, DMRG energy tolerance
  `1e-6`, inner/outer density tolerance `1e-3`, loose field gates `1e-6`
  absolute or `5e-3` relative, the corrected oscillation and slow-mode gates,
  a 20-update raw-map probe, and up to 80 MF updates. Starting `mu` values are
  rounded, V-informed bracketing guides (`0.55`, `1.10`, `1.65`); density is
  still solved independently and uses the carried compressibility slope plus
  `1e-8` warm re-solve noise.
- Launcher v1.16.0 adds `plan-square-smooth-pairing-grid` and
  `prepare-square-smooth-pairing-grid NEW_RUN`. Direct submission creates five
  scientific jobs and no smoke. The run has separate CFS and scratch trees
  from the currently running launcher-v1.15.0 square `chi=400` comparison.
  Worker and narrowly scoped continuation compatibility for that active run
  are retained because `src/` and the GPU Manifest did not change.
- Five 12-hour one-of-four-GPU requests reserve `15.000` node-hours. The same
  seed measured `0.123472222` and `0.228125000` node-hours at the completed
  `t0=1.4,V=-0.4` and `V=0` endpoints, projecting `0.878993055` actual
  node-hours for five jobs; the broader twelve-branch mean projects
  `1.010243055`. These are estimates only. The last synchronized active ledger
  total is `42.024097222`; including the running comparison's unreconciled
  `6.000` ceiling and the new `15.000` envelope would give `63.024097222`
  active and `336.975902778` unreserved. Live Perlmutter accounting supersedes
  this illustration.
- Full MPS artifacts remain scratch-first and only stateless mirrors go to CFS.
  No cross-point energy ranking is authorized. Codex submitted or cancelled no
  job, modified no ledger or HDF5 artifact, and performed no Perlmutter access.

## 2026-09-03: cubic-unfrustrated smooth-pairing eight-point grid prepared

- Defined the eight missing cells of the `cubic_unfrustrated`
  `t0={1.0,1.2,1.4}` by `V={-0.4,-0.2,0.0}` grid, treating the legacy
  `(1.0,0.0)` result as the ninth coverage point. All eight have exact signed
  `E_p` registry rows; no interpolation or additional CPU calculation is
  required.
- Reused the square-grid smooth `legacy_pairing` protocol exactly: matched
  field norm `1e-3`, RNG seed `1404`, relative-bond/leg-pair coefficients
  copied along every rung, and zero initial `beta` and `mu_cdw`. This opens the
  pairing sector without center-of-mass seed noise or inherited walls. It is a
  controlled access seed, not an exhaustive basin comparison.
- Added
  `configs/phase1_gpu_cubic_unfrustrated_grid_smooth_pairing_chi200_loose.toml`
  and generalized the existing grid preparer to validate either square or
  cubic-unfrustrated point contracts. It checks exact `E_p`, common numerical
  and seed fingerprints, distinct model fingerprints, zero normal fields,
  matched norm, center-of-mass uniformity, and absence of lineage before
  writing eight immutable configs and a manifest.
- Launcher v1.17.0 adds read-only
  `plan-cubic-unfrustrated-smooth-pairing-grid` and preparation-only
  `prepare-cubic-unfrustrated-smooth-pairing-grid NEW_RUN`. Direct `submit`
  creates eight scientific jobs and no smoke. Worker and continuation support
  is retained narrowly for the active v1.15.0 `chi=400` comparison and v1.16.0
  square-grid campaign; solver `src/` and the GPU Manifest are unchanged.
- Eight 12-hour one-of-four-GPU requests reserve `24.000` node-hours. Prior
  fresh cubic-unfrustrated `chi=200` campaigns scale to approximately
  `6.842777776--10.011481480` measured node-hours for eight jobs. The last
  synchronized ledger plus the active `6.000` chi=400 ceiling, submitted
  `15.000` square grid, and proposed cubic envelope projects
  `87.024097222` active and `312.975902778` unreserved. Live Perlmutter
  accounting remains authoritative.
- Full MPS data remain scratch-first with stateless CFS mirrors. Cross-point
  and cross-geometry energy ranking is forbidden. Codex performed no transfer,
  Perlmutter access, submission, cancellation, ledger mutation, or HDF5 write.

## 2026-09-03: square t0=1.4, V=-0.4 legacy-stripe comparison prepared

- Prepared a two-branch square `L=64`, `U=8`, `V=-0.4`, `t0=1.4`, `tp=0.1`,
  density `0.9375`, `chi=200` loose campaign. One branch imports the exact
  terminal fields and chemical potential of the legacy square `(1.0,0.0)`
  stripe; the other is a fresh center-of-mass-uniform smooth pairing control.
  Both begin from fresh product MPS states and share the current model,
  numerical, implementation, scalar, and exact-`E_p` contracts.
- Locked the input to SHA-256
  `ae6a3bfe76ca8f06f2396fd731b18bca8539e0b7ee68df016cc9156fdceeb074`.
  The source has active `max|beta|=0.03440698`, `max|mu_cdw|=0.05339365`, and
  `mu=1.6586343178`, confirming that this is the intended high-amplitude stripe
  rather than another weak seed.
- The legacy file also stores 256 inactive same-physical-site `beta` entries,
  with maximum magnitude `0.12824499`. Current MPO construction, the mean-field
  map, and the canonical functional omit these entries, but retaining them
  would create an artificial first-step raw residual. The preparer writes an
  immutable field-only derivative in the new CFS run directory, zeros only
  those inactive entries, verifies every physical field unchanged, and records
  both source hashes and the sanitization policy. The original HDF5 is untouched.
- Added
  `configs/phase1_gpu_square_t014_vm04_legacy_stripe_compare_chi200_loose.toml`
  and `scripts/prepare_phase1_square_legacy_stripe_compare.jl`. Launcher
  v1.18.0 adds read-only `plan-square-legacy-stripe-compare` and preparation-
  only `prepare-square-legacy-stripe-compare LEGACY_H5 NEW_RUN`; direct submit
  creates two scientific jobs and no smoke. Compatibility is retained for the
  active v1.15.0 chi=400, v1.16.0 square-grid, and v1.17.0 cubic-grid campaigns
  because solver `src/` and the GPU Manifest are unchanged.
- Formal energy ranking is limited to the two new endpoints and only if both
  are accepted with matching fingerprints. The six older `(1.4,-0.4)` states
  remain qualitative context because their numerical and implementation
  fingerprints differ. The legacy stored effective energy is not rankable.
- Two first segments reserve `6.000` node-hours. Historical target-point jobs
  imply a `0.2133--0.2469` node-hour analog estimate, with extra uncertainty
  for slow stripe drift or basin escape. Depending on whether the prepared
  cubic grid has also been submitted, the last synchronized ledger scenario
  plus known campaign ceilings would project either `69.024097222` or
  `93.024097222` active node-hours. The live Perlmutter ledger and `sacct`
  supersede both illustrations.
- A focused local preparation check passed against the supplied legacy file:
  exact hash and metadata, derived-seed readback, field-preservation checks,
  two generated configs, common fingerprints, and manifest construction. No
  DMRG, CUDA, transfer, Perlmutter access, scheduler action, or ledger mutation
  was performed. See
  `docs/SQUARE_T014_VM04_LEGACY_STRIPE_COMPARISON_2026-09-03.md`.

## 2026-09-04: durable project-continuity layer added

- Added `docs/README.md` as the stable documentation entry point,
  `docs/PROJECT_STATE.md` as the short mutable current snapshot,
  `docs/ARCHITECTURE.md` as the code/data/host map,
  `docs/decisions/README.md` as an index of established decisions, and
  `docs/plans/ACTIVE.md` as the current completion sequence.
- Updated `AGENTS.md`, the project `README.md`, and
  `docs/NEW_DEVICE_CHAT_PROMPT.md` so a new task reads the current snapshot and
  relevant documents rather than loading the full append-only run history.
  Marked `docs/DEVICE_HANDOFF_2026-08-25.md` explicitly historical.
- User-reported live status at the time of this documentation update: the three
  latest Perlmutter jobs remain pending. Their job IDs and campaign membership
  were not supplied with that report and were not inferred from local files.
- The continuity snapshot records the 400-additional-node-hour boundary and
  preserves budget for later bond-dimension and length convergence, while
  treating live Perlmutter ledgers and `sacct` as authoritative.
- Validation was documentation-only: file presence, links, Git state, and
  consistency with the latest campaign records were checked locally. No Julia
  or DMRG run, Perlmutter access, scheduler action, transfer, HDF5 mutation, or
  ledger change was performed.

## 2026-09-04: systematic implementation and scientific-status review

- Reviewed baseline `a744d29` on `codex/mps-mft-phase0-refactor`, including the
  modular solver and scientific contracts, legacy workflow context, current
  campaign controls, locally synchronized results, and primary literature.
  The dated narrative and evidence are in
  `docs/reports/systematic_review_20260904/`. Recommendations have not been
  applied to production solver code or campaign controls.
- Inventoried 52 readable Phase 1 terminal paths. Reused
  `scripts/audit_scf_numerics.py` for 42 with adequate histories; nine old files
  lack applied histories and one frozen diagnostic has only one record. The
  limited screen preserves 17 of 28 stored accepted flags and downgrades 11;
  it changes three additional already-unaccepted recurrence labels. Passing
  this screen is not current scientific recertification.
- All six current square `(t0,V)=(1.4,0)`, chi=200 terminal compact files match
  their recorded manifest SHA-256 and size and pass the history screen. The
  exact manifest checks are saved in `current_terminal_hash_checks.csv`.
  Their target-density-corrected energy spread is `6.0273786395e-7 t/site`;
  fine ordering is unresolved by the available error budget. Applied density
  correction magnitudes are not the remaining error after correction.
- Six focused Julia assertions in four test sets reproduced: omitted CUDA
  extension in the implementation fingerprint; missing/nonfinite ranking
  inputs admitted by the reader contract; absent inner-DMRG acceptance gate;
  and `mu_initial` included in the model fingerprint. The in-memory fixtures
  execute actual source without ITensor/DMRG. Exact source is preserved in
  `contract_checks.txt` and `review_notebook.ipynb`; the temporary `.jl` file
  was removed after execution. The tests establish code gaps, not invalidity
  of the six current endpoints.
- Local commands: `python -X utf8
  ladder_mps_mft/docs/reports/systematic_review_20260904/extract_review.py`
  (about 3.05 seconds); Julia 1.12.7 `--startup-file=no` on the temporary review
  contract file (4.77 seconds, six assertions passed); `python -m unittest
  discover -s ladder_mps_mft/test -p test_spatial_phase_defects.py` (six passed,
  0.008 seconds test execution). A generated tracked bytecode change was
  restored exactly to HEAD bytes; no index mutation was needed.
- Corrected the review interpretation of the even-particle finite-size charge
  gap: it probes compressibility and is not the eliminated pair-breaking
  excitation. `tp/|Ep|` remains about 0.6825 at the bare control, so selected
  weaker-hopping checks are still recommended. Stage 2 remains discovery:
  maximum omitted-response norm fraction 0.83267, second-amplitude validation
  absent. Recommend dressed-reference response, basis/amplitude controls,
  and matched geometry/stripe comparisons before phase claims.
- Literature synthesis includes the August 2026 Köhler/Kantian mixD preprint
  as related work, with different-Hamiltonian and finite-size limitations
  explicit. This review is targeted, not an exhaustive novelty certification.
- The report artifact passed validation and its MCP handoff returned success.
  Visual verification limits and final handoff receipts are recorded alongside
  the report. No full Julia suite, local DMRG solve, CUDA timing, full-scratch
  validation, Perlmutter access, transfer, scheduler action, reservation, or
  ledger change occurred. The previously user-reported three pending jobs
  remain the latest available live-status information; IDs are still unknown.

## 2026-09-05: synchronized chi=400 two-lineage energy analysis

- User synchronized `output/phase1_gpu` and reported the chi=400 comparison
  finished. Synced jobs `57905744` (pairing) and `57905745` (legacy-like)
  terminated as `stagnated` after 32 records and `time_limit` after 40 records,
  respectively. Both are unaccepted, solution kind `none`, period zero; their
  accepted-solution energy fields are NaN. Job completion is not fixed-point
  acceptance. Final summaries are dated 2026-09-05 23:51 UTC and
  2026-09-06 01:31 UTC.
- At square L=64, U=8, V=0, t0=1.4, tp=0.1, n=0.9375, chi=400, reconstructed
  target-corrected terminal energies are `-84.5963569761 t` (paired) and
  `-84.6275072176 t` (stripe). Stripe minus pairing is `-0.0311502415 t`, or
  `-2.43361262e-4 t/site`. This is a provisional diagnostic difference, not an
  accepted-only energy ranking. The gap is approximately 347 times the sum of
  the two last-ten-record energy ranges; that range is not an error bound.
- The stripe's favorable balance comes from the spin component of the
  transverse Hartree energy (`-0.00422717 t/site` difference), compensating
  bare-ladder, pairing, normal-exchange and charge costs. Calling the full
  spin-resolved Hartree contribution a charge-only gain would be misleading.
- Pairing passes the final two-record raw field/density checks and the final
  slow-mode, energy-stability and effective-consistency gates, but its
  Hamiltonian-identity error `2.68869e-10 t/site` exceeds `1e-10`. The stripe
  has relative residual `2.49005e-4` against `1e-4` and reaches the solver
  deadline during its last DMRG. Last-sweep total-energy changes are
  `2.17364e-9` and `4.29925e-8 t`, both above the `1e-9 t` inner tolerance.
  Maxlinkdim is 400 for both; final-sweep discarded weights are `1.19241e-6`
  and `6.20582e-7` (solve-wide maxima `6.35282e-5` and `1.59759e-4`).
- Distinct textures persist: central-half pairing-field proxy RMS is
  `0.00441635` versus `6.03183e-10 t`; charge peak-to-peak is `0.00255596`
  versus `0.14027481`; staggered leg-odd spin RMS is `1.65569e-6` versus
  `0.18624224`. Chi=200 parent overlays show qualitative survival with modest
  profile changes. The legacy parent is frozen-field, so parent comparisons
  are not controlled chi extrapolation or matched-fingerprint rankings.
- Reused existing SCF audit and spatial-profile definitions in
  `docs/reports/chi400_comparison_20260905/analyze.py`. All eight compact
  manifest artifacts pass SHA-256 and size checks; all six HDF5 mirrors pass
  stateless/no-MPS/recorded-full-hash checks. Model, numerical, implementation,
  full-tree, GPU Manifest, E_p registry and scalar fingerprints agree. Config
  and GPU Manifest hashes match provenance; recorded parent identities match
  the synced parent mirrors. No full scratch existence or hash verification
  is implied. Exact terminal full hashes are
  `c92c75428ceb3beec2f324101f784c0a353ebd000ed9d2f626251ef0f83a039c`
  (pairing) and
  `f6501c24decfcf25aab1484adc1a121b7aec0cbd3717e441e75b8870e61c276e`
  (stripe).
- Local command from repository root: `python -B -X utf8
  ladder_mps_mft/docs/reports/chi400_comparison_20260905/analyze.py` (about
  four seconds including assertions and static figure generation). Both final
  PNG figures were visually inspected. A preliminary inline HDF5 schema read
  encountered a group/dataset AttributeError; it was corrected before the
  successful saved analysis. No Julia or expensive DMRG tests were relevant.
- User identifies the three still-pending campaigns as square grid, cubic
  unfrustrated grid, and square `(1.4,-0.4)` stripe/control. Their synced
  submission IDs are `57908558,57908560--57908563` (5),
  `57909095--57909102` (8), and `57909911--57909912` (2). All have zero local
  terminal states. Pending is user-reported; jobs.tsv verifies submission
  identity, not current scheduler state. No reconciled accounting was supplied.
- Decision: retain the stripe's energy advantage as provisional; diagnose the
  paired identity failure and complete matched SCF/inner-DMRG convergence
  before ranking, then use selected higher-chi controls. Do not silently
  relax gates or relabel endpoints. Updated `PROJECT_STATE.md` and active plan;
  detailed analysis, CSV evidence and figures are under the dated directory.
  Preserved pre-existing September 4 review changes. No solver/config edit,
  acceptance/HDF5 mutation, continuation preparation, transfer, Perlmutter
  access, scheduler action, or ledger change occurred.

## 2026-09-06: L=96/128 chi=200 pairing/stripe seeds prepared for review

- User direction supersedes the previous next-action recommendation: treat
  the chi=400 paired endpoint's tiny identity discrepancy as converged for
  the energetic question; the stripe's falling residual and early plateau
  support a gap reasonably robust to the tested chi increase. Proceed with
  limited finite-size scaling. Historical HDF5 status and acceptance flags
  are unchanged. The user requested a seed snapshot before submission.
- Prepared exactly four field-only seeds: pairing and stripe at L=96 and 128,
  chi=200, square U=8,V=0,t0=1.4,tp=0.1,n=0.9375. The source is each latest
  L=64 chi=400 terminal `fields/measured`, not the original weak access seed
  or an MPS resize. Original compact hashes are
  `bfcb03c7b3948a8b0f552fb45860b2fa352ea5a58feecc52ba4e215b7dfb52f1`
  (pairing) and
  `229c118bec2f491997db0bf2be2da039863fca7695fdd0c44fdeadef1f2c76ab`
  (stripe). Sources match their compact manifests.
- Added `scripts/prepare_phase1_finite_size_seeds.py`. Every signed relative
  bond separation is retained and extended in its center coordinate. Original
  left/right 32-rung onsite fields and half-block bonds are exact. Pairing
  inserts a flat, sublattice-preserving center with eight-rung tapers. Stripes
  insert one/two 32-rung cells using positive quintic overlaps of the actual
  central source waveform. No global rescaling, end-to-end bond, range
  extension or inactive onsite-beta term is introduced.
- Stripe charge trough counts are 4,6,8 at L=64,96,128. The finite source's
  approximately periodic texture gives 15/17-rung spacings in the inserted
  cell; this small phase mismatch is absorbed in the smooth overlap. The
  maximum charge and staggered-spin neighboring-field steps do not exceed
  the original. The exact metric rows and plots are stored in
  `docs/reports/finite_size_seeds_20260906/`.
- User explicitly selected keeping the L=64 E_p at all lengths. Added the
  narrow `pair_binding.reference_L` configuration option with exact lookup,
  explicit fixed-reference mode, reference-length provenance/HDF5 metadata
  and model-fingerprint inclusion. Default same-length fingerprints remain
  unchanged. No fabricated registry entries or new pair-binding jobs exist.
  This changes the local implementation fingerprint; pending Perlmutter
  campaigns must retain their existing source checkout until finished, or
  the new comparison must use a separate checkout.
- Four review configs under `configs/phase1_gpu_square_size_compare_chi200/`
  use chi=200 and the tight comparison settings, with identity tolerance
  relaxed to `1e-8 t/site` and explicit fixed L=64 denominator. Target particle
  numbers are 180/240. Field inheritance retains each source chemical
  potential and starts a fresh product MPS. Output remains explicitly
  `UNPREPARED_SIZE_COMPARE`; no scratch/launcher campaign is prepared yet.
- Seeds live in `output/seed_previews/20260906_square_t014_v0_L96_L128_chi200/`.
  SHA-256 values: pairing L96
  `5ee1cfb55d3dbe67f488c4285e5593294c41149de190d60d1526968a21401193`,
  pairing L128
  `807eeddbc989ee4fd2f6a03fac4e4d2521a11278d1d87367bc865bf17f9388b8`,
  stripe L96
  `e4fe4003735d110a9b843302681cfe2711c784497987a2694f083ae4d86f8601`,
  stripe L128
  `0a397db2c74737ffac4e1ff2ef9ac0faf11ccef56383e24c43680657f2438316`.
- Local validation: preparation and assertions in about four seconds;
  `python -B -m unittest discover -s ladder_mps_mft/test -p
  test_finite_size_seeds.py` passed four geometry tests in 0.057 seconds.
  Julia 1.12.7 with `--startup-file=no --compiled-modules=existing
  --project=ladder_mps_mft -L ladder_mps_mft/test/test_fixed_reference_length.jl
  ladder_mps_mft/scripts/verify_phase1_finite_size_seeds.jl` passed 16 config
  assertions (7.6 seconds) and 83 real-HDF5 readback assertions (5.3 seconds),
  plus package loading. No local DMRG or expensive full suite was needed.
  The PNG was visually inspected; regeneration verified identical immutable
  seed contents and unchanged seed hashes. An exploratory inline profile
  summary had a Python parenthesis syntax error; the saved metric extraction
  was corrected and completed successfully.
- Updated current state, active plan, documentation links and the prior
  analysis's interpretation addendum. Prepared no extra L=64 run, submission,
  scheduler reservation, continuation or ledger change; performed no
  Perlmutter access, transfer or source update on that host. Next step is
  user seed review, followed by isolated submission preparation and current
  accounting reconciliation.

## 2026-09-07: Introduction literature review and maintained BibTeX bibliography

- Created `docs/literature/literature_review.tex`, `references.bib`, and a
  compiled 29-page PDF in response to the user's request. Followed the user's
  preference for plain LaTeX annotations: 49 papers in eight groups from the
  Hubbard problem to the closest MPS+MF predecessors, with approximately
  140-180 words per annotation, persistent links, stable citation keys, and
  a short introduction outline.
- Checked bibliographic metadata against publisher/Crossref and arXiv
  records. The collection contains 46 journal publications and three labeled
  preprints. `SOURCE_NOTES.md` records the narrative search scope and separates
  abstract-based summaries from nine papers checked in selected full-text
  sections. The review does not assert exhaustive coverage or project novelty.
- The closest comparisons include the repulsive-ladder MPS+MF demonstration
  published in 2023, the 2025 SC/CDW extension, and the 2026 mixed-dimensional
  ladder preprint. Model, hopping-symbol, binding-sign, boundary, and
  correlation-exponent distinctions are stated where relevant.
- Compiled locally with portable Tectonic 0.17.0 and BibTeX. Focused document
  checks found 49 unique matching annotation/bibliography keys, no duplicate
  DOI or arXiv identifiers, no unresolved citations, no BibTeX warnings, no
  overfull boxes, and no characters outside the checked page margins. All
  expected DOI/arXiv links are embedded in the PDF (86 distinct external
  links). Rendered and visually inspected all 29 pages; checked the copied
  final PDF against the build output by SHA-256.
- Added build and bibliography-maintenance instructions and a link from the
  documentation map. Raw metadata and one-time build/QA tools remain in the
  ignored `output/literature_tools/` directory. No numerical code, campaign
  state, or Perlmutter operations were needed; no DMRG tests were run.

## 2026-09-08: square chi=200 grid compiled and divergence diagnosed

- User reports the square grid finished, while cubic and the separate
  `(t0,V)=(1.4,-0.4)` stripe/control runs are still running. Five local square
  terminal states are synced: four accepted fixed points and one `diverging`
  endpoint at `(1.0,-0.4)`. Two cubic terminal files were seen during inventory
  but not analyzed. No live scheduler or accounting verification is implied.
- User explicitly approved using legacy `(1.0,0)` and `(1.4,-0.2)` coverage,
  whose seed ancestry is not recorded, and including the new diverging
  endpoint with a flag and analysis. Requested seed rule excludes inherited
  converged-stripe states but permits independent small stripe patterns.
- Added `scripts/compile_square_grid.py` and generated the self-contained
  40.97-MiB `output/square_grid_chi200_20260908/square_grid_chi200.h5`: six
  accepted small-seed fixed points, two labeled legacy-completed points, one
  flagged terminal diagnostic. Terminal fields/correlations, original field
  snapshots, provenance, config text and all 17 candidate records are embedded;
  only the divergent point additionally retains its complete field history.
  No source state or acceptance flag changed. Bundle SHA-256:
  `36baaf52b7fd44c7cb84796fc600e3be4f2e1c12c358ed58782541aec81fa2cb`.
- At both `(1.4,-0.4)` and `(1.4,0)`, selected the independent small
  `stripe_pairing_m004_chi200_loose` seed by corrected canonical solution
  energy among accepted, history-screened, same-fingerprint candidates. These
  energy choices are numerically near ties. August 30 pure `stripe_m004`
  fails the existing slow-mode screen and is excluded. Unaccepted tight-five
  probes, frozen inherited stripes and chi=400 endpoints were not substituted.
  All source paths, compact/full hashes and comparisons are in
  `docs/reports/square_grid_20260908/selection.json` and `selection.csv`.
- Added `plot_square_grid.jl`, reusing the original Fourier renderer without
  legacy-source edits. The default uses physical correlations; `source=:mf`
  uses measured field proxies with Hartree-to-beta-diagonal mapping. Both
  PNG/PDF variants retain the old five-region layout, default boundary trim
  and shared log scale, with LEGACY labels and divergence hatching. The
  snapshot-specific click callback opens terminal profiles and Fourier maps.
- `scripts/analyze_square_grid_divergence.py` reproduces the exact stop:
  iteration-30 residual `0.03824999010` exceeds `8 * 0.003551837833 =
  0.02841470267`, using the best trailing accelerated record (22), a ratio of
  10.76907. The global residual-best record 5 is an earlier pairing plateau.
  During unmixed records 2--21, pairing decays and a stripe-like normal texture
  develops. Record 21 has contraction `0.9994040` and extrapolated relative
  residual `5.9702`; its low raw residual is not acceptance.
- Final squared residual is 74.42% Hartree and 25.58% exchange, with negligible
  pairing. Fields remain finite; density error is `9.2173e-11`. Last corrected
  energy change is `-1.6535e-4 t/site`; identity/consistency pass, but field and
  energy-stability gates fail. Last DMRG sweep change `9.6662e-7 t` meets its
  loose tolerance, with chi=200 and last-sweep discarded weight `2.5142e-5`.
  Interpretation: unconverged slow striped texture with a spike during
  acceleration, not proven unbounded growth or a validated physical orbit.
  Detailed evidence and two static diagnostic figures are in the report.
- Local commands: `python -B -X utf8 scripts/compile_square_grid.py` (about
  two seconds), Julia 1.12.7 `--startup-file=no --compiled-modules=existing
  plot_square_grid.jl` (seconds plus package loading), and
  `python -B -X utf8 scripts/analyze_square_grid_divergence.py` (about three
  seconds). The local WindowsApps Julia alias was inaccessible; used the
  installed `C:/Users/Kevin/.julia/juliaup/julia-1.12.7+0.x64.w64.mingw32/bin/julia.exe`.
  Corrected an initial new-wrapper docstring parse error before successful
  rendering; the wrapper includes the legacy renderer directly.
- Validation: all 17 candidate compact-state SHA-256/size/full-hash/config
  checks and stateless/no-MPS checks passed. All nine plot snapshots exactly
  match source arrays after the documented Hartree mapping. A temporary Julia
  check passed 95 assertions (12.3 seconds): physical profiles and Fourier
  maxima at all nine points, flag counts, and click-through figures. Both
  grid variants and both divergence figures were visually inspected. Temporary
  inspection/test scripts were removed. Updated current state, documentation
  map and active plan. No expensive full suite, DMRG, full-scratch validation,
  Perlmutter access, transfer, scheduler action, continuation or ledger change.

## 2026-09-08: restore full histories on square-grid clicks

- User clarified that clicking a grid point must open the former full MF
  profiles-and-middle-histories plot. The first compilation retained only
  terminal snapshots for most points; corrected that omission without changing
  selected runs, source states, convergence flags or Fourier quantities.
- Rebuilt `output/square_grid_chi200_20260908/square_grid_chi200.h5` as schema
  v2, now 104.76 MiB. Every selected Phase 1 source retains full original
  applied/measured field histories, stored seeds and per-update diagnostics;
  both legacy sources retain all six saved history arrays. Every new history
  dataset and all original terminal plotting arrays were checked exactly
  against the source before atomically replacing the compiled bundle. Hash:
  `389954bf423b859bdeb005a6ffb11580226b5aaecac401dc8f8a1570478ff5cf`.
- Grid clicks lazily extract the embedded source history and use the existing
  `plot_phase1_mf_profiles_and_middle_histories` adapter for new results, with
  the seed at plotted iteration 1. Legacy clicks use the original correlation
  history view (or Hartree-mapped MF histories for `source=:mf`). The grid and
  Fourier map still use the selected terminal physical correlations by default.
  New-run per-update views explicitly display saved measured MF fields.
- Fixed the Phase 1 adapter's leading free-standing docstring, which Julia
  otherwise attaches to an `if` expression and rejects. Guarded renderer
  includes allow loading the grid in a Julia session where the old renderer
  is already present. Shortened the new MF view's long pairing subplot titles
  to fit the original two-column figure size.
- Local validation: Python rebuild about two seconds; all-nine-click Julia
  check passed 109 assertions in 34.1 seconds. Exact five-channel middle
  histories and their iteration axes match embedded inputs; plotted sample
  counts are 13,18,31,7,7,6,26,4,7 in sorted point-ID order. A second focused
  check passed 10 slider assertions, covering first/final profile changes and
  history cursor movement in both Phase 1 and legacy views. Rendered and
  inspected representative new/legacy history figures. Temporary Julia checks
  removed after validation. Updated report and project snapshot. No DMRG,
  expensive full suite, Perlmutter action or source-artifact modification.

## 2026-09-08: reassess loose paired endpoints and explain the SDW jump

- User observed growing SDW competing with d-wave pairing in multiple loose
  histories, including the V=0 six-seed test, and proposed representative
  uniform d-wave and stripe CDW/SDW starts across the full grid. User regards
  `(1.4,-0.4)` as the strongest paired-fixed-point candidate while its
  strong-stripe energetic comparison remains pending. Recorded these as
  observations and a proposed next direction, not established phase labels
  or authorization for a new solver/campaign implementation.
- Added `scripts/analyze_square_basin_stability.py` and durable evidence in
  `docs/reports/square_grid_20260908/BASIN_ASSESSMENT.md`,
  `basin_seed_summary.csv`, `sdw_jump_history.csv`, `anderson_replay.json`,
  `six_seed_sdw_growth.png`, and `sdw_jump.png`. All 17 modern candidate
  compact-state hashes match the prior selection record. Raw applied inputs
  match their preceding saved measured outputs. Legacy coverage points are
  not given a new stability certification.
- Three V=0 seeds stop after six saved map evaluations. The three longer
  seeds have coherent late raw spin growth (projection gains about 1.096,
  input/output cosine >0.9998), followed by Anderson suppression. This revises
  the earlier apparent-basin-collapse interpretation: the existing global
  convergence screen alone does not establish a stable paired basin.
- Selected V=0 `stripe_pairing_m004` source compact hash:
  `9e7e1680ad0891495df6916908d7230b7e85d065dec96ef4aace701e9d3b20c6`;
  recorded full hash:
  `5c37b0667a1dd0bfedefd7963a100d5954dbdb11aafff8ea00f1f61c360dbe59`.
  Plotted iterations 23-to-24 are stored 22-to-23 because the seed is plotted
  at 1. First Anderson combines stored inputs 21/22 at damping 0.5 using
  coefficients `+22.6810987947,-21.6810987947`. Replaying this transition gives
  maximum absolute input error `1.57e-15`; all four saved mixed transitions
  match within `1.58e-15`. Bulk spin RMS falls 76.7%; its spatial profile also
  changes (cosine -0.3394), while the middle-rung trace barely moves. The
  corrected canonical energy rises `2.72e-7 t/site` across this step.
- Terminal global relative residual `0.0018425` coexists with spin-only
  relative residual `0.0374755`. Period-one acceptance may occur before the
  configured initial 20-step raw probe finishes, and Anderson-accepted fixed
  points have no mandatory subsequent raw stability check. The terminal
  profile's spin projection gain is 0.9918; earlier growing profiles do not
  prove its transverse instability. A controlled competing-order perturbation
  above the numerical floor is needed. These gains are trajectory diagnostics,
  not calculated Jacobian eigenvalues or energy curvatures.
- At `(1.4,-0.4)`, four small stripe/coexistence seeds instead show coherent
  spin decay (projection gains 0.60–0.61). This supports paired attraction for
  tested perturbations without ruling out a distinct lower-energy strong
  stripe. At `(1.0,-0.2)`, pairing decays and a strong stripe settles; the
  `(1.0,-0.4)` striped trajectory remains diverging as previously documented.
  Short weak-spin t0=1.2 histories remain inconclusive for basin stability.
- Recommended first qualifying channel diagnostics and raw perturbed checks
  on the two t0=1.4 anchors, then filling paired/striped branch slots across
  the grid with matched model/numerical fingerprints, target-coupling seed
  transformations, and corrected canonical energies with resolved errors.
  Allow coexistence and reserve extra chi/length/wavelength checks for close
  competitions. The four prepared length-study seeds remain available.
- Local reproduction:
  `python -B -X utf8 ladder_mps_mft/scripts/analyze_square_basin_stability.py`.
  Final run completed in 2.94 seconds with hash, raw-closure, and Anderson
  replay assertions passing; both PNGs visually inspected. An earlier
  verbose stdout preview piped to `Select-Object -First` closed the pipe early;
  reduced stdout and reran successfully. Updated current state, documentation
  index, report introduction, and active plan. This append-only run log is
  the requested progress ledger. No source-state/bundle/acceptance changes,
  new DMRG, expensive suite, solver/config changes, Perlmutter access,
  transfers, scheduler actions, reservations, or accounting-ledger changes.

## 2026-09-08: clarify Anderson timing and its physical interpretation

- User asked why Anderson starts precisely at the selected V=0 jump and how
  this can be consistent with a growing order. Checked the archived run config
  and local `Solver.jl`, `Mixing.jl`, and `Convergence.jl` control flow against
  the already-replayed saved update modes.
- The initial evaluation is stored record 1, followed by 20 raw probe records
  2–21. Probe completion clears mixer history. One input/output pair gives a
  linear startup at stored 22; two pairs enable Anderson at stored 23, plotted
  24. There is no channel-growth check governing this switch.
- Added a scalar illustration to the basin assessment: `F(s)=1.1s` has a
  repelling zero, but ideal signed Anderson combinations can solve for that
  zero. Numerical root finding does not certify energy minimization; neither
  raw nor mixed SCF steps should be read as physical time evolution. Actual
  terminal transverse stability remains untested. Updated the current snapshot;
  no new phase claim, solver change, DMRG, or scheduler action. Validation was
  source/config review and elementary algebra; no expensive tests rerun.

## 2026-09-08: prepare the approved raw two-reference square campaign

- User approved the two-basin comparison and specified the converged legacy
  stripe at `(1.0,0.0)` and uniform d-wave state at `(1.4,-0.4)` as the two
  references, each perturbed by the other. User explicitly rejected further
  Anderson use, retained chi=200, requested tighter thresholds and many MF
  iterations, confirmed the matched canonical-energy comparison, and deferred
  higher chi, length, and stripe-wavelength work. No scheduler action was
  requested or performed from this workspace.
- Prepared 18 square starts: 99% primary plus 1% competing correlations,
  reconstructed through `mean_fields_from_correlations` at every target
  `(t0,V)`. The stripe source SHA is
  `ae6a3bfe76ca8f06f2396fd731b18bca8539e0b7ee68df016cc9156fdceeb074`;
  the explicit `pairing_dwave_m000_chi200_loose` reference SHA is
  `8a1cf2d64d2fbe0eb59521192b829cab43e19a4d7ac026519ea847f6ac0778b8`.
  The latter is the user-requested uniform-pairing lineage, rather than the
  nearly tied stripe-pairing lineage selected for the earlier Fourier grid.
  Legacy densities come from diagonal normal correlations. Both references
  retain their finite-boundary profiles and full retained bond structure.
- Created immutable-content correlation bundle
  `output/seed_previews/20260908_square_two_basin/references.h5`, 801747 bytes,
  SHA `e01a1ea7d6be813110870d26377db0df529d1584816c946af78584e1e1fbddc1`.
  Local 18-config/seed manifest is in the adjacent `grid/` directory.
  Derived seeds store both source hashes and epsilon, use target E_p/kernels,
  zero inactive onsite exchange through the standard constructor, and start
  fresh product MPS with common RNG 1404. These are explicit inherited field
  mixtures, not unbiased starts or inherited converged MPS.
- Added `configs/phase1_gpu_square_two_basin_chi200_raw.toml`: chi=200, up to
  500 evaluations, at least 50 before acceptance, five stable evaluations,
  absolute/relative field tolerances 1e-7/1e-4, density 1e-5, corrected-energy
  window 1e-8 t/site, and 20 DMRG sweeps with 1e-8 total sweep-energy tolerance.
  Identity and effective-energy consistency gates are 1e-8 t/site. The raw
  probe covers the full run; fallback is linear with min=damping=max=1 and
  adaptive=false. No Anderson or reduced damping can occur in this contract.
- Added opt-in `minimum_iterations`, `channel_residuals`, and
  `dmrg_sweep_energy_tol` controls. Six full-vector channels separate pairing,
  spin-even/spin-odd exchange, uniform/modulated charge, and spin Hartree.
  Channel gates include slow drift and an absolute floor; period-two recurrence
  remains phase-resolved. Minimum-window and inner-sweep evidence also gate
  orbit acceptance. Fixed-point energy stability now spans the stable window
  when the channel option is enabled. Historical defaults retain prior behavior;
  these controls change the implementation/numerical fingerprints for new work.
- Canonical and corrected canonical energy histories already existed. Added
  `history/channels/*` diagnostics and inner-sweep pass flags, a CSV/PNG/PDF
  energy-history exporter, and a grid comparison that reuses accepted-only
  canonical selection, requires state/manifest fingerprint agreement, and
  retains missing, unaccepted, incompatible, or unresolved cases. Seed ancestry
  is not used as the final phase label. The ten-times-energy-window/tolerance
  resolution screen is documented as a heuristic, not a rigorous error bound.
- Launcher v1.19.0 adds `prepare-square-two-basin-raw` with `anchors` (default,
  four jobs at t0=1.4,V=0/-0.4), `remainder` (14 disjoint starts), and `grid`
  (all 18, an alternative scope). Reuses existing guarded submission and the
  same shared live accounting ledger. One-GPU 12-hour first-segment ceilings
  are 12, 42, and 54 fractional node-hours respectively. No reservations or
  blanket continuations were created. Pending campaigns keep their old source.
- Added `docs/reports/two_basin_raw_20260908/README.md`, seed PNG/PDF and
  amplitude CSV, plus source-bundle tooling and a user-run isolated-snapshot
  handoff. Updated the project snapshot, active plan, convergence method, and
  documentation index. Existing square bundle, histories, flags, and source
  HDF5 artifacts remain unchanged.
- Validation: 31 existing convergence, 4 mixing, 8 DMRG-observer primitive,
  52 new convergence/storage/seed-algebra/comparison, 167 prepared-seed/startup/
  stage-partition, and 122 launcher assertions passed: **384 passes**. One
  Linux-only launcher integration test is skipped on Windows. Verified real
  inherited-field/fresh-MPS startup on CPU for all 18 starts, not GPU execution.
  Bash syntax passed using the detected local Git Bash executable. Both figures
  were inspected; energy export reproduced 31 stored records from two existing
  runs, including the older reconstructed density correction. No DMRG ran.
- Focused commands used the installed Julia 1.12.7 executable with
  `--startup-file=no --compiled-modules=existing --project=ladder_mps_mft`.
  Permanent focused tests are `test/test_raw_basin.jl`, also included in the
  normal suite. Temporary Julia harnesses were removed. Initial checks exposed
  a test fixture using the old loose orbit tolerance, a CPU harness initially
  retaining backend=gpu, and a missing explicit import in the extracted launcher
  test block; corrected the fixtures/harnesses and reran only affected checks.
  Fixed the plotting adapter's charge-profile key and handled old energy
  histories without a saved correction array. No expensive full suite rerun.

- Packaged the separate source snapshot as
  `output/source_bundles/two_basin_raw_20260908.zip`: 121 manifested files,
  1417719 bytes, SHA-256
  `ea167c75fecc9e64eb4e4d6ea7e5d77c1002847d0b8843f2c1e8e65af4589452`.
  Every archived member was read back and checked against its SHA-256; shell
  scripts use LF endings, machine-local preferences are excluded, and the
  reference HDF5 is included. Source manifest and checksum sidecars accompany
  the ZIP. Final `git diff --check` passed. Unrelated concurrent edits in
  `docs/METHODS_NOTES.tex` and `docs/literature/SOURCE_NOTES.md` were preserved.

## 2026-09-08: record the 2023 benchmark geometry for the manuscript

- At the user's request, recorded the equation-based correspondence between
  Bollmark et al., Phys. Rev. X 13, 011039 (2023), published Eqs. (51)-(58),
  and `cubic_frustrated` in `src/MeanField.jl`.
- Added a dated manuscript note to `docs/literature/SOURCE_NOTES.md`, a cited
  paragraph and revision entry to `docs/METHODS_NOTES.tex`, and a current-state
  pointer. Preserved the distinction between project terminology and the
  authors' terminology, and identified which geometry their Fig. 13 uses.
- Validation: compared the published equations with the implemented pairing
  and exchange coefficients/index orientations; reviewed the documentation
  diff and citation target. Documentation only; no solver changes, DMRG, or
  Perlmutter actions. The existing literature-review PDF was not modified.

## 2026-09-08: revise the two-basin seeds to 95%/5% and cap at 80 evaluations

- User revision: use 95% of the primary reference correlations plus 5% of the
  competing reference, and at most 80 MF evaluations to limit compute. Updated
  the preparer, seed provenance, base config, raw-probe/stagnation windows,
  seed contract, plot labels, active plan, project snapshot, and handoff.
  chi=200, at least 50 evaluations before acceptance, five stable records,
  tighter channel/energy/inner-DMRG gates, per-evaluation energy histories,
  and raw updates without Anderson remain the approved numerical contract.
- Regenerated all 18 seeds/configs and paired fingerprint manifests under
  `output/seed_previews/20260908_square_two_basin/eps005_iter80/grid/`, using
  the same hash-checked `references.h5`. Each correlation component is exactly
  the requested convex combination; fields are rebuilt using target couplings.
  The earlier 18 seed/config hashes and original source-ZIP hash were verified
  unchanged. Refreshed the seed PNG/PDF and amplitude CSV in the campaign report.
- The first stage remains four anchors, followed by the fourteen disjoint
  starts after review. New planned run ID:
  `20260908_square_two_basin_raw_eps005_iter80_anchors`. The 12-hour Slurm ceiling
  and shared budget guards are unchanged; the lower iteration cap reduces
  allowed work, not the reserved ceiling. No automatic continuations, jobs,
  transfers, scheduler actions, or reservations were performed.
- Local preparation command, from the repository root (Julia 1.12.7):

  ```powershell
  & 'C:/Users/Kevin/.julia/juliaup/julia-1.12.7+0.x64.w64.mingw32/bin/julia.exe' --startup-file=no --compiled-modules=existing --project=ladder_mps_mft ladder_mps_mft/scripts/prepare_phase1_two_basin_grid.jl ladder_mps_mft/configs/phase1_gpu_square_two_basin_chi200_raw.toml ladder_mps_mft/output/seed_previews/20260908_square_two_basin/references.h5 ladder_mps_mft/output/seed_previews/20260908_square_two_basin/eps005_iter80/grid ladder_mps_mft/output/seed_previews/20260908_square_two_basin/eps005_iter80/full_preview 20260908_square_two_basin_raw_eps005_iter80_grid grid
  ```

- Focused validation: ran `test/test_raw_basin.jl` with the same Julia flags:
  **52 assertions passed** (30.0 seconds in the test body). A read-only Python
  check verified all 18 exact 95%/5% mixtures, config/seed hashes, 80-evaluation
  caps, minimum window, chi, raw update controls, seed provenance, and paired
  fingerprints. Regenerated the figure with
  `python -B -X utf8 ladder_mps_mft/scripts/plot_two_basin_seeds.py` and visually
  inspected it. Validation is local configuration, algebra, storage, and unit
  checking; no DMRG or scientific convergence test ran. No full suite rerun.
- The revised source ZIP uses the new name
  `output/source_bundles/two_basin_raw_20260908_eps005_iter80.zip`; the handoff
  uses a matching fresh snapshot directory. Bundle command:
  `python -B -X utf8 ladder_mps_mft/scripts/bundle_two_basin_source.py`.
- Revised ZIP: 121 manifested files, 1418414 bytes, SHA-256
  `0337f9881a406b60b20110e85f13994d14e4df6f6feab83a6dc0835c271ae9e7`.
  Archive integrity and every member hash passed; checksum and manifest
  sidecars accompany it. The ZIP contains this revision entry up to bundle
  creation; this checksum is appended afterward. `git diff --check` passed.

## 2026-09-08: simple two-basin submission from the Git checkout

- User reports canceling queued runs and requests a simple submission after
  `git pull` on Perlmutter. No job IDs or accounting were supplied; this is a
  user-reported cancellation, not a locally verified scheduler transition.
- Added `slurm/submit_square_two_basin.sh`. With no arguments it prepares and
  submits four anchors under `20260908_square_two_basin_95_5_80_anchors`.
  Optional arguments select a fresh run ID and `anchors`, `remainder`, or
  `grid`. It uses the checkout containing the script, resets earlier snapshot
  source/config overrides, and preserves explicit shared run/scratch/ledger
  paths. Existing run IDs are refused. A nonempty existing reservation ledger
  is reconciled through the original `sacct`-based command before submission;
  missing or header-only ledgers skip reconciliation. Failures stop the wrapper.
  Existing GPU-preference, duplicate-branch, and budget gates remain in force.
- Versioned the 801747-byte correlation-only reference input at
  `data/two_basin_references.h5`, with a narrow `.gitignore` exception. Its
  SHA-256 is unchanged:
  `e01a1ea7d6be813110870d26377db0df529d1584816c946af78584e1e1fbddc1`.
  This removes the source-ZIP transfer and snapshot-setup dependency. The old
  snapshots and derived data remain intact. Added an LF checkout rule for the
  new shell script and documented the input in `data/README.md`.
- Numerical controls and solver source are unchanged: 95%/5% mixtures,
  chi=200, 80 evaluations maximum, minimum 50, no Anderson, and per-iteration
  energy histories. The first four anchors still precede the other fourteen.
- Local validation: Bash syntax passed; 22 wrapper-dispatch checks passed in
  an isolated checkout with a mock launcher, including paths with spaces,
  stale source overrides, explicit ledger preservation, default/alternate
  stages, duplicate run refusal, invalid arguments, accounting failure, and
  missing/header-only ledgers. An initial check identified the missing-ledger
  edge case; added the guard and reran only the wrapper checks.
- The real Julia preparer loaded the versioned input and generated four
  native anchor seeds/configs under
  `output/seed_previews/20260908_square_two_basin/git_checkout_anchors/`.
  Verified coordinates, 95%/5% provenance, chi=200, the 80-evaluation cap,
  reference/seed/config hashes, and the preparer's target-kernel/readback/
  fingerprint checks. Used Julia 1.12.7 with
  `--startup-file=no --compiled-modules=existing --project=ladder_mps_mft`
  and `scripts/prepare_phase1_two_basin_grid.jl` with stage `anchors`.
  Validation was local preparation and mocked dispatch; no DMRG, GPU,
  actual Slurm, NERSC connection, transfer, or reservation was performed.
  No unchanged solver tests or full suite were rerun.
- Updated the project snapshot, active plan, and campaign handoff. Perlmutter
  commands for the user, after this change is pushed to the current branch:

  ```bash
  cd "$CFS/m4863/MPS-MFT/ladder_mps_mft"
  git pull --ff-only
  bash slurm/submit_square_two_basin.sh
  ```
- Git-index checks confirmed the exact reference SHA-256 and LF bytes for both
  shell scripts; staged `git diff --check` passed. Committed locally as
  `Simplify two-basin submission` on `codex/mps-mft-phase0-refactor`.
  The subsequent push to the configured GitHub origin `kwang0/MPS-MFT` was
  rejected by automatic approval review before execution: the review requires
  explicit authorization to export the reference HDF5, configuration, and
  project documentation to that destination. The remote remains unpublished
  by this task pending that approval; no alternative transfer was attempted.

## 2026-09-08: user reports anchor submission; remainder waits for analysis

- The user reports submitting the four-anchor campaign. Job IDs, exact run ID,
  scheduler state, results, and reconciled accounting were not supplied; record
  this as user-reported submission. The wrapper's default run ID is
  `20260908_square_two_basin_95_5_80_anchors`.
- Confirmed from `slurm/submit_square_two_basin.sh` that the default scope is
  only four anchors: both 95%/5% seed families at `(t0,V)=(1.4,0)` and
  `(1.4,-0.4)`. No dependency or automatic continuation launches the remaining
  fourteen. Those require a separate explicit `remainder` submission.
- Next scientific step: inspect full spatial/channel histories for competing
  order growth and convergence, corrected canonical energy histories, and
  sufficiency of the 80-evaluation limit before deciding on the remainder.
  Updated the project snapshot and active plan. Documentation and launcher
  inspection only; no solver change, tests, remote connection, or scheduler
  operation was performed.

## 2026-09-10: V=-0.4 anchors reach a common pairing basin; acceptance is noise-limited

- User supplied status for the four-anchor campaign and synchronized the two
  V=-0.4 terminal histories. Jobs 58093799 (stripe) and 58093800 (pairing) are
  COMPLETED; each HDF5/log contains all 80 evaluations, with `maximum_iterations`
  and accepted=false. V=0 jobs 58093802/58093803 remain PENDING in that output.
- Analyzed the full applied/measured spatial histories, stored channel gates,
  physical anomalous correlators, corrected canonical energies, and timings.
  Both lineages reach the same d-wave-like paired basin at L=64, chi=200.
  Bulk leg-odd spin field RMS drops from 1.9902e-2/1.0475e-3 in the seeds to
  3.8782e-8/2.4933e-8 at iteration 80 (stripe/pairing). It reaches below 1e-7
  at iterations 26/20. Final physical leg/rung pair amplitudes have opposite
  signs and overlap, as do the charge profiles. The full pair matrices differ
  by 5.8521e-5 relative L2; terminal corrected energies differ by only
  7.4853e-9 t/site. This is trajectory agreement, not accepted-energy ranking.
- Final global field, slow-mode, density, inner-DMRG, pairing-channel, and
  identity checks pass in both runs. Tiny spin/spin-exchange residual maxima
  around 1e-7–4e-7 exceed the 1e-7 absolute floor. Uniform-charge changes below
  1.62e-8 in every last-20 record sometimes acquire infinite extrapolation
  factors: the new channel gate uses the nonzero background amplitude for its
  floor and interprets same-sign scalar fluctuations as coherent growth.
  At stripe iteration 80, residual 1.0581e-9 with ratio 3.37484 fails this way.
  The pairing-seeded run also marginally misses charge modulation and the
  1e-8 t/site energy window (observed 1.1346e-8). No thresholds were changed.
- The user supplied allocation-level sacct data after a focused request:
  elapsed seconds 30799 and 23390, both COMPLETED. At one quarter node these
  cost 2.138819444 and 1.624305556 node-hours, **3.763125 total**. Saved MF time
  alone totals 3.714447030 node-hours. The finished pair reserved 6, leaving
  2.236875 eligible for release through normal reconciliation; all four anchors
  reserved 12. The synced reconciliation ledger has no rows for these jobs.
  The pasted 61.883819445 active project total includes other reservations and
  is not the cost of this point. No live or local accounting ledger was edited.
- Added `scripts/analyze_two_basin_vm04.py` and
  `docs/reports/two_basin_vm04_20260910/`: narrative, PNG/PDF figure, analysis
  JSON, 160 energy/profile records, 960 channel records, and the user-supplied
  sacct receipt. Updated the current state, active plan, and documentation map.
  Recommendation: do not extend this pair unchanged solely for acceptance;
  qualify noise-aware channel gates against V=0 before the remaining fourteen.
- Validation: compact-state/config/inherited-seed hashes agree with synced
  manifests and state provenance; all four comparison fingerprints match.
  Initial applied fields equal the 95%/5% seeds; the next 79 inputs exactly
  equal previous raw outputs. Recomputed channel absolute/relative residuals
  from full vectors; checked log/HDF5 record counts, finite iteration energies,
  output CSV counts, and sacct timestamp differences/exact decimal charge.
  Visually inspected the scientific figure. Initial analysis assertion assumed
  all update labels were `unmixed_probe`; corrected it to one `initial` followed
  by 79 `unmixed_probe` labels. This was an analysis-harness correction.
- Compact terminal SHA-256: stripe
  `641277827094e384dbd6bede51334d20c4e00dd86d83d430a7ec8614af5e5f79`;
  pairing `026b622a94291f03684a287c39f8ec7306dda36e160a72da138ef2158d91e90a`.
  Full-source hashes are retained in `analysis.json` and stateless manifests.
  Reproduce locally with
  `python -B -X utf8 ladder_mps_mft/scripts/analyze_two_basin_vm04.py`.
  No DMRG, solver changes, source-state relabeling, remote connection, scheduler
  action, submission, or continuation occurred. Unrelated manuscript edits
  were preserved.

## 2026-09-10 — Show the beginning of the two-basin energy histories

- At the user's request, extended `scripts/analyze_two_basin_vm04.py` to save
  `docs/reports/two_basin_vm04_20260910/energy_convergence.png` and `.pdf`:
  all 80 evaluations, a first-15-evaluation close-up on the same absolute
  energy scale, and the existing late-time scale expanded to 1e-9 t/site.
  Added the figure and early-transient description to the report. Iteration 1
  is the first MF evaluation, not a separate seed-energy measurement.
- Reused the same immutable states for jobs 58093799/58093800 and the hashes
  recorded in the preceding analysis entry. The stripe-seeded energy reaches
  the common plateau in roughly ten evaluations; the pairing-seeded energy
  is already close after two. Energy settling alone is not field convergence.
- Local validation: ran
  `python -B -X utf8 ladder_mps_mft/scripts/analyze_two_basin_vm04.py` (about
  four seconds), preserving the 160 finite stored energy records, and visually
  inspected the new figure. No DMRG, acceptance changes, or remote actions.

## 2026-09-10 — Authorize and prepare the 40-evaluation two-basin remainder

- The user authorizes the other fourteen 95%/5% chi=200 starts now, without
  waiting for the remaining anchors, and asks for a 40-evaluation cap and
  minimally relaxed convergence checks that still reject growing order.
  The existing job-specific output identifies the pending anchors as
  `(t0,V)=(1.4,0.0)`, jobs 58093802/58093803. The latest prose says `(1.0,0.0)`;
  asked whether this is a separate submission, since that coordinate is in
  the remainder. Prepared the existing seven-point/fourteen-branch scope;
  no cancellation, replacement, or live submission was performed.
- Added the separate `phase1_gpu_square_two_basin_chi200_raw40.toml` base:
  maximum/probe 40, minimum 30, stable window 10, channel noise floor 5e-7 t,
  energy range 2e-8 t/site. Global absolute/relative field gates remain
  1e-7/1e-4; density, inner DMRG, identity, eigenvalue, and period-two
  recurrence controls are unchanged. The original 80-step base is retained.
- Added opt-in `channel_noise_floor` (zero preserves historical channel
  behavior). Two-step channel extrapolation now requires both residuals
  above the enabled floor, so large uniform-charge background does not
  amplify tiny scalar jitter. Above-floor coherent growth still fails.
  Added a componentwise full-profile span check across all applied/measured
  fields in the stable window: max span <= floor OR L2 span / largest
  profile norm <= relative tolerance. This rejects accumulated sub-floor
  drift and intermediate excursions. Saved window diagnostics accompany
  per-step channel diagnostics. Distinct period-two phases retain their
  existing orbit checks. The setting enters numerical fingerprints.
- Replayed the actual Julia field/channel/energy/density/inner-sweep gates
  on the two synced 80-step anchors and the six older V=0 raw prefixes.
  Available history gates first pass at iteration 35 for the stripe seed
  and 30 for pairing. All six older V=0 raw prefixes fail; the three with
  long coherent SDW growth fail spin gates independently of the minimum.
  Per-iteration identity/eigenvalue errors are not saved, so the replay
  cannot certify retrospective acceptance at 30/35. Both terminal identity
  checks pass at 80. All eight source hashes checked before/after replay;
  flags and states are unchanged. Evidence is in
  `docs/reports/two_basin_remainder_20260910/` and reproduced by
  `scripts/replay_two_basin_convergence.jl`.
- Prepared all fourteen seeds/configs in
  `output/seed_previews/20260910_square_two_basin40/final/remainder/` using
  `scripts/prepare_phase1_two_basin_grid.jl` with the new base and the existing
  reference bundle. Counts are two starts at each of the seven non-anchor
  coordinates; reference/derived-seed hashes, readback, target E_p, nonzero
  competing channels, and within-point fingerprints pass. Contract metadata
  now derives iteration values from the base instead of hard-coding 80/50.
- Added `slurm/submit_square_two_basin_remainder.sh`. Pending anchors load
  their source from the original checkout when they start; the handoff uses
  a separate Git worktree. The new wrapper reads original anchor `run.env`
  to share exact account, run/scratch roots, and budget/reconciliation ledgers,
  then selects only the new source/config. It refuses the original source
  directory, prepares only `remainder`, reconciles, and submits through the
  existing cap/duplicate guards. It does not modify the pending jobs.
- Budget: at most 560 MF evaluations for the fourteen new starts. The existing
  12-hour one-GPU ceiling still reserves at most 42 fractional node-hours;
  actual cost may be lower. Solver deadline remains 11.5 h. No automatic
  continuations or new ledger reservations were created locally.
- Validation: focused `test/test_raw_basin.jl` passed 52 existing + 23 new
  assertions (about 31 seconds of reported test execution, plus Julia load).
  Includes small spin jitter, uniform charge jitter with a growth ratio >1,
  accumulated weak drift, resolved spin growth, minimum/window gates,
  energy excursions, failed DMRG, physical period two, and saved diagnostics.
  `python -B -m unittest discover -s ladder_mps_mft/test -p
  test_two_basin_remainder_launcher.py -v` passed its local fake-launcher
  test, confirming exact scope/order, shared accounting, source separation,
  and refusal in the original checkout. Bash syntax and `git diff --check`
  passed. No DMRG/full-suite rerun or remote operation was needed.
- Initial Julia cache writes were denied by the local sandbox; using
  `--compiled-modules=existing` completed validation without escalation.
  Corrected two replay-script header/entry-point parsing mistakes before
  the successful replay. Refreshed final fingerprints and preparation after
  a source-comment clarification; unchanged solver tests were not repeated.
  The dated contract records source/config hashes and handoff commands.
  Updated current state, method notes, and active plan; unrelated manuscript
  and root `.claude/` changes are preserved.
- Publication remains pending: an earlier push to `origin`
  (`https://github.com/kwang0/MPS-MFT.git`) was rejected by automatic approval
  review for lack of explicit export authorization. No new push was attempted;
  the validated changes are concrete and ready for the user's publication
  decision. Local config SHA-256 is
  `71a00fa3b69d84a22f7676b5b4b16c604516f27f99cdf6a3fe04bab4e7edc35d`;
  local CPU implementation SHA-256 is
  `a29d3f9f0ed848454f66d02ff787e157127273171191173bb8e33893edcf2550`.
  GPU/native-path fingerprints are generated by preparation on Perlmutter.

## 2026-09-10 — User authorizes publication and requests submission commands

- The user explicitly authorizes pushing the validated revision to the
  configured `kwang0/MPS-MFT` origin. This resolves the earlier export-approval
  block. Commit scope includes the two-basin analysis, corrected convergence
  controls, focused tests, fourteen-start launcher, and project documentation;
  unrelated manuscript and root `.claude/` changes remain outside the commit.
- Handoff uses `git fetch` and a separate detached worktree on Perlmutter so
  the original pending anchors continue to load their submitted source.
  The user performs the fetch/worktree creation and all scheduler actions.
  The launcher retains shared run/scratch/accounting roots and submits only
  `20260910_square_two_basin_95_5_40_remainder`. No new solver change or
  repeated DMRG/test run is needed for publication.
- Publication completed: commit `7443d9d` (`Qualify 40-step two-basin runs`)
  was pushed normally to `origin/codex/mps-mft-phase0-refactor`; Git confirmed
  the remote update `0729d52..7443d9d` with exit status zero. Staged whitespace
  checks passed and only the intended 31 files were committed. The sandbox
  required escalated Git index access; automatic review allowed staging,
  commit, and push under the user's explicit authorization. Updated the
  current snapshot and handoff to record publication; no NERSC connection,
  transfer, scheduler query, reservation, or submission occurred.

## 2026-09-12 — V=0 anchors show stripe growth and loss of pairing

- The user identifies `(t0,V)=(1.4,0.0)` as completed and asks for the same
  analysis as V=-0.4. Synced states/logs confirm stripe job 58093802 ended
  `maximum_iterations` at 80 evaluations, pairing job 58093803 ended
  `time_limit` at 62. Both accepted=false, period=0, with one `initial`
  followed by raw `unmixed_probe` records. Every next applied field equals
  the preceding measured field exactly. The jobs retained the original
  80-step/50-minimum controls and implementation; no Anderson occurred.
- Stripe start: leg-pairing MF bulk RMS falls from 1.0782e-4 to 1.7462e-10 t;
  leg-odd spin MF bulk RMS ends at 0.0241366 t. Pairing start: pairing falls
  from 0.00450157 to 0.000709038 t (15.75% retained); spin grows from
  0.00181611 to 0.0225097 t (12.39x). In its last five records, pairing
  drops 65.6% and spin grows 12.0%. This supports a pairing-dominated
  transient evolving toward a similar stripe texture, not a settled paired
  or coexistence endpoint. The conclusion is visible before deadline record
  62, which additionally misses the density target by 2.4418e-5.
- Both endpoints have dominant full-L charge DFT mode m=4 (q/pi=0.125),
  and physical leg-odd spin mode m=30 (q/pi=0.9375). Bulk spin profiles
  still differ by 26.4% relative L2; they cannot be called the same converged
  state. RMS uses rungs 6–59. Physical spin uses (n_up-n_down)/2, separately
  from coupling-weighted MF fields; staggered profile figures expose its
  envelope. The full scalar and selected spatial histories are plotted.
- Terminal corrected energies/site are -0.661122033082 and -0.661032086408;
  separation 8.9947e-5 t/site is an unfinished-trajectory diagnostic, not
  an accepted-state energy ranking. Stripe last-five energy span is
  1.3237e-8; pairing span is 4.5738e-5. The pronounced stripe startup dip
  at evaluation 2 is not a selectable low-energy solution: the stored
  functional uses applied partner fields and current correlations away from
  self-consistency. The full energy figure includes iterations 1 onward.
- Global relative residuals remain 5.8755e-4 and 2.6513e-2; extrapolated
  values are 0.06242 and 0.18531. Stripe spin/charge residual directions
  are highly coherent, with contraction estimates near 0.990–0.993, so
  the nearly flat energy masks slow spatial relaxation. Spin residual
  maxima are 4.4907e-5 and 2.0140e-3 t, about 90x/4028x the revised channel
  floor. Stripe inner-DMRG gate passes only 2/5 recent records; pairing
  passes 5/5 but misses density. Both endpoint identity checks are below
  3e-11 t/site. Neither saved history contains even one passing global
  field record under the unchanged 1e-7 absolute OR 1e-4 relative gate;
  therefore neither would pass the revised ten-record fixed-point window.
  No expensive full replay is necessary for this necessary-gate conclusion.
- At 40, the pairing start retains 85.6% pairing while spin has grown
  almost sevenfold; a 40-step cap can leave this point unresolved without
  falsely accepting it. Keep the revised thresholds. Targeted continuation,
  especially of the pairing start, is a future decision after accounting;
  no new controls, continuation, or compute was prepared by this analysis.
- Recorded MF times are 22189.828914 and 41341.377641 seconds: estimated
  1.540960 and 2.870929 fractional node-hours, 4.411889 total. This excludes
  allocation overhead. The synced ledger has only the two 3-hour reservations;
  requested user-supplied `sacct -n -X -j 58093802,58093803
  --format=JobIDRaw,State,ElapsedRaw,Start,End -P` for exact allocation cost.
  No live scheduler query or accounting-ledger edit was performed.
- Added `scripts/analyze_two_basin_v000.py` and the dated report directory
  `docs/reports/two_basin_v000_20260912/`: narrative, three PNG/PDF figures,
  JSON, two source records, 142 energy and scalar rows, 852 channel rows,
  and 9088 rung/iteration rows. Generalized the existing V=-0.4 loader only
  for point selection, variable record counts, and absent threshold crossings.
  Updated the project snapshot, active plan, and documentation index.
- Local validation: compact/config/seed hashes match manifests and state
  provenance; all four comparison fingerprints match. Recomputed six-channel
  residuals from full vectors, verified raw chain and log/HDF5 counts,
  independently reconstructed target-density-corrected energies from saved
  canonical energy/mu/density, and rehashed both sources after analysis.
  The script completed in about five seconds; all three figures were viewed.
  Both original V=-0.4 loader summaries still match their prior JSON exactly.
  Output row counts and `git diff --check` pass. No DMRG, solver changes,
  remote connection, source relabeling, or unrelated manuscript edits.
- Compact state SHA-256: stripe
  `ba38078fd047653251f321d14995e89acc2c63d05cdba8e1e41fac6b2a3b5bb0`;
  pairing `8d18a149ef1b6707f2377b35f9b5ed9ada97f6a287d693b09ede6b0c7e0c5ab9`.
  Full-source hashes and matching original numerical/implementation/E_p
  fingerprints are preserved in `analysis.json`. Reproduce locally with
  `python -B -X utf8 ladder_mps_mft/scripts/analyze_two_basin_v000.py`.

### 2026-09-13 — Explain the visually settled V=0 stripe's tolerance failure

- Followed up on the (t0,V)=(1.4,0.0) stripe-seeded endpoint using the
  existing immutable 80-record history. Its final spin residual is almost
  entirely a profile change: projection onto the applied field accounts
  for only 2.066e-6 of residual power. About 99.0% of spin residual power
  is in rungs 6–59; the largest changes are around rungs 39–40.
- Linear interpolation of the staggered spin MF zero crossings shows
  unequal wall shifts. Between evaluations 60 and 80 the third crossing
  moves 39.3999 -> 39.5165 and fourth 55.1830 -> 55.2673, while the bulk
  spin RMS changes only 0.0528% across the final twenty saved values.
  This diagnoses slow texture rearrangement hidden by an amplitude plot;
  it is not a fitted physical translation or a new phase identification.
- Direct global residual 5.8755e-4 still exceeds 1e-4. The separate
  stopping-only slow-mode estimate has lambda=0.9905866 and residual
  cosine=0.9998334, giving factor 106.23 and a heuristic 6.24% remaining
  relative motion if this decay persists. It is not a certified error
  bound, physical eigenvalue, or Anderson update. Additional original
  blockers remain the marginal energy span and 2/5 inner-DMRG passes.
- Interpretation: stripe basin established, strict spatial self-consistency
  pending. Coherent residuals greatly exceed the revised channel floor;
  finite-chi bias and ultimate asymptotic decay remain unseparated. No
  tolerance change, solver run, remote action, or acceptance relabeling.
- Added residual geometry and zero-crossing diagnostics to the existing
  V=0 analysis script/JSON and a September 13 subsection in its report.
  The roughly five-second local rerun passed the existing provenance,
  raw-chain, energy, and source-hash checks plus an orthogonal-decomposition
  consistency check. Plot definitions and scientific conclusions from
  the September 12 analysis remain unchanged.

### 2026-09-13 — Verify the existing Phase 1 plotter with the raw V=0 anchors

- Confirmed `plot_phase1_mf_observables.jl` loads both new compact state
  files without changes. Explicitly select campaign
  `20260908_square_two_basin_95_5_80_anchors`; its default still points to
  `20260823_phase1_gpu_v2`. Accepted=false does not suppress the plots.
- Local Julia 1.12.7 check used the existing plotting environment,
  `--startup-file=no --compiled-modules=existing`, and a temporary script
  with the Agg backend. Verified the full measured-history shapes, all five
  history rows, slider endpoints, and saved PNGs for stripe/pairing: 81/63
  samples including the seed at plotted iteration 1. Both checks passed;
  both images were viewed, and the temporary script was removed.
- PNGs are in the existing campaign's
  `plots/mf_profiles/profiles_and_saved_histories/`. Added interactive Julia
  commands and the `include_seed=false` numbering option to the V=0 report.
  No plotter/solver changes, DMRG, source-state edits, or remote actions.

### 2026-09-15 - Create an introduction/background and current-results draft

- Created docs/manuscript/introduction_and_results.tex and its compiled PDF
  at the user's request. The manuscript-style introduction and background
  connect Mott physics, stripe interpretations, local ladder pairing,
  interladder coherence, extended interactions, and the multichannel model.
- The dated results collection covers the isolated-ladder backbone,
  covariance/response pilot, corrected charge-gap interpretation, revision
  of the early V=0 paired-basin claim, V=-0.4 and V=0 raw anchors, static
  stripe wavevectors, numerical wall relaxation, and the qualified chi=400
  energy comparison. It distinguishes local pairing from anomalous order,
  iteration from physical dynamics, and provisional textures from
  thermodynamic conclusions. Larger-length seeds remain preparation only.
- Reused 29 references from the existing maintained bibliography and added
  five draft-specific entries in additional_references.bib: Tranquada2004,
  Vojta2004, Li2007, Berg2007, and Missiaen2025. Primary author/publisher
  records were checked. The prior literature review and shared bibliography
  are unchanged. Source notes and eleven groups of local evidence identify
  the basis and limits of the synthesis.
- Reused the existing V=0 spatial-history figure without altering its data.
  Added local build instructions and links from the documentation map and
  project state. No new simulation, remote action, acceptance relabeling,
  scheduler operation, or accounting change was made.
- Validation: compiled with the existing repository-local Tectonic 0.17.0;
  downloaded missing standard TeX packages into its workspace cache with
  approved network access, then completed the final build from cache.
  The 16-page PDF has 34 resolved citations, no undefined cross-references,
  no overfull boxes, and no text outside checked margins. All sixteen local
  evidence links and the reused figure path resolve. Poppler rendered every
  page for visual inspection. Build/QA files are in the ignored
  output/manuscript_draft directory. No DMRG or unrelated test suite ran.

### 2026-09-15 - Make LaTeX paths portable between local builds and Overleaf

- The manuscript now detects the shared bibliography at the project root
  and selects root-relative or source-directory-relative bibliography and
  figure paths. The literature review uses the same conditional approach
  for its bibliography. Scientific text and bibliography data are unchanged.
- Added the exact Overleaf upload layout and main-document instructions to
  docs/manuscript/README.md. Prepared an ignored transfer snapshot at
  output/manuscript_draft/overleaf_upload.zip containing the two TeX files,
  two bibliography files, the required report figure, and the README.
  Local evidence hyperlinks remain local pointers, not bundled reports.
- Focused validation: ran the ignored check_portability.py helper using the
  bundled Python runtime and cached Tectonic 0.17.0. Both documents compiled
  in their own directories and through project-root input wrappers in an
  isolated copy of the upload layout. The wrappers prevent Tectonic's
  source-directory switch from testing the same path branch twice.
- Checked the actual BibTeX database paths in all four auxiliary files:
  34 manuscript references and 49 review references resolve in both layouts.
  All page text and page sizes match the existing 16-page and 29-page PDFs.
  ZIP integrity and the bytes of every packaged file match the local sources.
  Verification outputs are in output/manuscript_draft/portability/.
  Existing published PDFs were left unchanged. No live Overleaf session,
  numerical test, remote action, or scientific-status change was involved.

### 2026-09-15 - Make the Overleaf transfer package available through Git

- The manuscript sources, compiled PDF, bibliography, source notes, literature
  review, and required report figure were already tracked. The original upload
  ZIP was under ignored output/, so it was unavailable to a remote checkout.
- Added docs/manuscript/overleaf_upload.zip and a standard-library Python
  packaging helper beside it. Updated the README to link the tracked archive
  and explain regeneration. The archive preserves the checked Overleaf layout.
- Validation: the packaging helper checks ZIP integrity, member names, and
  byte-for-byte agreement of all six members with the maintained source files.
  This is a packaging-only change; the prior four compilation checks still
  cover the unchanged LaTeX sources. No numerical calculation was run.

### 2026-09-15 — Analyze the complete square two-basin grid

- User reports the full grid complete. Locally verified all 18 unique
  point/seed histories: four original anchors plus fourteen runs in
  `20260910_square_two_basin_95_5_40_remainder`, jobs 58172797–58172810.
  The fourteen all saved 40 evaluations; original anchors saved 80, 80,
  80 and 62. Total 862 evaluations, 17 maximum_iterations and one time_limit,
  zero accepted endpoints and no recorded periodic solution. No remote
  access or scheduler action was performed.
- Preliminary diagram: both seeds reach stripe CDW/SDW at all six
  t0=1.0/1.2 points, and paired states at t0=1.4,V=-0.4/-0.2. The seventh
  stripe assignment, t0=1.4,V=0, is starred because its pairing start is
  still converting at the deadline. This is a basin/trajectory assignment,
  not an accepted-energy ranking or an interpolated phase boundary.
- At the new paired (1.4,-0.2) point, physical leg-pair RMS is 0.0335676
  from both seeds, rung means are -0.0501517, and spin RMS is only
  1.16e-6/4.57e-6. Endpoint corrected energies agree within 2.1235e-9 t/site.
  Pairing-start acceptance fails only the terminal ten-record spin span
  (1.0742e-6 versus 5e-7 t); the final spin update is already 3.486e-8 t.
  Stripe-start spin still decays, and its earlier global-field records and
  several channel spans also fail. No growing instability is inferred.
- At (1.2,-0.4), the pairing start has an extended paired transient:
  leg-pair MF RMS 0.00203 at 10, 0.00195 at 20, 0.000639 at 30, and
  7.83e-8 at 40. Final-ten pairing falls 99.976% while spin grows 13.6%.
  Its late energy span remains 1.558e-4 t/site. Other stripe points have
  negligible pairing but persistent profile relaxation; (1.0,-0.4) has
  particularly resolved ~0.0034 global relative residuals.
- All twelve t0<=1.2 endpoints have physical spin RMS 0.175–0.261,
  leg-pair RMS <=9.85e-7, full-L charge DFT m=4 and spin DFT m=30.
  The two paired points have physical leg-pair RMS 0.0336–0.0352 and
  opposite rung signs. Finite-boundary modulation and tiny channel noise
  are not treated as additional ordered phases. Descriptive phase labels
  are unchanged under a tenfold variation of the amplitude separators.
- All 18 jobs now have user-synced terminal sacct reconciliations. Actual
  allocation charges: original V=-0.4 pair 3.763125, original V=0 pair
  4.444722, fourteen remaining starts 18.590486, total 26.798333 node-hours.
  Summed saved solver time estimates 26.474287. The V=0 actual charge
  supersedes its prior estimate; the accounting ledger remains unchanged.
- Added `scripts/analyze_two_basin_grid.py` and
  `docs/reports/two_basin_grid_20260915/`: phase diagram, spin/pairing
  history grids, full/late energy grids (five PNG/PDF pairs), narrative,
  full JSON, 18 source/summary records, 862 iteration records and 1152
  terminal rung records. Updated project state, documentation map and
  active plan. Existing manuscript and unrelated untracked .claude remain
  untouched. No threshold changes, continuation preparation or DMRG.
- Reused the anchor loader with optional t0/campaign arguments; both
  original V=-0.4 default summaries still match their prior report exactly.
  All 18 compact/config/seed hashes, seed readbacks, raw handoffs, channel
  residuals and logs agree. Pointwise four-fingerprint comparisons pass.
  Recomputed terminal configured-window spans, corrected energies,
  physical-spin/Hartree mapping and allocation arithmetic; rehashed every
  source after reading. Initial analysis checks exposed an absent numerical
  HDF5 group and the square kernel's leg-odd minus sign; the analysis now
  uses the hashed config and documented leg swap. The successful local
  rerun took under one minute. All five PNGs were viewed; the pairing log
  range was expanded to retain measured sub-1e-11 values and viewed again.
- Energy differences remain trajectory diagnostics because no endpoint is
  accepted. Reproduce locally with
  `python -B -X utf8 ladder_mps_mft/scripts/analyze_two_basin_grid.py`.

### 2026-09-15 — Prepare cubic two-basin grid and short square V=0 continuation

- User approved the preliminary square diagram and requested the same
  cubic_unfrustrated grid plus a shorter square (1.4,0) continuation.
  Prepared two direct Perlmutter entry points, with no remote access,
  transfer, scheduler action, local reservation or DMRG solve.
- Cubic: all 18 point/family starts, L=64, chi=200, t0=1/1.2/1.4 and
  V=-0.4/-0.2/0, reciprocal 95%/5% correlations from the existing versioned
  square reference bundle. Reconstruct each seed with its target cubic
  kernel and exact E_p. Fresh MPS, max 60 raw evaluations, minimum 40,
  ten stable records. No Anderson, damping or automatic further segments.
  Slurm ceiling 12 h per GPU branch, solver deadline 11.5 h: 54 node-hours.
- Square: continue both original V=0 MPS lineages from jobs 58093802 and
  58093803 (80 and 62 evaluations) for at most 20 additional evaluations
  each. Minimum ten fresh records, ten-record window. Slurm ceiling 8 h
  per GPU branch, solver deadline 7.5 h: 4 node-hours. The pairing lineage's
  late 21.6 minutes/evaluation motivates the eight-hour request. Max totals
  are 100/82; the drift checks may still prevent formal convergence.
- Energy-window tolerance is 1e-7 t/site; inner-DMRG stopping and gate are
  both 1e-7 t total. Density 1e-5 and identity/effective-energy consistency
  1e-8 t/site are unchanged. Relative field tolerance remains 1e-4 and
  spatial channel/slow-mode guards are retained. Square absolute field
  tolerance/noise floor remain 1e-7/5e-7 t. Cubic uses 3e-7/1.5e-6 t,
  scaling with its 6g versus 2g density kernel; this is a provisional
  transfer of physical spin resolution, not measured cubic noise.
- The square preparer pins both compact hashes, original configs, full
  SHA links and E_p registry; checks the source Hamiltonian; and requires
  the full MPS, matching fields and full-file hash on Perlmutter. Full
  hashes are 9ad2d9ea1239727e577be2997a7a9f62e1e6aaeae0653bd8591448f55dc58ea5
  and d4b2ef33f8d969e23b519f08642f50eacfd158948c0b6710756613fd89344237.
  Same-model parent ancestry restores the checkpoint chemical potential
  and permits the existing plotting adapter to stitch the entire history.
  Only compact files are present locally. Preview mode is explicitly
  non-submittable; the production validation function rejects it.
- Config SHA-256: cubic
  7a492c399cff7d9891f3b6430fe4287dadd228e6115a10178d050b6513df222c;
  square 31817ca8b8efce4dcb8b0a43834e7d0e1d8b9ab9dc5e4cad47ec4a6d539c0428.
  Qualification implementation fingerprint:
  a29d3f9f0ed848454f66d02ff787e157127273171191173bb8e33893edcf2550.
  Provenance and the 36-row saved-history receipt are in
  `docs/reports/two_basin_next_campaigns_20260915/`.
- Local preparation passed 182 assertions; existing raw-basin tests passed
  52+23 assertions. All 18 source histories were hashed before and after
  replay. Under square extension controls the paired V=-0.4 histories
  first pass available gates at stripe/pairing iterations 35/27. Both V=0
  histories and both (1.2,-0.4) histories fail at every eligible prefix.
  Applying cubic tolerances to unscaled square fields as a permissive
  stress test also rejects these instabilities; settled V=-0.4 controls
  pass at forty. Missing per-iteration identity errors preclude relabeling
  old results as accepted. New inner-DMRG noise must be assessed from new data.
- Local mock-launcher and production prepared-run-guard tests pass; all
  four shell files pass Bash syntax. The old remainder wrapper regression
  also passes. Checks caught and fixed a model-fingerprint change caused
  by copying the checkpoint mu into model.mu_initial (the solver already
  restores checkpoint mu), and stale source-campaign version/scratch
  metadata inherited through run.env. Shared account and budget settings
  are retained while new campaign metadata is regenerated. Initial test
  harness failures involved Julia macro syntax, Windows path normalization,
  and Git Bash PATH/line endings; corrected focused checks pass.
- Launcher v1.20.0 adds the two preparation modes and retains older-run
  compatibility. Both wrappers reconcile the existing append-only ledger
  and submit only through its budget gates. Combined ceiling is 58
  node-hours. The original anchor run.env remains unchanged. New default
  IDs: 20260915_cubic_unfrustrated_two_basin_95_5_60 and
  20260915_square_t014_v000_two_basin_finish20.
- User handoff from `cd "$CFS/m4863/MPS-MFT/ladder_mps_mft"`: after
  `git pull --ff-only`, run `bash slurm/submit_square_two_basin_finish.sh`
  and `bash slurm/submit_cubic_unfrustrated_two_basin.sh`. Prepared source,
  controls, documentation and preceding full-grid analysis are published
  under the existing push authorization. No job IDs exist for these new
  campaigns in the local evidence at handoff.

### 2026-09-15 — Share the full square energy grid's y-axis

- User requested identical energy scales across all nine full-history
  panels to help interpret the transition. Updated the reusable plotting
  function and regenerated only `energy_grid.png` and its PDF companion
  from the already verified 862-row iteration table. Shared limits are
  [-1.1995785687934484, -0.49175597421212397] t/site.
- Verified all nine axis limits agree exactly, all 18 curves remain inside
  the range, and the source table hash is unchanged. Viewed the PNG:
  labels and curves are visible without clipping. No simulation or other
  figure was rerun; the temporary redraw helper was removed.
- The common absolute scale is dominated by between-Hamiltonian energy
  differences. MF relaxation trajectories alone do not establish the
  order of the equilibrium transition; acceptance and phase assignments
  are unchanged. The late-history grid retains its local energy zooms.

### 2026-09-15 — Restore individual energy-panel scales

- At the user's request, reverted the shared-y-axis change because it hid
  useful relaxation features. Restored the original `energy_grid.png`, PDF,
  plotting code and accompanying mutable documentation byte-for-byte from
  commit d41cdbd. Each panel again uses its individual energy scale.
- Verified restored bytes against Git. No simulation or redraw was needed;
  this ledger retains the earlier experiment and records its reversal.

### 2026-09-15 — Expectations for the stripe–pairing transition

- User asked whether transition order is predictable or worth determining.
  Working interpretation: a direct first-order stripe/paired transition is
  plausible in a conventional competing-order Landau description, but
  not established or required for this model. An intermediate state with
  both orders can instead permit separate continuous boundaries. No Landau
  coefficients have been fitted to the current data.
- Reviewed the general two-order free-energy analysis in
  https://arxiv.org/html/0912.3556v1 (a pnictide application, not this model),
  the Hubbard stripe/SC coexistence result https://arxiv.org/abs/2303.08376
  (Science 2024, different hopping/model geometry), and the closest
  two-channel MPS+MF study https://arxiv.org/abs/2301.08116
  (PRB 2025, attractive chains rather than repulsive ladders). These support
  the alternatives, not a prediction or novelty claim for our parameters.
- The coarse square grid and delayed MF collapse do not determine order.
  Two stable competing minima at the same parameters have not been
  demonstrated in this campaign. A candidate future diagnostic is a
  denser fixed-t0=1.4 cut between V=-0.2 and 0, tracking both continuations,
  order parameters and comparable corrected energies at each coordinate.
  First order is associated with a branch crossing and order-parameter
  discontinuities; energy itself need not jump. Distinguish equilibrium
  crossing from loss of metastability and incomplete MF convergence.
- Establishing direct competition versus coexistence, and its geometry
  dependence, would strengthen the report. Any initial conclusion remains
  within the static interladder MPS+MF approximation at finite chi and L;
  thermodynamic robustness and universality need additional work. No new
  runs, threshold changes or manuscript claims were introduced.

### 2026-09-15 — Prepare finer square cuts and inspect bare-ladder energies

- User requested t0=1.4 at V=-0.05/-0.10/-0.15 and V=-0.4 at
  t0=1.25/1.30/1.35, using the same two-basin approach as the newly submitted
  cubic campaign. The user explicitly requested naive linear E_p
  interpolation between the coarse endpoints, then bare E0 and E_p plots
  along both cuts to inspect the approximation. Cubic submission is
  user-reported; no new job IDs/accounting were supplied. Square V=0
  continuation submission is unreported. No Perlmutter action performed.
- Added explicit V-axis interpolation while preserving exact/default-t0
  behavior and historical model fingerprints. It requires a bracket of
  matching L/U/density/t0, never extrapolates, and rejects sign-crossing
  endpoints. Configs, seeds, manifest, model provenance and checkpoints
  record interpolation mode, endpoints and weight. The new preparation
  pins the registry hash and requested coarse bounds rather than silently
  selecting new intermediate measurements if the registry later changes.
- Prepared all twelve independent field starts from the versioned reference
  correlations: reciprocal 95%/5% mixtures, rebuilt with target couplings,
  fresh MPS, square geometry, L=64, U=8, n=0.9375, tp=0.1 and chi=200.
  Raw updates, no Anderson, max 60/minimum 40, ten stable records. Square
  abs/relative field gates remain 1e-7/1e-4 with a 5e-7 channel floor;
  energy window is 1e-7 t/site and inner DMRG tolerance 1e-7 t total.
  Full-window drift and slow-mode gates remain active. These are basin
  comparisons, not parameter-continuation/hysteresis scans.
- Linear signed E_p values in requested order are -0.16160336393289043,
  -0.17666899694661709, -0.19173462996034374, -0.2508405031610721,
  -0.25043512171026805 and -0.250029740259464. The first three interpolate
  V=-0.2 to 0 at t0=1.4; the latter interpolate t0=1.2 to 1.4 at V=-0.4.
  The MF denominator is |E_p|. No additional bare-ladder jobs are prepared.
- Reused data/E_p_values.csv at bare chi=1000, L=64, n=0.9375 and U=8 to
  generate the requested cut plots. Five plus four measured rows represent
  eight unique points; relevant root-registry values agree. No measurements
  exist at the six new points. Source registry SHA-256 is
  2209bd2ca3c1ad02c0e542d1a9d63ecf90fdfa49120ad9cc3af599a5b4bc1f0e.
  Plots and numeric tables are in
  docs/reports/two_basin_fine_cuts_20260915/, reproduced with
  scripts/plot_bare_ladder_fine_cuts.py. Existing root figures and restored
  per-panel MF energy scales are preserved.
- Total bare energies look smooth at sampled resolution. The binding
  magnitude has a broad maximum around t0=1.2–1.4 at V=-0.4. A degree-three
  interpolant through four neighboring measurements changes the inferred
  coupling by 0.12–0.61% on the V cut and -3.60% to -4.99% on the t0 cut.
  This is an interpolation-shape sensitivity diagnostic, not measured data
  or an uncertainty bound, and it is not used by the campaign. Retain the
  user's straight-line approximation, with its stronger t0 sensitivity
  documented. Neither sparse bare energies nor unaccepted MF competitors
  determine transition order or a precise equilibrium boundary.
- Local unit/preparation checks passed 16 fixed-reference-length, 45
  interpolation/regression and 171 prepared-branch assertions. Raw-basin
  regression checks passed 52+23. Five local mocked launcher/guard tests
  passed, including refusal to use the original source checkout and
  retention of account/shared ledgers. Bash syntax checks passed. Python
  plot estimates match the Julia-prepared receipt, and source bytes are
  unchanged. Viewed the overview and representative separate PNGs; fixed
  a clipped standalone title. No full suite or DMRG simulation was run.
  Local provenance Git calls emitted sandbox ownership warnings; numerical
  and model fingerprints were still verified.
- Config SHA-256:
  92aed5482edf8723bd363a61b3e573d951f0532e24e43315f3269771ea55ca63.
  Numerical fingerprint:
  d928882239844f67c88e7020a4f1f3bf9ae009060c3dc5256406fcf6b18a8e2e.
  Implementation fingerprint:
  c054eb9690ce308e1dfb413bbc82d5e430eeecd3bf7f0b018acce267be9ccc99.
- Launcher v1.21.0 and slurm/submit_square_two_basin_fine_cuts.sh prepare
  and submit twelve branches, default run ID
  20260915_square_two_basin_fine_cuts_95_5_60. Each requests 12 hours at
  one-quarter GPU-node share (11.5-hour solver limit), reserving at most
  36 node-hours and 720 MF evaluations total. Shared append-only accounting
  remains enforced; one segment per branch, no automatic extension.
  The handoff uses a separate worktree after fetching, preserving submitted
  cubic source. The original anchor run.env is unchanged. Publication uses
  the user's existing authorization to push and provide submission commands.

### 2026-09-15 — Positive-V square coexistence test with four seed families

- User proposed square (t0,V)=(1.2,+0.2), retaining the two established
  stripe/pairing mixtures and adding two legacy-inspired intertwined-order
  starts at physically motivated wavelengths. Their working picture is
  predominantly stripe/pairing competition at V<0 and possible coexistence
  at V>0; the former is not taken as an exclusion of all coexistence and
  the latter remains to be tested. User identified legacy convergence and
  positional pinning concerns. No broad scan or new bare jobs were added.
- Read the local legacy V=+0.2 profiles at t0=1.0/1.2/1.4; used the matching
  t0=1.2 nodamping file to extract the new shapes. SHA-256:
  8a8f5b917d11259d34ce773cb2860fb20d809f5dccfd252342f69a4596b9fec6.
  It is incomplete after 60 records and has mean density 0.9388028989,
  missing the 0.9375 target. Its geometry label is absent; Hartree-kernel
  residuals favor cubic_frustrated (max 1.10e-6) over square (0.0837) and
  cubic_unfrustrated (0.1642). This is recorded as inferred geometry.
- The final legacy profile has hole peaks at rungs 8,19,30,42,55. In bulk
  rungs 9–56, |rung pairing| correlates +0.9923 with holes and -0.9657 with
  |staggered leg-odd Sz|. Rung and leg singlet amplitudes have opposite
  signs, with no rung-pair sign changes between peaks. No accepted energy
  or optimal wavelength is inferred from this unfinished trajectory.
- Added scripts/inspect_positive_v_legacy.py, producing source-hashed
  evidence and data/positive_v_intertwined_recipe.toml. The recipe uses
  bulk 5th/95th percentiles for charge/pair amplitudes and 95th-percentile
  |Sz| for spin, plus translation-averaged relative-bond pairing and normal
  correlation coefficients at offsets 0–4. Source bytes are unchanged.
  Recipe SHA: 786fa2e8846820f42aabbabb625c3a645558d25e5388afa4058abf63cf834d35.
- Added a four-branch preparer and square chi=200 base config. The first
  two starts reproduce the reciprocal 95%/5% reference mixtures. The new
  starts use charge/pairing periods 8 and 16 rungs and spin envelopes 16
  and 32, with common pairing sign between maxima and leg-odd AF spin.
  At n=15/16, these cells contain one and two holes, motivating half-filled
  and filled stripe-counting alternatives. The legacy's irregular 11–13
  rung spacings are discarded. Both new seeds have exact target density,
  zero total Sz and pairing maxima at hole-rich spin antiphase walls.
- New seed amplitudes: charge 0.0520678392, Sz envelope 0.3061558088,
  rung-pair mean 0.0260201643 and modulation 0.0167601652. The nearest-leg
  pair coefficient is -0.604435 of the rung envelope. These are synthetic
  initial-correlation templates, not an asserted physical MPS. The square
  kernel rebuilds all fields, including its zero cross-leg alpha/beta;
  no legacy MPS, irregular peak positions or persistent external pins enter.
  All starts use fresh MPS initialization with random_seed=1404.
- Controls match the recent square finer cuts: max 60/minimum 40 raw
  evaluations, ten stable records, no Anderson; field abs/rel 1e-7/1e-4,
  channel floor 5e-7, energy window 1e-7 t/site and inner-DMRG 1e-7 t total.
  Full drift, slow-mode and identity gates remain unchanged. Exact bare
  chi=1000 E_p=-0.15307266912955697 is used. No src/ solver files changed.
- Local focused tests passed 33 shape/kernel and 49 prepared-run assertions
  (82 total), including density/spin sums, pair/charge locking, antiphase
  wavelengths, square field reconstruction, reference mixtures, seed
  hashes and matching fingerprints. The preparation test was made repeatable
  using a temporary directory and rerun successfully. The durable preview
  remains in output/seed_previews/20260915_square_positive_v/control.
  Six local Bash syntax/mock/guard tests passed, including existing
  wrappers and the new wrapper's source isolation and shared accounting.
  Both figures were visually checked. No DMRG or full suite was run.
- Config SHA: f5f859581210324dc388e41250fb28961f5b0f12fa811365cbaf88c4142b0e5a.
  All four model fingerprints:
  a994c1ee7bf7b9cbe952f448e20f451520143edecb2432f26c5bcc07e744d50c.
  Numerical fingerprint:
  d928882239844f67c88e7020a4f1f3bf9ae009060c3dc5256406fcf6b18a8e2e.
  Unchanged implementation fingerprint:
  c054eb9690ce308e1dfb413bbc82d5e430eeecd3bf7f0b018acce267be9ccc99.
- Launcher v1.22.0 adds prepare-square-positive-v and
  slurm/submit_square_positive_v.sh. Default ID:
  20260915_square_t012_vp02_four_seeds_60. Four 12-hour one-GPU jobs
  reserve at most 12 node-hours (11.5-hour solver limit) and 240 evaluations.
  Shared append-only accounting remains enforced, with one segment per
  branch and no automatic extension. The user handoff uses a new worktree
  to preserve submitted sources. No Perlmutter connection, transfer or
  scheduler action was performed; publication uses standing push authority.
- Full source/seed figures, numeric receipts and interpretation boundaries
  are in docs/reports/square_positive_v_seeds_20260915/README.md. Stable
  coexistence requires surviving stationary bulk orders and the usual gates.
  Residual pairing during collapse is not coexistence. Distinct converged
  wavelength states may be metastable; unresolved evolution remains flagged.
  Open-boundary and algorithmic pinning are not ruled out by these starts;
  translated seeds can be considered later if needed. Existing campaign
  source artifacts, acceptance flags and energy rankings are unchanged.

### 2026-09-15 — Physical interpretation of the possible V-sign contrast

- User asked whether negative-V stripe/pairing competition versus positive-V
  intertwining has a simple physical interpretation, conditional on transfer
  of the legacy pattern to square geometry. Working explanation, not a
  established mechanism: local magnetic/pairing competition can remain
  positive on both sides while spatial organization permits coexistence.
  A phenomenological positive gamma |Delta(x)|^2 M(x)^2 term favors pairing
  at weak-spin domain walls and magnetism between them. Observed spatial
  anticorrelation does not require the local coupling to become cooperative.
- In the implemented Hamiltonian, V multiplies nearest-neighbor n_i n_j
  on both ladder legs and rungs. Writing n_i=1-h_i leaves a V h_i h_j
  interaction plus constant and one-body terms (the latter are boundary
  dependent on the open ladder). Attraction favors neighboring-hole
  association; repulsion penalizes it. A possible mechanism is a change
  in stripe localization/filling and carrier mobility that allows pairing
  on the walls. This does not imply that repulsion generically creates
  long-wavelength stripes, or that attraction precludes intertwining.
- Primary-source checks: https://arxiv.org/abs/2303.14723 reports attraction
  enhancing SC and eventually causing phase separation at stronger coupling
  in a different extended two-leg model; https://arxiv.org/abs/2206.03486
  reports enhanced SC and suppressed CDW on four-leg cylinders;
  https://arxiv.org/abs/2109.00213 finds attraction-enhanced short-distance
  d-wave correlations in striped states. These illustrate competing
  tendencies and rule out treating the proposed sign contrast as universal.
- Rechecked the repository E_p registry at t0=1.2: signed E_p is
  -0.25124588461187614 at V=-0.4 and -0.15307266912955697 at V=+0.2.
  Thus g=tp^2/|E_p| rises from 0.03980164696 to 0.06532844862, about 64%.
  In this effective model V changes both the bare ladder and interladder
  coupling; g rescales spin/charge/exchange as well as pairing. An eventual
  fixed-g comparison could separate those effects, but no new runs or
  implementation changes are authorized or prepared by this discussion.
- Proposed interpretation remains conditional on stationary square results.
  Useful later checks are stripe width/filling, pair/spin spatial overlap,
  comparable accepted energies and response with the interladder scale held
  fixed. No phase classification, acceptance flag or campaign status changed.

### 2026-09-15 — Verify the Zhou2023 literature-review entry

- User asked whether the previously linked two-leg ladder study is included.
  Confirmed existing key Zhou2023 in literature/literature_review.tex:207,
  literature/references.bib:420 and literature/SOURCE_NOTES.md:82, with
  selected-full-text reading depth S. The manuscript already cites it at
  introduction_and_results.tex:158. No duplicate entry or rebuild is needed.
- Rechecked the published paper, https://doi.org/10.1103/PhysRevB.108.195136,
  for the requested chat summary. Its repulsion results support persistence
  of pairing alongside stripe correlations, but do not substantiate the
  previously proposed stripe-delocalization mechanism. Retain that mechanism
  as untested. Distinguish isolated-ladder correlations from array mean-field
  order, and match density and diagonal hopping before comparing boundaries.

### 2026-09-15 - Clarify stripe orientation in the manuscript

- At the user's request, added the agreed statement to manuscript section
  2.4: equivalent ladders are the prescribed microscopic geometry, while the
  observed stripe texture is an additional longitudinal charge/spin modulation
  whose domain walls extend transverse to the ladder axes in the repeated-
  ladder embedding. Explicitly qualified transverse alignment as an embedding
  restriction, not an independently optimized arrangement of distinct ladders.
- Refined the adjacent Arrigoni2004 comparison: alternating dopings characterize
  its period-four construction; the paper also studies equivalent ladders in
  its period-two limit. Reused the existing citation, with no bibliography change.
- Rebuilt the manuscript PDF with cached Tectonic and refreshed the tracked
  Overleaf ZIP. The PDF remains 16 pages with 34 references, no undefined
  references or overfull boxes, and no out-of-margin text. All sixteen rendered
  pages were inspected, and the package helper verified every archive member.
  No numerical calculation or change to scientific acceptance was involved.

### 2026-09-16 — Analyze synced square continuations and partial cubic grid

- User reports Perlmutter monthly maintenance and a fresh local output sync.
  Analyzed ten terminal states: both square finish20 jobs 58383124/58383125
  and cubic jobs 58383141/58383142, 58383146/58383148,
  58383153/58383156 and 58383157/58383158. These cover square (1.4,0)
  and cubic (1.0,-0.4/-0.2/0), (1.2,-0.4). All reached their configured
  iteration limits: 20 additional per square lineage, 60 per cubic start;
  520 new MF evaluations. All remain unaccepted, period 0. Logs confirm
  orderly solver completion rather than a maintenance interruption.
- Added scripts/analyze_two_basin_progress.py, reusing existing HDF5 axis,
  field-channel and profile helpers. Report, JSON evidence, 520-record CSV,
  and three PNG/PDF figure pairs are under
  docs/reports/two_basin_progress_20260916/. The square figure stitches full
  100/82-evaluation lineages after exact parent restart-field verification.
  Cube plots show all available full histories, without filling missing cells.
- Physical observables are reconstructed using the actual square/cubic
  kernels and checked against endpoint correlations. Verified compact/full
  manifest relationships, seed/config hashes, stored model/numerical/source
  fingerprints, raw-update adjacency, energy density correction, channel
  window spans and log iteration counts. Full hashes and job-by-job metrics
  are recorded in analysis.json. Parents retain full SHA-256
  9ad2d9ea1239727e577be2997a7a9f62e1e6aaeae0653bd8591448f55dc58ea5
  (stripe) and d4b2ef33f8d969e23b519f08642f50eacfd158948c0b6710756613fd89344237
  (pairing). Campaign results retain implementation SHA-256
  d37e6563416286a9257133490f6e9e345960b950c08b157bc66c964eb369b624.
- Square pairing RMS drops from 0.003307 at cumulative iteration 63 to
  4.42e-8 at 82; the stripe endpoint has 1.04e-9. The former paired lineage
  has reached the stripe basin observationally. Stripe-node motion remains
  resolved (up to 0.043/0.223 rung over the final ten records). The stripe
  energy range passes 1e-7 but its field/slow/channel gates fail; the pairing
  lineage still fails the energy range (1.83e-6) as well. Absolute endpoint
  energy separation is 1.93e-6 t/site; no accepted-energy ranking is made.
- Cubic starts all reach essentially unpaired stripes: physical spin RMS
  0.307–0.356 and leg-pair RMS below 9e-12. Pairing-seeded histories stay
  below 1e-4 pairing by iterations 5–8, much earlier than the corresponding
  square transient at (1.2,-0.4). Dominant charge/spin modes are 4/30.
  Physical stripe amplitudes exceed square values at matching coordinates;
  this is not just the threefold MF-field normalization. The cubic boundary
  remains unknown because no t0=1.4 results are yet locally available.
- Cubic energy windows and inner/density/identity checks pass. Closest case,
  stripe (1.0,-0.2), fails only charge span: 1.01306e-4 vs 1e-4. Others
  retain channel/slow-mode failures, including near-unit-contraction
  extrapolation sensitivity and more resolved drift at (1.2,-0.4).
  Preserve all acceptance flags; no tolerance or solver changes are made.
- Synced sacct reconciliation records give square elapsed 3930/4025 s,
  quarter-node charges 0.272916667/0.279513889, total 0.552430556 node-hours.
  No cubic reconciliation is synced: its saved MF times give 6.160414401
  solver node-hours excluding overhead, not an exact allocation total.
  Accounting ledgers were read only. Synced submission records also identify
  twelve fine-cut jobs 58387972–58387983 and four positive-V jobs
  58394103–58394106. Those and ten remaining cubic starts have no local
  state/stdout; no live scheduler state is inferred or queried.
- Local command: C:/Python313/python.exe -B -X utf8
  ladder_mps_mft/scripts/analyze_two_basin_progress.py. Runtime about 7 s,
  with all embedded evidence checks passing and all three PNGs visually
  inspected. Updated PROJECT_STATE, ACTIVE plan and documentation map.
  No DMRG, transfer, Perlmutter access, new submission, continuation or
  acceptance mutation. Next scientific information should come from the
  already-submitted remaining cubic points, fine cuts and positive-V starts.

### 2026-09-16 — Incorporate continuations into the full square-grid report

- User requested an in-place update of docs/reports/two_basin_grid_20260915/.
  Updated the existing grid analyzer and shared anchor loader to read the two
  finish20 parents/children with compact/full hashes, model/job/count checks
  and exact restart-field handoffs. Continuation jobs 58383124/58383125
  extend the original 58093802/58093803 histories to 100/82 cumulative
  evaluations. Acceptance gates use only each latest source segment and
  its archived controls; no thresholds or acceptance flags change.
- Regenerated all five PNG/PDF pairs, endpoint/profile tables, source inventory,
  iteration CSV and analysis JSON. Full histories keep original records and
  mark continuation starts; late-energy panels use each lineage's final 15
  records with cumulative indices. Individual energy y scales are preserved.
  The (1.4,0) label changes from S* to S because both lineages now have tiny
  pairing; seven stripe/two paired assignments remain preliminary.
- The report now has 18 independent lineages, 20 source artifacts/jobs,
  902 unique evaluation records, zero accepted endpoints, 27.350763889
  actual allocation node-hours and 26.983410572 solver-only node-hours.
  JSON source_runs retains original endpoint diagnostics and parent controls;
  CSV source job/iteration columns distinguish source-local and cumulative
  numbering. Both continuation costs and parent costs are counted once.
- Ran the existing Python grid analyzer locally with its embedded source,
  channel, history and accounting checks; visually inspected all five PNGs.
  A narrow metadata regeneration retained those inspected plots. Compared
  the 16 unaffected endpoint rows, 720 unaffected history rows and 1024
  unaffected profile rows against pre-update digests: all unchanged.
  Checked unique job/iteration pairs, 100/82 continuity, 20-job accounting,
  and equality of new energies/pair amplitudes/source hashes with the separate
  September 16 continuation report. git diff --check passes. No DMRG or
  Perlmutter operations. Updated the report narrative, documentation map,
  PROJECT_STATE and active plan; the earlier progress report remains intact.

### 2026-09-16 — Prepare four trellis runs with two spatial implementations

- User requested (U,t0,tau0,tau1,V)=(8,1,.1,.1,0) with the established
  stripe/pairing 95%/5% correlation seeds in both a fixed reciprocal
  one-ladder map and an explicit two-ladder cell. The user reports the other
  runs are still in progress. No scheduler state was queried or inferred;
  no existing source checkout, result artifact, acceptance flag or ledger
  was changed on Perlmutter.
- Added src/Trellis.jl and dispatch from run_scf. One-ladder leg maps use
  A and A^T, A=tau0 I+tau1 S with OBC. The rectangular A/B cell uses
  forward paths on A and reverse paths on B for equal hoppings. A/B are
  independent spatial MPS states, solved against frozen incoming fields
  before a simultaneous raw update. The convention never alternates by
  iteration parity. Each ladder retains the existing fixed mean density.
- The same r_range projection acts on input and output, preserving
  reciprocity even where a shift crosses the cutoff. Retained zigzag cross
  terms include bond contributions to mu_cdw and the off-diagonal
  -T T^T/Delta normal term. The existing centered variational functional
  is reused with current simultaneous cell correlations for the interaction
  energy and the actually applied fields for the Hamiltonian identity.
  Tests verify its derivative for both cells, unequal hoppings and cutoffs.
- Complete-cell HDF5 stores per-ladder MPS, fields, correlations, raw
  histories and convergence evidence, plus cell totals and per-site energy.
  Spatial A/B order is period one of the cell update. Both members must
  pass all stationary gates in one sweep; temporal cycles remain unaccepted.
  Compact copying removes each nested MPS; complete-cell resume requires a
  pinned full checkpoint. Different mean A/B fillings are outside this run.
- Added the trellis base config, preparation script, v1.23 GPU-launcher
  campaign kind and separate-source submission wrapper. L64 per ladder,
  n=.9375, chi200, r_range4, max60/min40 cell sweeps, ten stable records,
  field tolerances 1e-7/1e-4, channel floor5e-7, inner-DMRG1e-7 total and
  energy-window1e-7 per site. Four 12-hour shared-GPU jobs, one segment each,
  cap reservation at12 node-hours and at360 density-targeted ladder solves.
  No new E_p jobs: exact highest-chi E_p=-.13251724, bare chi1000.
- Preview at output/seed_previews/20260916_trellis_comparison/ has exactly
  four configs and hashed seeds. Reference SHA-256 is
  e01a1ea7d6be813110870d26377db0df529d1584816c946af78584e1e1fbddc1.
  Local GPU-manifest implementation SHA-256 is
  dc3f49684b789a7645358f33c1bd9168b3ff491fe3ca9b55afa1f6a9492832ef.
  Full seed/model/numerical/registry hashes are in the manifest and tracked
  docs/reports/trellis_comparison_20260916/prepared_branches.csv. Host paths
  and source fingerprints are regenerated during Perlmutter preparation.
- Focused validation passed150 algebra, projection, preparation and spatial
  stationarity assertions; tiny CPU L2/chi16 tests passed51 driver, identity,
  history, complete-cell loading and compact-storage assertions. The initial
  smoke exposed HDF5's unsupported BitVector input in a new Boolean history;
  changed it to a dense vector and reran the complete smoke successfully.
- One repository-wide test attempt stopped at two pre-existing unqualified
  numerical_fingerprint calls in tests. Qualified them and ran only the
  failed/remaining testsets, deleting the temporary continuation driver
  afterwards. Combined coverage:784 passing assertions and two existing
  Windows-specific shell skips, including150 trellis unit assertions.
  Six Python launcher tests passed with actual local Git Bash syntax and
  fake-launcher isolation/accounting checks; no Slurm commands ran.
  Full logs are output/trellis_validation/runtests.log and remaining_tests.log.
- Local Julia's app alias and writable compile cache were inconsistent under
  the sandbox. Validation completed with the direct1.12.7 executable and
  --startup-file=no --compiled-modules=existing, reusing installed packages
  without changing user caches. A private-depot attempt failed in dependency
  precompilation before tests; no NERSC authentication or connection occurred.
  Git ownership was handled with per-process safe.directory only.
- Extended the existing source bundler with --trellis, including the actual
  versioned reference HDF5 and the new method/campaign docs. The handoff uses
  output/source_bundles/trellis_comparison_20260916.zip with verified member
  hashes, CRCs and SHA-256 sidecars, extracted to a new Perlmutter source
  directory. The original accounting environment and budget gates are reused
  by the user-run wrapper. No commit, push, transfer or submission performed.
- Updated PROJECT_STATE, ACTIVE, architecture, variational-method pointer
  and documentation map. The method report distinguishes skew repetition
  from rectangular transverse order and their OBC end cuts. This is local
  implementation validation, not GPU performance or scientific convergence.

### 2026-09-17 — Repair CRLF checksum sidecar in the trellis handoff

- User-provided Perlmutter output shows sha256sum trying to open a filename
  ending in a carriage return. The set -e handoff stopped at verification,
  before extraction or submission; no new job IDs were supplied.
- Confirmed the local checksum sidecar contains CRLF. The Python bundle
  writer used platform-default text newlines. Changed both checksum and
  manifest sidecar writes to explicit LF, and normalized only the existing
  local checksum file. Preserved the dated ZIP byte-for-byte, SHA-256
  b27d1825b66f5dcec3d4e002fb29ba700123d442b7895f7368fde6538cad7629.
- Updated the handoff to pipe the checksum through tr -d '\r' before
  sha256sum -c -. Verified that exact pipeline against the original CRLF
  sidecar using local Git Bash, then verified ordinary sha256sum -c against
  the repaired LF sidecar. Both report the ZIP OK. The first local Bash
  attempt lacked /usr/bin in PATH; adding that local tool path resolved it.
- Updated PROJECT_STATE. No ZIP regeneration, solver/config changes,
  DMRG tests, Git pull, transfer or Perlmutter operation. The user can use
  the existing transferred ZIP with the corrected shell block; ordinary
  underscores must be copied without Markdown backslashes.

### 2026-09-18 — Complete cubic/fine-cut review, positive-V results and first trellis endpoint

- User requested the same analysis as the completed square-grid report for
  cubic unfrustrated, the finer square transition cuts and (1.2,+0.2), then
  added the first completed trellis run and a single readable report plus
  updated project notes. Work used only user-synchronized local output.
  No Perlmutter authentication, transfer, scheduler query or submission.
- Added `docs/reports/campaign_review_20260918/README.md` as the consolidated
  scientific readout, with four detailed report directories:
  `cubic_two_basin_grid_20260918`, `square_fine_cuts_20260918`,
  `square_positive_v_20260918`, and `trellis_progress_20260918`.
  They retain complete/late energy plots on individual y scales, physical
  order histories/profiles, failed gates, source hashes and CSV/JSON data.
- Cubic campaign `20260915_cubic_unfrustrated_two_basin_95_5_60` has all
  18 terminal states, 60 evaluations each, 1080 total. Both seeds reach
  stripes at all nine points, including square's paired (1.4,-0.4/-0.2)
  coordinates. Physical spin RMS 0.280–0.356; maximum leg-pair RMS 2.214e-11;
  dominant charge/spin modes4/30. All final energy windows, density,
  inner-DMRG and identity/effective gates pass; field/profile/slow-mode
  failures remain. Closest is stripe (1,-0.2), charge span 1.01306e-4 versus
  1e-4. No acceptance flag or threshold changed. All job IDs are retained
  in that report's sources.csv and run_summary.csv.
- Finer square campaign `20260915_square_two_basin_fine_cuts_95_5_60` has
  all 12 terminal states (jobs 58387972–58387983), 720 evaluations. Both starts
  are paired at (1.4,-0.15/-0.10) and(1.30/1.35,-0.4). Distinct stripe/paired
  trajectories remain at (1.25,-0.4) and(1.4,-0.05). Their signed endpoint
  E(pair seed)-E(stripe seed) diagnostics are -3.18913e-5/-5.95196e-5 t/site;
  summed final-ten ranges 1.57449e-7/7.46127e-8. These do not constitute
  accepted ranking, hysteresis or proof of transition order. All six E_p
  estimates match manifest-recorded linear interpolation; no new bare jobs.
- At paired (1.4,-0.05), the usual spin RMS grows 1.50% over evaluations 51–60,
  but96.7% of the final full-chain spin-squared weight is in the outer 14
  rungs at each end. Central 32-rung spin RMS 0.001703 still falls 0.385%.
  Added a dedicated boundary-spin figure; no claim of bulk coexistence or
  resolved bulk stripe growth. Paired t0=1.25 spin falls 52% over the same
  window; both t0=1.4,V=-0.10 remnants also decay.
- Positive-V jobs 58394103/58394104/58394106 finish60 each: stripe, pairing
  and intertwined period16 all lose pairing (leg RMS<=8.29e-10), spin RMS
  about 0.233, dominant modes4/30. One global spin flip is aligned only in
  figures; raw exported signs are retained. Job58394105, period8, has 46
  complete MF stdout rows and part of the next DMRG solve, no synced state
  or checkpoint. Latest logged residual 1.046e-3 and corrected energy
  -0.332811522836 t/site remain scalar transient evidence, not a phase label.
- Trellis campaign `20260916_trellis_two_basin_comparison_60` has one complete
  state, one-ladder stripe job 58468871,60 sweeps. It develops pairing:
  spin RMS 0.062715 to 1.21484e-5; leg-pair RMS 0.0096514 to 0.0179689; leg/rung
  mean signs +0.017956/-0.032923. Corrected endpoint energy -0.518820336467
  t/site; global field/slow/density/identity gates pass late, but energy
  span 2.84372e-7, inner-DMRG and channel/profile gates fail. The final
  DMRG sweep gap is 3.3598e-7 total and discarded weight 6.2954e-5.
  Physical profiles come from stored correlation histories: trellis fields
  mix density with normal bonds and cannot use square/cubic inversion.
  Compact SHA256 b3d204a10eeeced5446f22a7b65ffef8314cf5243e49d8bceae11046a1db7b00;
  recorded full SHA256 b6eba891c3da1df315840beed9e2048267ef639ae200ec39dabb098f269fb805.
  The remote full artifact was not accessed or certified locally.
- Other trellis stdout: one-ladder pairing58468873 has 21 complete sweeps;
  two-ladder stripe58468875 has 18; two-ladder pairing58468876 has 1. No synced
  spatial artifacts. One-ladder pairing energy nearly matches the completed
  trajectory, but seed merging and rectangular A/B outcomes remain unproven.
  No ranking of unaccepted cells or different geometries was performed.
- Total reviewed completed subset:34 histories,2040 MF evaluations, zero
  accepted. Saved solver-only fractional node-hours: cubic 13.88699481,
  cuts 13.03048545, positive-V 3.02706540, trellis 1.98680398; total 31.93134965.
  Actual synced cubic reconciliations cover12/18 jobs and 9.44159722 node-hours;
  no actual fine-cut/positive-V/trellis reconciliation. Partial jobs and
  allocation overhead are excluded from solver totals. Prior coarse square
  remains 902 evaluations/27.350764 actual node-hours and is not counted again.
  Both append-only budget ledgers were left unchanged.
- Added read-only scripts `analyze_two_basin_campaigns_20260918.py` and
  `analyze_trellis_progress_20260918.py`. Local commands use
  `C:/Python313/python.exe -B -X utf8 ladder_mps_mft/scripts/<script>.py`.
  Audits verify compact/config/seed hashes, model/numerical/implementation/
  registry fingerprints, raw applied-to -measured adjacency, stdout counts,
  physical correlation consistency, target-density canonical correction and
  channel spans. Trellis additionally verifies cell energy normalization and
  direct canonical reconstruction. Focused plotting/data validation only;
  no solver source change, DMRG run or unrelated regression suite.
- Updated PROJECT_STATE, ACTIVE, documentation/architecture maps, campaign
  result pointers, and manuscript/literature evidence notes. The LaTeX/PDF
  manuscript retains its September 13 numerical cutoff. The September 16
  partial snapshot is preserved as history; coarse square data are unchanged.
  User-requested higher chi/length work remains deferred. No new compute,
  continuation, threshold change, commit, push or remote action in this review.
- Final focused validation: 34 unique completed job IDs, 2040 CSV history
  rows, endpoint/JSON agreement, summed solver cost and 161 local document
  links checked. Both analysis scripts parse; every figure has PNG/PDF
  counterparts. Figures were visually inspected for axes, units, labels,
  clipping and full/late history coverage. Git diff whitespace check passes.
- A redundant check with core.autocrlf=false incorrectly treated Windows
  CRLF endings as trailing whitespace. Repeating with the repository's
  configured normalization passed; no wholesale line-ending rewrite was
  performed. Normalized Git diff confirms the run log has additions only.

### 2026-09-18 — Fix trellis full-history plotting adapter

- User's direct `plot_phase1_mf_profiles_and_middle_histories(state)` call
  on completed trellis job 58468871 failed with a missing recorded seed ''.
  The adapter only looked for root `history/fields` and `fields/initial`.
  Trellis has both under `ladders/A`, so the reader incorrectly entered the
  legacy sparse-snapshot/external-seed fallback. The embedded seed is present.
- Added a common spatial-ladder group selector to seed, complete-history,
  saved-snapshot and profile readers. Trellis defaults to A; `ladder=:B`
  selects B, with explicit errors for unavailable/invalid selections. Titles
  identify the cell and ladder. Parent-history stitching propagates the same
  selection and retains the exact field-handoff check; A/B are never joined
  as temporal records. Root-layout square/cubic behavior is preserved.
- Added standalone `test/test_phase1_plotting.jl` (HDF5/PyPlot environment,
  Agg backend, no solver imports). Eighty-three fixture checks passed for
  root/one-/two-ladder files, embedded seeds with missing legacy provenance,
  applied/measured histories, snapshots, invalid selections, actual slider
  updates and separate A/B continuation stitching. Five further checks
  passed against the user's exact locally synced state, comparing the read
  arrays with HDF5 and rendering the full history with its embedded seed.
- Local command: direct Julia 1.12.7 executable with `--startup-file=no
  --compiled-modules=existing test/test_phase1_plotting.jl <state.h5>`.
  Test execution reports 27.2 seconds for fixtures and 2.5 seconds for the
  real-file check, excluding Julia/package startup. Inspected the resulting
  `output/plot_validation/trellis_mf_and_middle_histories.png`: all five
  profile/history rows and 61 slider positions (seed plus 60 updates) render.
- Updated trellis method instructions and current project notes. These plots
  remain MF fields; trellis fields cannot be read directly as physical
  density/spin/pair correlations. No legacy plotting file, solver controls,
  simulation data, acceptance flag or accounting ledger changed. No DMRG or
  unrelated regression suite was run. Git baseline at task start: fc3db31.

### 2026-09-18 — Full transition-cut energies and actual LaTeX notes

- User clarified that the finer-cut analysis should show the full variational
  energy versus Hamiltonian parameter, not just seed gaps or MF histories,
  and explicitly requested updating the actual LaTeX notes with the new data.
  Baseline: bb76a71; only unrelated `.claude/` was untracked at task start.
- Added a narrow `--energy-cuts-only` path to
  `scripts/analyze_two_basin_campaigns_20260918.py`. It reads twelve fine
  endpoints (jobs 58387972--58387983) and eight coarse endpoints, including
  the latest square V=0 continuations (58383124/58383125), to cover both
  five-coordinate cuts from both seeds. All 20 compact hashes match the
  prior audited reports. Endpoint source/config hashes, job IDs, seed family,
  iteration counts, E_p, canonical and corrected energies are retained in
  `docs/reports/square_fine_cuts_20260918/variational_energy_cuts.csv` and
  `variational_energy_analysis.json`. No simulation artifacts were edited.
- Plotted E_var,target/N versus V at t0=1.4 and versus t0 at V=-0.4, with
  N=128 physical sites. Verified the stored full canonical functional and
  correction e_target=e_var+mu*(n_target-n), matching the existing summaries
  to 2e-14 t/site. No extra field-independent offset or effective-H eigenvalue
  substitution. CSV includes both total and per-site canonical/corrected
  energies. Seed curves connect independent starts, not continued phase
  branches; coarse/fine controls remain distinguishable.
- Added a separate shape figure with one common endpoint chord per cut,
  preserving seed gaps and changes in slope, plus 16 adjacent secants.
  The pairing-seed V slopes are 1.266231,1.266144,1.266212,1.261367:
  a small final-interval downturn of 0.38%. The t0 slopes are
  -0.340515,-0.368857,-0.393241,-0.417749, with nearly regular curvature.
  A weak first-order crossing is compatible with the V bend, but the current
  five-point curves do not resolve a derivative discontinuity. Only one
  point per cut retains distinct textures; no accepted crossing or hysteresis
  is inferred. The interpolated E_p changes the transverse coupling along
  the cuts. Last-ten drift bars are explicitly not convergence error bounds;
  the coarse t0=1.2 pairing endpoint has a 1.56e-4 t/site recent range.
- Updated the fine-cut and combined reports, PROJECT_STATE, ACTIVE and
  documentation map. Updated actual `manuscript/introduction_and_results.tex`
  to include completed square/cubic grids, continuations, full cut energies,
  positive-V and trellis observations; advanced the cutoff explicitly to
  September 18. Added method/interpretation notes to `METHODS_NOTES.tex` and
  a local-project evidence section to `literature/literature_review.tex`.
  Existing bibliography and search cutoff are unchanged. Source notes now
  describe completed incorporation rather than deferring to a later draft.
- Local figure command: `C:/Python313/python.exe -B -X utf8
  ladder_mps_mft/scripts/analyze_two_basin_campaigns_20260918.py
  --energy-cuts-only` (about 3 seconds). No campaign-wide rerun or DMRG test.
  Verified 20 endpoint rows, 16 secants, normalization and Python syntax.
  Inspected both new PNGs and their rendered LaTeX placement.
- Rebuilt manuscript (21 pages/34 references), literature review
  (29 pages/49 references), and methods PDF (21 pages). Tectonic initially
  could not use its default external cache; setting TECTONIC_CACHE_DIR to
  the existing repository cache resolved that. Missing font/package files
  required an approved compiler download after restricted network attempts
  failed; compilation then completed. The bundled Python PDF dependencies
  were used for validation after system Python lacked pypdf.
- All three final builds have no unresolved citations/references, missing
  characters or overfull boxes. Methods retains four harmless underfull
  spacing warnings in its existing code map/bibliography. Verified 25 local
  manuscript evidence links and four figure files; inspected new results
  pages and figure captions. The 10-file Overleaf ZIP includes updated sources,
  methods notes and all required figure PDFs. A cached root-layout build
  reproduces the delivered manuscript's text on all 21 pages. Build logs,
  validation JSON and rendered checks live under ignored
  `output/notes_update_20260918/`. No remote scheduler, transfer, simulation,
  acceptance change, threshold change or allocation-ledger mutation occurred.
- Final checks: 96 local Markdown links resolve, the run-log update is strictly
  append-only, and the configured Git whitespace check passes.

### 2026-09-18 — Side-by-side square/cubic figures in report and manuscript

- User requested matching square/cubic phase diagrams, full energy grids
  (not late-only energies), spin/pairing RMS grids, and figure references
  in the text. Baseline: 72a4359; unrelated `.claude/` left outside scope.
- Added `--geometry-comparison-only` to the existing campaign analysis script.
  Four paired figures keep square left and cubic right with the same point
  order. Every saved evaluation is retained: 902 square and 1080 cubic,
  across 36 independent seed lineages. Square continuation joins retain
  dotted markers after evaluations 80/62 at (1.4,0). Matching coordinates
  share iteration limits; each series stops at its own saved endpoint.
  Energy panels preserve individual y scales as previously requested.
- For the RMS comparisons, both geometries now use physical leg-odd spin
  on rungs 6–59 and physical leg-pair correlations on bonds 6–58, with common
  logarithmic y scales. The older square CSV instead records MF fields and
  averages leg bonds onto rungs before its pairing RMS. Reused the existing
  `read_arrays` correlation reader for all 20 square source artifacts,
  checking their compact hashes, raw-map adjacency, canonical corrections
  and endpoint physical RMS against the existing analysis. Cubic physical
  histories come from the existing audited CSV. No source data were edited.
- Saved four PNG/PDF pairs, the 1982-row `square_cubic_histories.csv`, and
  `square_cubic_comparison_sources.json` under the combined-report folder.
  The latter records source-table and square state hashes plus definitions.
  Initial CSV loading used a TSV helper and failed before plotting; fixed
  it to use csv.DictReader. The successful narrow run took about 8 seconds.
- Updated the combined Markdown report with numbered Figures 1–9 and prose
  references. In the LaTeX manuscript, Figure 2 now compares both phase
  diagrams and Figures 3–5 compare full energy, spin and pairing histories.
  All four are referenced in the scientific discussion. History figures
  occupy landscape pages; remaining results and acceptance statements are
  unchanged. Updated manuscript README/source notes, PROJECT_STATE and ACTIVE.
- Rebuilt the 24-page manuscript with seven figures and 34 references.
  The cached compiler lacked pdflscape/lscape; an approved TeX-resource
  download supplied them, after which compilation completed. No undefined
  references, missing characters, overfull or underfull text boxes remain.
  Checked all 27 manuscript evidence links and seven figures, and visually
  inspected each added comparison and its placement on PDF pages 12–15.
  Updated the Overleaf packaging list to include all comparison PDFs.
- Validation is local plotting/document validation only; no DMRG, numerical
  control changes, new acceptance decisions, Perlmutter access, transfers,
  scheduler actions or allocation-ledger updates were performed.
- Final checks confirm 36 complete plotted histories / 1982 evaluations,
  four prose figure references, 48 local Markdown links, an append-only ledger
  and a clean whitespace diff. The 13-file Overleaf bundle compiles from its
  root with all 24 pages matching the delivered manuscript PDF text.

### 2026-09-18 — Ladder-material discussion and trellis motivation in manuscript

- User requested the missing material/study discussion from the conversation
  in the introduction and Section 2.4 of introduction_and_results.tex.
  Baseline: 046eb0e; the untracked root .claude/ directory was left untouched.
- Added magnetic ladder insulators and the doped (La,Sr)CuO2.5 example;
  Uehara's superconducting chain--ladder material; isovalent Ca substitution,
  pressure-driven charge transfer, competing CDW and magnetic order, and
  distinctions between static order, spin dynamics and pairing symmetry.
  High-Ca commensurate hole crystals prevent a universal no-CDW claim.
- Added the optical provenance of the approximate (8,1,0.1,0.1) parameter
  point, explicitly separating it from a pressure-calibrated material fit.
  V=0 remains a baseline assumption. The Padma RIXS model has a different
  rung hopping, intraladder diagonal hopping, attractive V and U(V) protocol;
  its parameters are not transferred directly to our fixed-U Hamiltonian.
- Added the trellis hopping and reciprocal projected MF kernels, spatial
  rather than iteration-dependent opposite shifts, and one-/two-ladder cell
  restrictions. Linked these motivations to the existing trellis results
  without changing numerical results, convergence thresholds or acceptance.
- Reused shared bibliography records and added 11 primary studies to the
  supplemental bibliography. Checked publisher metadata via Crossref and
  specific primary-source passages; reading depths are recorded in manuscript
  SOURCE_NOTES.md. Restricted-network metadata retrieval initially failed;
  the approved public-metadata request succeeded, with spaced retries after
  rate limiting. No private data were sent.
- Rebuilt using the cached local Tectonic with --only-cached --keep-logs
  --keep-intermediates. Fixed a citation-spacing LaTeX typo found by the first
  compile. Focused validation initially needed to exclude a parameterized
  macro label and enable UTF-8 console output; those checker issues were fixed.
  Final build: 27 pages, 48 references, seven figures, 27 local evidence links,
  no undefined citations/references, missing characters, or over/underfull boxes.
  Inspected the added prose/equations and new references as rendered pages.
- Updated the maintained PDF, README, source notes, current-state snapshot and
  13-file Overleaf bundle. Reused the root-layout compilation check: all 27
  pages match the delivered PDF text. Build/metadata/validation artifacts are
  under ignored output/trellis_manuscript_20260918/. Validation is document-only;
  no DMRG calculation, Perlmutter access, scheduler action or transfer occurred.

### 2026-09-18 — Complete terminal correlations and latest-campaign CPU backfill

- User requested comprehensive equal-time observables for all future SCF runs,
  including maximum-iteration endpoints, and a retrospective measurement script
  for the latest square/cubic two-seed grids, square finer cuts, square (1.2,+0.2)
  four-seed comparison and trellis spatial cells. Baseline: 672bafb. No new
  convergence threshold, iteration limit, acceptance decision or DMRG campaign
  was requested. The root .claude directory remains outside scope.
- The first attempt could not launch a sandboxed command because C: had zero
  free bytes. A read-only capacity check confirmed that, and no files were
  changed in that attempt. After the user freed space, about 41 GB was available.
  The implementation and validation below then proceeded locally.
- Enabled full pair diagnostics in RunSettings, config defaults, all current
  Phase 1 scientific templates, the example SCF config and the frozen-legacy
  preparation path. Trellis now enables the master diagnostic switch. Ordinary
  and trellis drivers save the immutable state first, then measure accepted or
  maximum-iteration endpoints through one shared state-measurement entry point.
  Archived prepared configurations and submitted source trees remain unchanged;
  the deferred finite-size configs can be regenerated from their updated base.
- Reused the existing cached pair transfer sweep, extending it to the complete
  onsite/rung/nearest-leg basis and cross-channel entries. Save complex pair
  addition/removal matrices, one-point expectations, connected subtraction,
  coordinates and class labels; preserve historical class-specific datasets.
  Schema 2 persists raw/connected charge, longitudinal/transverse spin and
  density-spin matrices, spin-resolved Green matrices, anomalous correlators,
  double occupancy, existing connected structure factors and entanglement.
  No dynamic susceptibility or new fixed-sector gap DMRG is silently included.
- The shared reader reconstructs and verifies the stored model fingerprint,
  checks full-source hashes, separates accepted temporal phases from trellis
  spatial A/B samples, and marks unaccepted data as terminal snapshots with the
  original status/period/acceptance. Outputs have measurement implementation
  fingerprints, completion markers, timings and immutable writes. Repeat offline
  measurements reuse only matching complete files. Original states are read-only.
- Added scripts/measure_latest_campaigns.jl and slurm/measure_latest_campaigns.sh.
  Their target is 56 latest branches / 58 spatial MPSs: square 18, cubic 18,
  fine cuts 12, positive-V 4 and trellis 4 (6 MPSs). The two original square V=0
  anchors are superseded by the existing finish20 continuations. Discovery
  requires every final state; it does not substitute a rolling checkpoint.
- The user-run launcher freezes its code and manifest, uses 56 CPU shared jobs
  with four Julia threads/eight logical CPUs/32 GiB each, and a two-hour ceiling
  per MPS (four for two-ladder trellis). The memory-aware nine-core/128 fraction
  yields 8.15625 requested CPU node-hours, bounded by a nine-node-hour local cap
  and the existing 400-additional-node-hour ledger. These are ceilings, not
  measured performance. It invokes the required read-only Phase 0 plan and
  reuses the existing append-only reservation/reconciliation functions.
- Local compact inventory validation found 52 of 56 endpoints with matching
  config/model fingerprints and valid source metadata, including both square
  continuations. The local square lambda08 and other three trellis final states
  remain absent, so the plan correctly exits nonzero listing the missing rows.
  The user reports waiting only for the final trellis run; the older local sync
  does not establish live scheduler state. Full scratch availability and hashes
  must be checked on Perlmutter by the user-run workflow.
- Focused Julia validation: 93 measurement checks (including complex cross-bond
  contractions against independent OpSum MPOs, connected-matrix positivity,
  QN-forbidden expectations, immutability and trellis A/B metadata), five
  four-site square SCF max-iteration checks, 207 trellis checks including actual
  four-site one-/two-ladder max-iteration diagnostics, and nine retrospective
  worker/receipt checks. All passed. These are tiny local DMRG/code checks, not
  scientific convergence or performance evidence for the L=64 chi=200 runs.
- Nine Python launcher checks passed: three new mocked budget/source/duplicate
  checks and six existing handoff regressions. Mock submission verifies exactly
  56 CPU jobs, the 8.15625 ceiling, two four-hour trellis requests, cap rejection
  before submission, no duplicate jobs and source-tampering rejection. An initial
  mock PATH used a Windows drive colon; converting the fixture path to MSYS form
  fixed the test harness. No real scheduler command ran. Julia initially needed
  approved access to its existing precompile cache; later checks completed.
- Updated DIAGNOSTICS.md, README, PROJECT_STATE and ACTIVE with dataset
  conventions, exact Perlmutter handoff, current evidence and interpretation
  boundaries. No full campaign MPS contraction, Perlmutter authentication,
  transfer, real submission, local budget-ledger change or full test-suite run
  occurred. New scientific correlation results await user execution and sync.

## 2026-09-19 — final positive-V square / trellis analysis and LaTeX completion

- User reports all final runs complete and output synchronized, and requests
  completion of the positive-V and trellis analyses, separate comprehensive
  manuscript Sections 3.11/3.12, and spin/pairing figures across both fine cuts.
  Baseline: 9b7d4d5 on codex/mps-mft-phase0-refactor. The root .claude directory
  is unrelated and untouched. No solver, mixing, threshold or launcher change
  was made. Standing user push authorization is retained.
- Reused the existing verified campaign/trellis loaders. Positive-V jobs
  58394103/58394104/58394105/58394106 all have 60-evaluation maximum-iteration
  states, accepted=false. The newly available period-eight compact SHA-256 is
  22b5f44e7d96057f50191d90dd1c88369f7ff3accd46977ee0128d88879a2566.
  Trellis jobs 58468871/58468873/58468875/58468876 all have 60 cell sweeps,
  maximum_iterations, accepted=false. New compact hashes are:
  58468873 acbfff6dc8df900310d77499ef380656b82dc92be864e05bae82fe52dc5e71a5;
  58468875 1b360712c6731f3efda0a68b80faefcc39ebbf0b36001a79819ee877c752e13b;
  58468876 d9a803de57b3acbb6006ab9328173bd93bf7ccf6d30a1053f07e335057420e8d.
  Complete config/seed/state/full-source hashes remain in report inventories.
- All four positive-V square seeds lose anomalous pairing. Period eight ends
  at spin RMS 0.207004, pair RMS 7.005e-9 and six uneven envelope nodes, after
  starting with eight nodes. Dominant charge m=4 and mixed spin m=31/30/28
  support a coarsening/defect interpretation rather than a clean retained
  wavelength. Its endpoint excess over the stripe seed is 9.324203425e-4
  t/site, diagnostic only. Last-ten pointwise spin changes 0.00358–0.00655
  show why nearly constant RMS and energy do not establish stationarity.
- Both skew one-ladder trellis seeds approach the same paired texture,
  pair RMS about 0.017969 and spin about 1.3e-5. Both rectangular A/B seeds
  instead develop unpaired stripes with spin RMS 0.2282–0.2288. Final global
  relative residuals 0.1845–0.2085 remain large; final inner-DMRG windows pass
  for both two-ladder runs. Increment cosines -0.999962/-0.999943 and norm
  ratios about 0.975 identify slowly damped alternating numerical relaxation.
  Two-sweep/one-sweep maximum differences are 0.03215/0.03472, still resolved;
  no accepted orbit, physical dynamics or certified energetic winner follows.
  Spatial A/B labels, their -1/2-rung registration, and different cell/end-cut
  constraints remain explicit. All physical trellis profiles use stored raw
  correlations, not square/cubic Hartree inversion.
- Added complete_campaign_analysis_20260919.py for consistent physical order
  cuts, spin nodes, full-cell lag-one/lag-two diagnostics and accounting.
  The cut figures use the same twenty hashed coarse/fine endpoints as the
  full energy curves: physical leg-odd spin on rungs 6–59, symmetrized
  two-leg-averaged anomalous pairing on bonds with left rungs 6–58, plus
  central rungs 17–48 for spin. Coarse anchors in the older cut_summary were
  corrected to this same physical RMS convention. Linear and log companion
  plots preserve the weak end-weighted V=-0.05 spin qualification. Neither
  the cuts nor their connecting lines establish a first-order discontinuity.
- The updated combined report covers 38 runs / 2280 cell updates / 2400
  individual ladder solves. Newly synced completed-job accounting covers
  all 38: cubic 14.196666667, fine square 13.274722222, positive-V 4.602430556,
  trellis 8.708541667, total 40.782361111 actual node-hours. Recorded solver
  time totals 40.068426596 node-hours. Coarse square remains separate at
  27.350764 actual node-hours. Ledger SHA-256:
  493347fb61caee99a071cf9d19ddb4a4b4b5c6d3c351df9fc432960d6a6b5893.
  Job-level records are in campaign_review_20260918/completion_accounting.json;
  no budget ledger was edited and no remote accounting query was made.
- Updated the two detailed reports and combined review, actual manuscript
  LaTeX/PDF, METHODS_NOTES LaTeX/PDF, literature-review project-evidence
  LaTeX/PDF, source notes, README, PROJECT_STATE, ACTIVE and preparation-result
  pointers. Section 3.10 gains physical order cuts; 3.11 now covers positive-V
  square; 3.12 covers all trellis cells/seeds and their alternating relaxation.
  Square/cubic side-by-side figures and the material introduction are preserved.
  The manuscript is 37 pages / 13 figures / 48 cited references. Numerical
  cutoff is September 19; literature search/bibliographies are unchanged.
  The Overleaf ZIP is verified with all 19 entries and thirteen figure PDFs.
- Local reproduction commands (C:/Python313/python.exe -B -X utf8):
  scripts/analyze_two_basin_campaigns_20260918.py --positive-only;
  scripts/analyze_trellis_progress_20260918.py;
  scripts/complete_campaign_analysis_20260919.py;
  docs/manuscript/package_overleaf.py. These are local analysis/build commands,
  not Perlmutter submissions. Checks validate compact/config/seed hashes,
  within-cell fingerprints, raw adjacency, correlation endpoints, canonical
  reconstruction and site normalization, channel windows, log counts and costs.
  Input state hashes match after analysis. No DMRG test suite was rerun.
- Cached Tectonic builds succeed for manuscript (37 pages), methods (21) and
  literature review (30). All thirteen manuscript graphics and text references,
  48 bibliography entries, and ZIP contents are verified. No unresolved
  references, missing figures or overfull boxes occur. Minor underfull spacing
  remains in a long energy paragraph and existing methods tables; rendered
  pages were inspected without clipping. Local QA/summary/render files are
  retained under output/notes_update_20260919. Build intermediates are kept
  there rather than added to versioned documentation.
- This completes the requested endpoint analysis and notes. Full terminal
  correlations/backfill remain prepared, not newly measured campaign evidence.
  No GPU port, extra compute, submission, authentication, synchronization or
  scheduler operation was performed. Selective stability/convergence work and
  any modest linear-damping trellis test remain future user decisions.

## 2026-09-20: leg parity, transverse stripe registration and iteration cycles

- User asked whether trellis frustration persists with two spatial ladders,
  how Bollmark's temporal two-period construction relates to these results,
  whether paired square points need an A/B test, and why charge could not
  alternate between the two legs of one ladder. Baseline: 13b289d on
  codex/mps-mft-phase0-refactor. No simulation controls or outputs changed.
- Read the actual Geometry, MeanField and Trellis kernels and archived square
  controls. Both trellis cells keep tau0/tau1 zigzag bonds and triangles.
  Skew one-ladder and rectangular A/B repetition impose different longitudinal
  registrations and finite OBC cuts. Two skew repetitions translate by
  (-1,2), versus rectangular (0,2). The former's temporal two-cycle therefore
  need not reproduce the latter's static spatial pattern.
- Direct raw-correlation audit of jobs 58468871/58468873/58468875/58468876:
  in central rungs 17-48 the two-ladder charge-even modulation is
  0.046667-0.046984, while RMS[(n0-n1)/2] is only 5.006e-6 to 6.506e-6.
  One-ladder leg-odd charge is about 1.072e-5. The solver allows independent
  leg densities; small observed leg-odd charge is not an explicit stability
  test. Alternating rung charge between ladders is a different mode from
  polarizing the two legs within a rung. No relative-charge gap was measured.
- At qx=pi/8, with physical x_A=i and x_B=i-1/2, the two-ladder endpoint
  A/B-odd charge weights are 0.814554/0.788119 (full) and
  0.829795/0.794464 (central), for stripe/pairing starts respectively.
  Central relative phases are 131.2699/126.0837 degrees, so both even and odd
  components remain. Central odd weights across the final ten sweeps range
  0.76332-0.88698 and 0.71579-0.85498. These are nonstationary OBC profiles,
  not pure transverse-momentum eigenstates or accepted phases. Leg parity
  from old two-leg Fourier plots is distinct from A/B ladder parity.
- The browser-accessible arXiv:2301.08116 preprint, Sec. III A/Fig. 2, was
  checked for Bollmark's density-avoidance two-cycle mechanism. Published
  DOI: 10.1103/PhysRevB.111.125141 (2025). Direct published-PDF retrieval
  failed; the note explicitly links the accessible preprint as the text read.
  For a correct same-map bipartite extension, X_A=R(X_B), X_B=R(X_A) is
  both a stationary cell and a raw two-cycle. Clarified in METHODS_NOTES.tex
  that R composed with itself has lambda squared, while a simultaneous
  one-sweep cell has Jacobian [0 J; J 0] and eigenvalues plus/minus lambda.
  Complete raw linear stability therefore tests both parities in that
  specific extension; the observed few seed trajectories are weaker evidence.
- Square archived anchors and remainder both have damping=1 and
  accepted_periods=[1,2]. Their paired outcomes remain useful evidence.
  Recommended targeted A/B tests at (1.4,-0.2) and (1.4,-0.4), with paired
  plus explicit transverse perturbations and translated-stripe starts; a
  leg-odd charge perturbation is a distinct useful control. Verify A=B fields
  and per-site energy against actual square bonds first. Rectangular trellis
  at tau1=0 retains a shifted interface and is not a drop-in square A/B map.
  Cubic cell tests remain useful for registration/energy, with lower priority
  for the paired/stripe boundary and explicit neighbor assignments required.
- Added scripts/analyze_trellis_transverse_sectors_20260920.py and linked
  TRANSVERSE_INTERPRETATION_20260920.md plus transverse_sectors_20260920.json
  in the trellis report. Updated PROJECT_STATE and ACTIVE. Numerical JSON
  records all four compact paths/hashes and all 60 Fourier evaluations.
  Verified compact hashes before and after reading:
  58468871 b3d204a10eeeced5446f22a7b65ffef8314cf5243e49d8bceae11046a1db7b00;
  58468873 acbfff6dc8df900310d77499ef380656b82dc92be864e05bae82fe52dc5e71a5;
  58468875 1b360712c6731f3efda0a68b80faefcc39ebbf0b36001a79819ee877c752e13b;
  58468876 d9a803de57b3acbb6006ab9328173bd93bf7ccf6d30a1053f07e335057420e8d.
- Local reproduction: C:/Python313/python.exe -B -X utf8
  ladder_mps_mft/scripts/analyze_trellis_transverse_sectors_20260920.py.
  Focused checks cover density normalization, endpoint/history equality,
  origin phase, even/odd Parseval identity, pure parity examples and bounded
  finite weights. No new DMRG calculation, susceptibility estimate or full
  solver suite was needed. All original acceptance flags remain false.
- Rebuilt only the living methods PDF with cached Tectonic: 22 pages;
  new equation/paragraphs and neighboring pages 8-10 visually checked.
  No missing references, overfull boxes or clipped material. Existing
  underfull table/paragraph warnings remain. Scratch QA/build files are
  under output/notes_update_20260920. An initial multi-file patch rejected
  a stale context without partial edits, then applied successfully. PDF text
  checks needed tolerance for extraction kerning in the word Trellis;
  semantic checks and rendered pages pass. Manuscript/Overleaf retain their
  September 19 completed results and were not rebuilt for this methods note.
- No new jobs, GPU port, budget changes, authentication, synchronization,
  scheduler actions or terminal correlation measurements were performed.
  The proposed cell/stability calculations remain future user decisions.

## 2026-09-20: physical interpretation of one- versus two-ladder trellis

- Follow-up asked whether two ladders are more physical and why this was
  not an initial phase-selection control. Rechecked the implemented kernels
  and existing geometry/reciprocity tests; no solver tests or simulations
  were rerun. Baseline commit: 1f08a8d.
- Clarified the physical variable: relative stripe registration between
  neighboring ladders changes transverse interaction energy, while a common
  translation mainly changes absolute position/boundary pinning. The
  stationary skew one-ladder repetition fixes the former geometrically.
  It should have been identified as a limitation of the original phase
  comparison, beyond the initial motivation of numerical translation drift.
- For a homogeneous infinite-bulk correlation matrix, longitudinal
  translations commute with that matrix and forward/backward kernels agree.
  Nonzero-wavevector stripes instead depend on registration. Thus a spatial
  restriction can bias the competition against stripes relative to uniform
  pairing. This is a kernel-level interpretation, not a newly measured
  susceptibility or proof of the mode initiating the observed instability.
- Independent A/B profiles are a minimum useful phase-selection control.
  Both cells preserve the microscopic zigzag frustration and use the same
  product-of-ladder-MPS approximation. Current finite skew/rectangular cells
  are not strictly nested, so larger cell size alone does not prove a
  variational ordering of their endpoints. Strong central stripes make the
  observed difference more than a displaced copy of the paired texture,
  but boundary-driven selection remains possible.
- Added a physical-interpretation section to the existing trellis note and
  a short PROJECT_STATE clarification. Recommended a future same-cell
  comparison initialized from the actual paired trellis endpoint, followed
  by weak relative-stripe perturbations and comparison with a stationary
  stripe branch. This separates paired-branch instability from competing
  basins. No code, run controls, original artifacts, acceptance flags,
  accounting, PDFs or submission plans changed; no Perlmutter actions.

## 2026-09-20: direct trellis trial-energy comparison and diagonal-stripe cells

- User challenged whether nonnested finite cells actually preclude comparing
  striped and paired trial energies, then proposed the diagonal stripe in an
  attached slide as a candidate requiring more ladders. Baseline: fcd175a.
  Corrected the earlier overly broad energy qualification: a common-functional
  expectation value can be compared before stationarity. Acceptance still
  governs stationary-branch claims; original flags remain false.
- Added scripts/audit_trellis_same_cell_energy_20260920.jl. It loads the
  actual array kernels without the DMRG solver, reads saved bare energies
  and correlations, and duplicates each paired state in the rectangular
  cell. It recomputes all interaction fields, never copies applied fields
  into an energy comparison. Common L=64, U=8, V=0, t=t0=1, tau0=tau1=0.1,
  density=15/16, r_range=4 and |E_p|=0.13251724 were checked across all four.
- Recomputed original fields agree exactly; canonical/target energies agree
  within 1e-12 t/site. Paired sources 58468871/58468873 give rectangular
  target energies -0.518817872689/-0.518817859078. The embedding shift is
  +2.46378e-6 t/site. Saved striped sources 58468875/58468876 give
  -0.521125816792/-0.521055560230 in that same cell. The four same-cell
  target gaps favor the striped trials by 0.002238-0.002308 t/site.
  Uncorrected canonical gaps favor stripes by 0.002229-0.002301; maximum
  target-density correction is 8.32448e-6. This correction is not exact
  number projection, but does not determine the observed ordering.
- Numeric output is docs/reports/trellis_progress_20260918/
  same_cell_energy_audit_20260920.toml. Source hashes checked before/after:
  58468871 b3d204a10eeeced5446f22a7b65ffef8314cf5243e49d8bceae11046a1db7b00;
  58468873 acbfff6dc8df900310d77499ef380656b82dc92be864e05bae82fe52dc5e71a5;
  58468875 1b360712c6731f3efda0a68b80faefcc39ebbf0b36001a79819ee877c752e13b;
  58468876 d9a803de57b3acbb6006ab9328173bd93bf7ccf6d30a1053f07e335057420e8d.
  The output also records Trellis.jl and Variational.jl SHA-256 values.
- Local command: Julia 1.12.7 --startup-file=no --compiled-modules=existing
  --project=ladder_mps_mft ladder_mps_mft/scripts/audit_trellis_same_cell_energy_20260920.jl.
  Used the installed runtime directly because the WindowsApps alias was
  inaccessible. Initial Julia script header was changed from a docstring to
  a block comment after Julia rejected documenting a using statement; the
  corrected audit completed successfully. No DMRG or unchanged solver tests.
- Current ABAB phase offsets near 126-131 degrees alternate in sign; they
  do not establish a continuing tilted stripe. For a rectangular n-ladder
  cell, a constant charge phase advance obeys n theta=2 pi p. Rigid shifts
  require n d to close both charge and full spin textures, not only charge.
  Nominal periods 16/32 with d=2 imply n=16 without a further spin operation;
  this is a conditional example, not a run recommendation or measured tilt.
  Actual translations include shear and the leg basis. A properly defined
  translated cell may test some tilts with fewer independent profiles; OBC
  prevents treating this as a simple circular array roll.
- Consulted the primary Miyazaki/Yanagisawa/Yamaji JPSJ 73, 1643 (2004) PDF
  at https://staff.aist.go.jp/t-yanagisawa/activity/JPSJ-Miyazaki04.pdf.
  Its U=8, t'=-0.2 square-lattice VMC favors bond-centered diagonal stripes
  near hole doping 1/16. Recorded as motivation in the transverse note,
  not a prediction for the different weakly coupled trellis Hamiltonian.
- Updated the trellis and combined reports, transverse interpretation,
  method contracts, PROJECT_STATE, ACTIVE, manuscript source/evidence notes
  and living methods LaTeX. Rebuilt both PDFs using cached Tectonic: methods
  22 pages; manuscript 37 pages, thirteen figures and 48 cited references.
  The first methods build required an uncached bold-math font; used ordinary
  vector accents for the new Q dot T expression, then rebuilt successfully.
  No undefined references or overfull boxes; existing underfull warnings
  remain. Visually checked methods pages 20-21 and manuscript pages 28-30.
  Scratch builds/renders are under output/energy_comparison_20260920.
- No SCF settings, geometries, original data, acceptance flags, accounting,
  submissions or measurement controls changed. No Perlmutter connection or
  scheduler action. Further relaxation and larger-cell tests remain proposals.

## 2026-09-20: square two-ladder, two-seed comparison prepared

- User authorized the analogous unfrustrated square A/B test at (t0,V)=
  (1.4,-0.4) and (1.4,-0.2). Baseline commit: 5a590f3. Prepared exactly four
  jobs; no scheduler submission, transfer, authentication or allocation
  action was performed locally. Existing scientific states and accounting
  ledgers remain unchanged. Unrelated root .claude/ remains untouched.
- Added model.spatial_cell=two_ladder for square and reused the simultaneous
  spatial-cell driver. The new kernel applies the existing square map to
  the opposite ladder: F_A=K(C_B), F_B=K(C_A). Opposite legs meet at equal
  rungs, with no diagonal hopping, shear, wrapped physical bond or extra
  coordination factor. A=B reproduces the original finite-OBC fields and
  canonical energy per site. Each independent two-leg MPS has L=64 and
  its own density target 15/16; mean charge transfer between ladders and
  quantum entanglement between the two MPSs are outside this approximation.
- Seeds retain the established 95%/5% stripe/pairing mixtures. A uses the
  original stripe reference; B uses an eight-rung displacement of that
  reference, half the nominal charge wavelength. The pairing template is
  unshifted and has the same sign in A/B. Initial fields are rebuilt with
  target couplings from the opposite member's template. Fresh MPSs use
  random/product-state seed 1404. This explicitly excites A/B asymmetry;
  identical square templates would remain in an invariant A=B subspace.
- The initial stripe displacement permutes both correlation-matrix indices
  and density vectors cyclically. This preserves number and matrix symmetry
  but relocates the finite-reference end structure too. It is only a seed
  operation, not a translated boundary condition or physical periodic bond.
  Eight rungs is one trial registration, not a measured optimum or pinning
  constraint. All charge, spin, pairing and exchange fields remain free.
- Reference bundle SHA-256:
  e01a1ea7d6be813110870d26377db0df529d1584816c946af78584e1e1fbddc1.
  Stripe source: ae6a3bfe76ca8f06f2396fd731b18bca8539e0b7ee68df016cc9156fdceeb074.
  Pairing source: 8a1cf2d64d2fbe0eb59521192b829cab43e19a4d7ac026519ea847f6ac0778b8.
  Both target E_p values are exact registry entries: -0.24962435880865996
  and -0.2068002629740704 respectively. No interpolation/new bare-ladder run.
  Registry SHA: 2209bd2ca3c1ad02c0e542d1a9d63ecf90fdfa49120ad9cc3af599a5b4bc1f0e.
- Controls: chi=200, t=1, U=8, tp=0.1, r_range=4, float64 dense GPU MPSs;
  60 maximum cell sweeps, minimum 40, ten stable records; simultaneous raw
  updates, damping=1, no Anderson and fixed-point acceptance only. Retained
  field tolerances 1e-7/1e-4, channel noise floor 5e-7, energy window
  1e-7 t/site, inner-DMRG 1e-7 total and density tolerance 1e-5 per ladder.
  At most 480 density-targeted ladder solves across the four jobs; chemical
  potential targeting can require multiple inner DMRG solves per ladder.
- Each sweep saves separate A/B full histories and current simultaneous
  variational energies, with 4L cell normalization and established double
  counting. New artifacts use spatial_cell_mps_mft_state. Diagnostics,
  compact mirrors, full-cell resume and history plots handle both members;
  historical one-ladder/trellis model fingerprints retain their definitions.
  The saved-model reader in the prior energy-audit script now defaults
  missing fields so the added model setting does not break old inputs.
- Complete intraladder terminal measurements are enabled for accepted and
  maximum-iteration states on both A/B: raw/connected pair, charge, spin,
  density-spin, single-particle/anomalous, double occupancy and entanglement.
  The existing measurement implementation uses CPU and writes sidecars after
  the immutable full state. Solver time-limit exits remain eligible for
  explicit offline measurement but do not automatically start this pass.
- Launcher v1.24.0 adds prepare-square-two-ladder and a four-branch contract.
  slurm/submit_square_two_ladder.sh imports the original anchor run.env,
  prepares, reconciles and submits through existing project guards when run
  by the user. It retains shared account/scratch/ledger settings, a single
  segment and the 400 additional-node-hour cap. New default run ID:
  20260920_square_two_ladder_two_basin_60. Each job requests one shared GPU,
  32 logical CPU cores and 16 hours, with an 11.5-hour SCF deadline leaving
  4.5 hours for two measurement passes. Total reservation ceiling: 16 GPU
  node-hours; no automatic continuation. Measurement allowance is not a
  benchmark. Checked current NERSC policy (shared GPU permits up to 48 h;
  one GPU is a quarter node): https://docs.nersc.gov/jobs/policy/.
- Local validation, Julia 1.12.7 with --startup-file=no and
  --compiled-modules=existing --project=ladder_mps_mft:
  SQUARE_CELL_DMRG_SMOKE=1 test/test_square_two_ladder.jl passed 1107 checks
  (999 field/energy/boundary checks, 67 preparation/compatibility checks,
  41 tiny CPU driver/storage/terminal-measurement/resume checks). The tiny
  DMRG section took 1m43s. Its first provenance lookup encountered Windows
  git ownership warnings and used the existing unknown-commit fallback;
  later checks supplied a process-local safe.directory setting.
- Existing test_trellis.jl with TRELLIS_DMRG_SMOKE=1 plus
  test_state_diagnostics.jl passed 305 checks: 150 kernel/preparation,
  57 tiny spatial-driver, 93 complete equal-time measurement and five
  automatic max-iteration checks. Existing trellis tiny solves took 1m46s;
  the complete equal-time checks took 1m27s. No GPU timing or scientific
  convergence is established by these local tests.
- With MPLBACKEND=Agg and --project=@v1.12, test_phase1_plotting.jl passed
  119 checks in 38s, including square A/B seeds, full histories and slider
  values. Python unittest discovery for test_square_two_ladder_launcher.py
  passed three tests; test_two_basin_next_launchers.py passed six. The local
  Git Bash executable was used only for syntax/fake-launcher checks, with
  no Slurm access. No unchanged expensive full solver suite was rerun.
- Ran scripts/prepare_phase1_square_two_ladder.jl with the new config,
  data/two_basin_references.h5 and output/square_two_ladder_validation/
  preview/full directories. It produced four immutable preview configs and
  seeds, verified matching fingerprints within each point and wrote the
  seed contract. A following search for a nonexistent root Project.toml
  returned nonzero after successful preparation; plotting uses the global
  Julia environment instead. Logs and preview are retained under that
  ignored validation directory; the temporary test runner was removed.
- Added the preparation report and portable four-row prepared_branches.csv
  under docs/reports/square_two_ladder_20260920; updated current state,
  active plan, architecture, cell method notes, diagnostics and docs index.
  Existing result reports and LaTeX/PDF results were not changed because
  this is a run preparation, not new scientific evidence. User handoff:
  cd "$CFS/m4863/MPS-MFT/ladder_mps_mft"; git pull --ff-only;
  module load julia; bash slurm/submit_square_two_ladder.sh.

## 2026-09-21: first synced square A/B endpoint reproduces the paired state

- User reported one completed run and requested comparison with the original
  single-ladder result. The initial sync was on another device; after the
  user corrected it, the local campaign contained job 58654275's state and
  two diagnostic sidecars, plus partial stdout for 58654274. Baseline adba8c2.
  No Perlmutter connection, scheduler action, sync or simulation by Codex.
- Completed branch: pairing seed, square (1.4,-0.4), chi=200. Saved status
  fixed_point, accepted=true, period one at 40 cell sweeps / 80 ladder
  evaluations. Recomputed both members' final ten-record channel, field,
  slow-mode, density, inner-DMRG, energy and consistency gates: all pass.
  This is the minimum permitted sweep count, not the 60-sweep cap.
- Physical leg-pair RMS A/B=0.0352250769/0.0352251722; rung means
  -0.0532419724/-0.0532421064. Spin RMS falls from 0.00845/0.00804 on the
  first evaluation to 5.369e-7/2.519e-7, without a resolved late growth.
  The full-length A-minus-B spin RMS falls from 0.01176 to 3.187e-7;
  maximum final A/B density and leg-pair differences are 7.31e-6/1.08e-6.
  Profiles match the existing paired phase, with the same open-end structure.
- Simultaneous-cell corrected canonical energy is -1.167404800545 t/site,
  with a final ten-record span of 3.218e-10. Relative to the old pairing
  and stripe seeds, differences are +7.727e-9/+2.416e-10 t/site. Uncorrected
  canonical energy is -1.167406636373; the larger +1.118e-6/+2.478e-6 raw
  differences accompany slightly different numerical densities. Identical
  physical parameters and E_p, and 256/128-site normalization, were checked.
  Old 80-step states remain unaccepted under their tighter archived controls;
  new acceptance does not demonstrate an intrinsically faster larger cell.
- Stripe-start stdout (58654274) independently reports fixed_point at 40
  and -1.167404800596 t/site, within 5.1e-11 of the completed seed, then
  begins measurements. There is no synced stripe terminal artifact or
  completed-measurement record, so this is log-only energetic corroboration.
  No V=-0.2 results are synced. No claim of all-seed or all-wavevector
  stability follows from the first completed paired endpoint.
- Verified compact/full lineage, config/seed hashes, manifest fingerprints,
  all raw update links, terminal/history agreement, physical density,
  direct energy components, target-density correction and cell energy sums.
  State compact SHA: 5edf0d7de2c959446aeb675c62df87b1b97fd123c6a3cc6dbad9d739c14ee5e8.
  Full-state SHA: f02014387006d2b0d4f8c21c7b177c7d1415fb908f35012e900ec4ad3c5566de.
  Config SHA: bdd1c3708acb5412aab488cb1f39618eff11da4b7e6230cf3bd315bd8d2140ec.
  Seed SHA: 16e34bf339abf7d07707121bd23bfe57908c59242f63e7798a4438502a04dac8.
- Both full-correlation sidecars are complete at sweep 40, linked to that
  same full-state hash. Their compact hashes are
  A: 40a14addf8771cde64ca36cc56a43da4c43aa19ee402db016a8b6e47b167458d;
  B: 74906a93ad930652688bbbe578e1e2e4f8ec287de827aecba459cee6664a33a0.
  Checked terminal densities and raw-minus-disconnected identities for
  charge, spin and the 318-operator singlet-pair matrix. These are intraladder
  measurements; no matching old full-correlation comparison is claimed.
- Solver time 16531.96 s; A/B measurement time 5950.30/5964.31 s. Combined
  recorded work is about 7.90 hours, or 1.9755 shared-GPU node-hours before
  overhead. No completed-job reconciliation is synced for these job IDs;
  exact allocation cost remains unavailable and ledgers were not changed.
- Added scripts/analyze_square_two_ladder_20260921.py, reusing the archived
  anchor reader. Local Python 3.13 command with -B -X utf8 completes in
  about three seconds. Initial run exposed the older history key
  variational_energy rather than canonical_variational_energy; corrected
  the reader, reran successfully, then added explicit E_p equality checks
  and reran the affected analysis once. No DMRG or solver suite was needed.
- Wrote FIRST_RESULT_20260921.md, numerical JSON and one six-panel PNG in
  docs/reports/square_two_ladder_20260920. Visually checked full histories,
  legends, labels and overlapping endpoint profiles; residual-spin scale is
  explicitly 1e-6 and energy uses a labeled symlog axis. Updated the campaign
  index, docs index, PROJECT_STATE and ACTIVE. This is an interim single-point
  readout; no manuscript/methods PDF rebuild or submission changes. Original
  data and flags remain immutable, and unrelated .claude/ remains untouched.

## 2026-09-21: retrospective correlation submission failed before measurement

- User reported submitting the retrospective correlation script and asked
  whether pair correlations were now available for all data. Inspected the
  local synchronized output/phase1_diagnostics/20260918_latest_correlations
  manifest, job records, all logs and output inventory; no remote access.
- Manifest contains 56 branches / 58 spatial MPSs. jobs.tsv records 56
  submissions, and all 56 corresponding log files are present. Every log
  consists of the same startup error (with its own job ID):
  `/var/spool/slurmd/job.../slurm_script: line 5: /var/spool/slurmd/job.../phase1_gpu.sh: No such file or directory`.
  Representative jobs: 58586747 (row 1) and 58586805 (row 56).
- The frozen source/slurm/measure_latest_campaigns.sh sources phase1_gpu.sh
  relative to BASH_SOURCE[0] before dispatching _run. Slurm executes its
  spooled copy, so that sibling lookup points into /var/spool/slurmd rather
  than the saved source directory. Failure precedes Julia/MPS contractions.
- Local coverage: no results directory, zero diagnostic HDF5 files and zero
  measurement_receipt.toml files for this campaign: 0/56 branches, 0/58 MPSs.
  The two complete A/B sidecars from square pairing job 58654275 are separate
  and remain available. Older diagnostic files do not replace this backfill.
- Manifest SHA-256:
  6fab4585b0887a17ac310b0d01349bbde0398992aab32c6f4c05cadfa0fbab0f.
  jobs.tsv SHA-256:
  82bd445f16fe441eb7c7465fcbfe4bc911a17d8a9f39d32cfad1e35b8c537972.
- Validation was a PowerShell inventory and exact error-pattern comparison
  across all 56 logs, with manifest/job counts and file hashes. No DMRG,
  correlation contractions, scheduler checks, transfers or accounting edits.
  This establishes the synced submission's startup failure, not live status
  or billed cost. Updated PROJECT_STATE and ACTIVE to supersede the earlier
  prepared/awaiting-sync wording.
- No launcher code or frozen source was changed in this status-only task.
  Recovery requires a focused launcher repair and user-managed retry after
  accounting reconciliation and source checks. The existing submit command
  skips already recorded jobs; repeating it unchanged cannot retry them.

## 2026-09-21: measurement startup corrected and separate retry handoff prepared

- User authorized correcting the failed jobs and preparing submission; also
  reported two new square A/B runs still ongoing. That live status is
  user-reported; identities were not inferred and no scheduler was accessed.
- Changed slurm/measure_latest_campaigns.sh so its _run worker dispatches
  from the explicit frozen run-directory argument before importing sibling
  submission helpers. The Slurm spool copy no longer needs phase1_gpu.sh
  beside itself. Retained manifest/source hash checks, Julia/BLAS thread
  controls, srun arguments, source project and measurement worker. No Julia
  measurement implementation or scientific settings changed.
- Added slurm/retry_latest_correlations.sh plan|submit. Defaults:
  parent=20260918_latest_correlations,
  new=20260921_latest_correlations_retry1. It verifies the parent snapshot,
  exactly 56 jobs / 58 MPSs, no parent results directory and the known startup
  failure in every log. Plan reads only. Submit reconciles only the parent
  measurement campaign and requires terminal failure accounting for all jobs.
- The existing inventory/preparation machinery builds the new manifest,
  which must match the parent byte for byte before any sbatch call. The new
  measurement.env and retry_parent_manifest.sha256 preserve the parent link.
  New source/log/results/job records are separate; resuming an interrupted
  retry skips recorded submissions. Old frozen source, manifests and job
  history remain unchanged. No automatic resubmission of failed retry jobs.
- CPU resources and ceilings remain: 56 jobs, four Julia threads, eight
  logical Slurm CPUs, 32 GiB per job, two hours/MPS (four for A/B trellis),
  8.15625 requested CPU node-hours, nine-node-hour measurement cap and the
  existing shared 400-additional-node-hour cap. Other campaigns' reservations
  remain included. The four new square A/B branches are outside this fixed
  backfill inventory and are not submitted, continued or reconciled here.
- Local Python 3.13 command:
  `python -B -m unittest discover -s ladder_mps_mft/test -p test_measurement_launcher.py -v`
  passed nine tests in 177.046 seconds. Fixtures use the confirmed local
  Git Bash executable and fake sbatch/sacct/srun/module/Julia commands only.
  Checks cover spool execution without siblings, worker exit propagation,
  hash failure before execution, parent immutability, byte-identical retry
  manifests, duplicate prevention, terminal accounting, changed inventory,
  recognized failure, existing-output rejection and retained project caps.
  The fake accounting test queried all 56 parent jobs and never the running
  square fixture; its reservation remained in the shared ledger.
- Read-only local inventory command:
  `julia --startup-file=no --compiled-modules=existing --project=ladder_mps_mft ladder_mps_mft/scripts/measure_latest_campaigns.jl plan ladder_mps_mft/output/phase1_gpu --local`
  completed with ready_states=56/56 and mps_measurements=58. This checks
  compact states/configuration provenance, not current full scratch-MPS
  availability, live accounting or successful scientific measurements.
- Original manifest/jobs hashes remain respectively
  6fab4585b0887a17ac310b0d01349bbde0398992aab32c6f4c05cadfa0fbab0f and
  82bd445f16fe441eb7c7465fcbfe4bc911a17d8a9f39d32cfad1e35b8c537972.
  Updated measurement launcher SHA-256:
  6584b07a6fd33939d00629b607a0b7e9cdd90765285ed1e6de1924d731b91ddc.
  New retry wrapper SHA-256:
  d357015c221cf9751ffc55f41b2508c668678f8f1194221758683bf0dddf3dd0.
- Added docs/reports/correlation_retry_20260921/README.md and updated
  DIAGNOSTICS, PROJECT_STATE and ACTIVE. git diff --check passed. No broad
  DMRG/measurement suite was rerun for this shell-only repair. No remote
  access, transfer, real submission, scheduler/accounting operation or
  modification of original SCF/measurement artifacts was performed.
- User handoff after synchronizing the two updated/new launchers:
  `cd "$CFS/m4863/MPS-MFT/ladder_mps_mft"`, `module load julia`,
  `bash slurm/retry_latest_correlations.sh plan`, then
  `bash slurm/retry_latest_correlations.sh submit`.
  Source transfer and every live action remain user-managed. No commit or
  push was performed in this preparation task; unrelated .claude/ is untouched.

## 2026-09-21: restore the git-pull handoff for the correlation retry

- User explicitly requested committing and pushing the prepared changes so
  synchronization on Perlmutter is the usual `git pull`. Publication target
  is the existing origin/codex/mps-mft-phase0-refactor branch at
  https://github.com/kwang0/MPS-MFT.git. This supersedes the preceding
  local-only handoff; it does not authorize any Perlmutter connection.
- Updated the retry README and current-state handoff to start with
  `cd "$CFS/m4863/MPS-MFT/ladder_mps_mft"` and `git pull --ff-only`, then
  module load julia and the existing retry plan/submit commands. No manual
  transfer of individual scripts is needed after the release is pushed.
- Release scope is the corrected launcher, dedicated retry wrapper, focused
  launcher tests, recovery README and associated status/log documentation.
  Unrelated .claude/ remains excluded. Source changes are identical to the
  nine passing launcher tests; only handoff documentation changed afterward,
  so no unchanged DMRG or launcher suite is repeated.
- The user-reported two ongoing square A/B jobs and all frozen campaign
  artifacts remain outside the retry. Git publication and Perlmutter
  submission are separate: only the user runs the latter.
