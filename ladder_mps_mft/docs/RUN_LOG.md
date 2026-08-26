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
