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
