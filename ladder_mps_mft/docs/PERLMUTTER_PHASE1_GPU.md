# Perlmutter Phase 1 GPU workflow

This workflow runs the refactored solver in Float64 on one dense CUDA GPU. It does **not**
restore the legacy SCF code or its automatic resubmission loop. Convergence,
unmixed periodic-orbit detection, Anderson fallback, common variational energy,
portable checkpoints, and provenance all remain those of `LadderMPSMFT`.

## One-time setup after pulling the commit

From `ladder_mps_mft/` on a Perlmutter login node:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
JULIA_PKG_PRECOMPILE_AUTO=0 julia --project=gpu \
  -e 'using Pkg; Pkg.instantiate(; allow_autoprecomp=false)'
bash slurm/phase1_gpu.sh plan
```

The committed GPU manifest pins CUDA.jl 5.9.5 together with ITensors 0.9.15 and
ITensorMPS 0.3.25. Do not import or precompile CUDA on the GPU-less login node:
the allocated smoke job performs the first CUDA import and is the device proof.
CUDA.jl is configured to use its pinned artifact toolkit, so do not load the
Perlmutter `cudatoolkit` module for this workflow. The launcher unloads that
module, removes inherited NVIDIA-HPC-SDK runtime-library paths, and aborts if a
non-artifact cuBLAS, cuSOLVER, cuSPARSE, or other CUDA runtime library is loaded.
This follows [CUDA.jl's recommended artifact-toolkit
configuration](https://cuda.juliagpu.org/stable/installation/overview/); the
Perlmutter system toolkit remains appropriate for software compiled against
it, but must not be mixed into this artifact-based Julia process.

## Staged submission

Choose a new immutable run ID:

```bash
RUN_ID=20260823_phase1_gpu_v2
bash slurm/phase1_gpu.sh prepare-standard "$RUN_ID"
bash slurm/phase1_gpu.sh submit "$RUN_ID"
bash slurm/phase1_gpu.sh status "$RUN_ID"
```

`prepare-standard` creates all nine configs without submitting or reserving
anything. `submit` accepts only an existing prepared campaign and submits only
a 30-minute smoke test; it can no longer create a default campaign implicitly.
Literal placeholders such as `RUN_ID` are rejected for submission. The smoke
is not a scientific calculation. It requires artifact-only CUDA
runtime libraries, exercises a 256-by-256 dense GPU matrix multiplication and
Hermitian eigendecomposition through cuBLAS/cuSOLVER, runs the tiny DMRG, and
round-trips its MPS through HDF5. After `status` reports the smoke job as
`COMPLETED` and `gpu_smoke.h5` exists, inspect the smoke log for the recorded
preflight and submit the branch matrix:

```bash
grep -E 'gpu_smoke_path|linalg_preflight_dimension|tensor_scalar_type|CUDA runtime library.*system path' \
  "output/phase1_gpu/$RUN_ID"/logs/smoke-*.out
bash slurm/phase1_gpu.sh submit-matrix "$RUN_ID"
bash slurm/phase1_gpu.sh status "$RUN_ID"
```

The corrected smoke must print `linalg_preflight_dimension=256` and
`tensor_scalar_type=float64`, must not print a `system path` CUDA-runtime
warning, and must save a Float64 MPS. `submit-matrix` validates all of these
properties from `gpu_smoke.h5`; scheduler completion alone is insufficient.

The matrix contains pairing, SDW, and CDW seeds for each of
`cubic_frustrated`, `cubic_unfrustrated`, and `square`. Each job requests one of
four GPUs for 12 hours in shared QOS, a conservative charge of 3 GPU node-hours.
The smoke plus matrix reserves 27.125 node-hours. NERSC shared jobs are charged
by the dominant node fraction; see the [NERSC queue and charge
policy](https://docs.nersc.gov/jobs/policy/#calculating-charges).

Launcher version 1.3.0 writes all MPS-bearing branch artifacts to Perlmutter
scratch and automatically creates MPS-free analysis mirrors on CFS. The same
policy applies to guarded CPU `E_p` calculations: `psi_N_*` sectors remain in
scratch while energies and metadata are mirrored to CFS. Continuation and
recovery resolve the full scratch files; plotting, audits, and field-only
inheritance use the lightweight files. See `PERLMUTTER_STORAGE.md` for the
layout, migration commands, verification, and the scratch-purge boundary.

Run `20260822_phase1_gpu_v1` is retained only for audit: its smoke passed despite
CUDA.jl warnings, but all nine branches then mixed CUDA.jl artifact libraries
with CUDA 13.2 libraries from the NVIDIA HPC SDK and segfaulted in the first
DMRG eigendecomposition. It produced no result or checkpoint files. Script
version 1.0.1 can display its status but refuses new submissions or
continuations from that run; use a new run ID.

Run `20260823_phase1_gpu_v2` completed all nine scientific jobs, but its MPS and
MPO tensors were silently converted to Float32 by the opinionated `CUDA.cu`
adaptor. All nine states fail the configured Hamiltonian-consistency gates, and
none is accepted. Preserve v2 unchanged. The corrected device path explicitly
promotes CPU checkpoints and Hamiltonians to the configured scalar type, then
uses NDTensors' type-preserving CUDA adaptor.

Recover v2 without repeating its independent-seed transient:

```bash
SOURCE_RUN_ID=20260823_phase1_gpu_v2
RUN_ID=20260824_phase1_gpu_v3_float64
bash slurm/phase1_gpu.sh prepare-recovery "$SOURCE_RUN_ID" "$RUN_ID"
sed -n '1,12p' "output/phase1_gpu/$RUN_ID/manifest.tsv"
bash slurm/phase1_gpu.sh submit "$RUN_ID"
bash slurm/phase1_gpu.sh status "$RUN_ID"
```

`prepare-recovery` creates and hashes all nine controls but performs no Slurm
submission and makes no budget reservation. The older one-command
`submit-recovery SOURCE_RUN NEW_RUN` interface remains available and is
equivalent to preparation followed by `submit NEW_RUN`.

The recovery manifest hashes every immutable v2 `state.h5`, records its status,
numerical fingerprint, and Float32 storage type, and uses it as a parent seed.
The new numerical fingerprint is intentionally different because the MPS is
promoted to Float64. Each branch starts with a fresh raw-map probe, so the v2
mixer-dependent recurrences are not inherited as physical solutions. After the
new smoke passes the Float64 gate:

```bash
bash slurm/phase1_gpu.sh submit-matrix "$RUN_ID"
```

Preparation and matrix submission are restart-safe: a failed smoke submission
can reuse the validated prepared run, and a partially submitted matrix retries
only missing labels. Manifest validation happens before GPU allocation, while
matrix selection and the full budget check share one ledger lock.

Every schema-v5 checkpoint and terminal state saves the exact initial field in
`fields/initial` and the complete applied and measured MF fields for every
iteration under `history/fields`. This restores the legacy analysis history
and additionally distinguishes the field used to construct the effective
Hamiltonian from the raw field measured after DMRG.
For field-only initialization from an older run, set `inherit_from` and
`inherit_sha256`; legacy top-level `alpha`, `beta`, `mu_cdw`, and `mu` are read,
but the old MPS is deliberately not loaded.
The stateless mirrors retain all of these fields and histories. They therefore
support both plotting and field-only `inherit_from`, while deliberately
rejecting MPS parent/resume use with a pointer to the full scratch artifact.

## Explicit continuations

Every segment stops before Slurm walltime and writes a CPU-portable state. There
is no automatic recursion. Continue only a branch that ended with `time_limit`
or `maximum_iterations`:

```bash
bash slurm/phase1_gpu.sh continue "$RUN_ID" frustrated__pairing_s1
```

The launcher hashes the source state, creates a new same-model resume config,
and records another 3-node-hour reservation. The default ceiling is four
segments per branch. Accepted branches cannot be continued automatically.

## Targeted unfrustrated-pairing recurrence control

The v3 candidate already contains two complete 20-update raw probes. Its final
probe has increasing period-two field and energy drift while the chi=200 DMRG
sweeps remain truncation-limited. Do not force it through the generic unsafe
continuation path. Prepare the dedicated chi=400 control instead:

```bash
bash slurm/phase1_gpu.sh plan-recurrence
RUN_ID=20260825_phase1_unfrustrated_pairing_recurrence_chi400
bash slurm/phase1_gpu.sh prepare-recurrence \
  20260824_phase1_gpu_v3_float64_history "$RUN_ID"
```

Preparation performs no Slurm submission and makes no ledger reservation. It
requires the source full scratch state to exist, checks its SHA-256 against the
stateless mirror, cross-checks the full status, fingerprints, Float64 scalar
type, phase iterations, and raw update modes, verifies that both stored orbit
members contain full MPSs, and produces three fingerprint-matched controls:

- the full v3 orbit member `001` as a hash-pinned parent;
- the full v3 orbit member `002` as a separate hash-pinned parent; and
- a second independent pairing seed, `pairing_s2`.

All three use `chi=400`, 16 DMRG sweeps, `cutoff=1e-11`,
`energy_tol=1e-9`, and a 20-update unmixed period-one/two probe. The segment
ends at the probe boundary with `cycle_action=stop`, so Anderson cannot average
or certify the candidate. Inspect the generated manifest and run the plan again
against the live ledger before the separately authorized staged submission:

```bash
sed -n '1,4p' "output/phase1_gpu/$RUN_ID/manifest.tsv"
bash slurm/phase1_gpu.sh submit "$RUN_ID"          # smoke only
bash slurm/phase1_gpu.sh status "$RUN_ID"
bash slurm/phase1_gpu.sh submit-matrix "$RUN_ID"   # only after smoke gates pass
```

The Stage A plan-only upper bound is `0.125 + 3*3 = 9.125` node-hours. Preparation
never substitutes the stateless mirror for a restartable parent: both phase
configs reference the same immutable full source SHA plus distinct
`parent_orbit_phase` values.

Do not launch same-geometry competitors at the same time. First audit all three
Stage A results. Conditional Stage B is unlocked only if (1) at least one of
the two phase-parent branches and (2) the independent `pairing_s2` branch are
accepted solutions with `max|alpha| >= 1e-4`. For an accepted orbit, every
stored phase must clear that floor separately. An accepted period-two survivor
must be an unmixed raw-map orbit; its members remain separate. The `1e-4`
threshold is a numerical screening floor, not a phase or long-range-order
claim. The preparation command independently rechecks acceptance, geometry,
Float64 storage, model/numerical/implementation fingerprints, the `E_p`
registry hash, the hash-linked full scratch artifact, and the current
implementation before writing an immutable gate record:

```bash
CONTROL_RUN=20260826_phase1_unfrustrated_competitors_chi400
bash slurm/phase1_gpu.sh prepare-recurrence-competitors \
  "$RUN_ID" "$CONTROL_RUN"
sed -n '1,4p' "output/phase1_gpu/$CONTROL_RUN/conditional_gate.tsv"
sed -n '1,3p' "output/phase1_gpu/$CONTROL_RUN/manifest.tsv"
```

This preparation also performs no submission or ledger reservation. It creates
two fresh-MPS, independent-seed controls, `sdw_s2` and `cdw_s2`, at the same
model point, `chi=400`, DMRG settings, convergence settings, Float64-CUDA
representation, and recurrence policy as Stage A. They receive an 80-iteration
execution ceiling: the first 20 updates are the same unmixed raw-map probe; an
unaccepted recurrence stops and is preserved, while a recurrence-free path may
then use Anderson only to accelerate a fixed point. `max_iterations` is run
provenance rather than part of the numerical fingerprint, so accepted Stage A
and Stage B results remain eligible for the canonical comparator if all other
fingerprints match.

Only after inspecting the gate record, stage Stage B through its own smoke:

```bash
bash slurm/phase1_gpu.sh submit "$CONTROL_RUN"
bash slurm/phase1_gpu.sh status "$CONTROL_RUN"
bash slurm/phase1_gpu.sh submit-matrix "$CONTROL_RUN"
```

Stage B's first-segment bound is `0.125 + 2*3 = 6.125` node-hours. Stage A plus
conditional Stage B is therefore `15.250` node-hours for first segments. With
the ledger snapshot of `114.500` after the accidental standard campaign, this
would project to `123.625` after Stage A or `129.750` after both, leaving
`276.375` or `270.250` node-hours under the 400-node-hour project cap. The live
Perlmutter ledger is authoritative. The combined four-segment emergency
ceiling is `60.250` node-hours, but continuations are not pre-authorized and
must be justified branch by branch; preserving the remaining budget for
higher-chi and scaling runs is part of the gate.

## Matched-seed convergence pilot

The next independent-start control is a three-branch chi=400 pilot at the same
unfrustrated representative point. It tests whether clean, norm-matched source
fields reduce seed-induced texture and slow drift. The locked branches are:

- pairing: finite-open-ladder mode `n=0`, phase `0`, `d_wave` form factor;
- SDW: mode `n=58`, phase `0`, odd leg parity; and
- CDW: mode `n=11`, phase `0`, even leg parity.

Every branch uses field norm `1e-3`, common product-state random seed `1404`,
Float64 CUDA, chi `400`, 16 sweeps, and the same 20-update raw period-one/two
probe. `max_iterations=21` and `cycle_action=stop` prevent entry into Anderson
acceleration. There is no inherited, parent, or resume checkpoint. The three
different carrier modes are targeted from the observed order-channel profiles;
this pilot is a convergence/basin test, not an unbiased wavevector survey or a
thermodynamic phase comparison.

After syncing the reviewed tree to Perlmutter, first run the read-only plan
against the live authoritative ledger, then prepare the campaign:

```bash
cd "$CFS/m4863/MPS-MFT/ladder_mps_mft"
bash slurm/phase1_gpu.sh plan-matched-seed-pilot

MATCHED_RUN=20260828_phase1_unfrustrated_matched_seed_chi400
bash slurm/phase1_gpu.sh prepare-matched-seed-pilot "$MATCHED_RUN"
sed -n '1,4p' "output/phase1_gpu/$MATCHED_RUN/manifest.tsv"
```

Preparation creates configs and scratch directories only; it does not submit
or reserve. The first-segment upper bound is `0.125 + 3*3 = 9.125` node-hours.
Only after checking the manifest and live ledger, stage the smoke and matrix:

```bash
bash slurm/phase1_gpu.sh submit "$MATCHED_RUN"
bash slurm/phase1_gpu.sh status "$MATCHED_RUN"
bash slurm/phase1_gpu.sh submit-matrix "$MATCHED_RUN"
```

The first `submit` reserves only the `0.125`-node-hour smoke. The matrix action
is unavailable until the smoke is `COMPLETED` and its Float64/runtime/linear-
algebra artifact passes validation; it then reserves three `3.0`-node-hour
segments atomically under the ledger lock. No continuation is pre-authorized.
Reserve later work for higher bond dimension and length convergence.

## Pair-binding interpolation and optional calculations

At the representative point, the registry contains

- `E_p(t0=1.0) = -0.1545120066237189`, and
- `E_p(t0=1.2) = -0.21453418655934797`.

Linear interpolation gives
`E_p(t0=1.1) = -0.18452309659153343` and
`t_perp^2/|E_p| = 0.05419375777188662`. The self-consistency maps contain this
coupling, with an additional factor of two in the pairing/exchange prefactor
and geometry-dependent integer factors. Thus changing `E_p` at fixed
Hamiltonian parameters changes the MF map in the same way as changing
`t_perp` to preserve `t_perp^2/|E_p|`. This does not make the isolated-ladder
Hamiltonians at different `t0` identical; it only justifies controlled
interpolation of the perturbative MF denominator.

Interpolation is exact-first, bracket-only, signed, and recorded in every
config/state. It refuses extrapolation or a sign-changing bracket. If a new
isolated-ladder calculation is scientifically warranted, submit it through the
same budget ledger:

```bash
bash slurm/phase1_gpu.sh submit-ep "$RUN_ID" vm02_t0115 64 8 -0.2 1.15 0.9375
```

That CPU job reserves 12 node-hours. Its result is not inserted into
`data/E_p_values.csv` automatically; inspect convergence and add it in a
separate reviewed change.

## Hard budget boundary

`output/project_budget/additional_node_hours.tsv` is an append-only reservation
ledger capped at 400 node-hours beyond the user-reported baseline of 277 used
from an approximately 1000-node-hour allocation. It counts requested upper
bounds and does not reclaim early finishes, so actual charge can only be lower.
CPU and GPU allocation pools are [separate NERSC
pools](https://docs.nersc.gov/jobs/policy/#charge-factors), but the project cap
deliberately sums their raw node-hour numbers.

The cap is enforceable only for submissions made through this launcher. Do not
submit the old GPU wrapper, call `sbatch` directly for this project, or run
`submit_E_p_ladder.sh` outside the guarded `submit-ep` action while this cap is
in force.

## QN statement

The dense GPU Phase 1 state carries no block-sparse `S_z`, particle-number, or
fermion-parity quantum number. Pairing terms already break full particle-number
conservation while preserving parity and `S_z` at the Hamiltonian level; those
symmetries remain properties of the operator even though the tensors do not
enforce their sectors. Separate CPU `E_p` and gap calculations continue to use
fixed QN sectors.
