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
each allocated scientific branch performs its own CUDA import and device
preflight before entering the SCF calculation.
CUDA.jl is configured to use its pinned artifact toolkit, so do not load the
Perlmutter `cudatoolkit` module for this workflow. The launcher unloads that
module, removes inherited NVIDIA-HPC-SDK runtime-library paths, and aborts if a
non-artifact cuBLAS, cuSOLVER, cuSPARSE, or other CUDA runtime library is loaded.
The GPU `Project.toml` explicitly exports `CUDA_Runtime_jll.local = "false"`
and `CUDA_Runtime_jll.version = "13.0"`. These settings prevent a Julia-module
or higher-load-path preference from switching the calculation back to a local
toolkit or requesting Perlmutter's newer system-toolkit version from an artifact
wrapper that does not provide it. CUDA 13.0 is the artifact runtime recorded by
the successful Float64 campaigns. The project also selects artifact `MPICH_jll`
with an empty preload list for HDF5's single-process MPI dependency; these jobs
do not use GPU-aware MPI, so they must not preload Cray `libmpi_gtl_cuda.so` and
its system `libcudart` dependency.
This follows [CUDA.jl's recommended artifact-toolkit
configuration](https://cuda.juliagpu.org/stable/installation/overview/). The
Perlmutter system toolkit remains appropriate for software compiled against
it, but must not be mixed into this artifact-based Julia process.

Before a scientific submission, the launcher checks Julia's effective merged
preferences without importing CUDA or requesting a compute node:

```bash
bash slurm/phase1_gpu.sh check-gpu-preferences
```

The check requires artifact mode, CUDA 13.0, artifact `MPICH_jll`, and an empty
MPI preload list. `submit`, `submit-matrix`, and GPU continuations run the same
check automatically before taking the budget lock or calling `sbatch`.

## Direct submission

Choose a new immutable run ID:

```bash
RUN_ID=20260823_phase1_gpu_v2
bash slurm/phase1_gpu.sh prepare-standard "$RUN_ID"
bash slurm/phase1_gpu.sh submit "$RUN_ID"
bash slurm/phase1_gpu.sh status "$RUN_ID"
```

`prepare-standard` creates all nine configs without submitting or reserving
anything. `submit` accepts only an existing prepared campaign and directly
submits every still-pending scientific branch; it cannot create a default
campaign implicitly. Literal placeholders such as `RUN_ID` are rejected.
`submit-matrix` is retained only as a backward-compatible alias for the same
direct action. There is no standalone smoke gate in launcher v1.13.1.

Every branch requires artifact-only CUDA runtime libraries and runs the
256-by-256 dense matrix-multiplication/eigendecomposition preflight through
cuBLAS/cuSOLVER before scientific work. A failed preflight aborts that branch;
it does not certify or contaminate an SCF result.

The matrix contains pairing, SDW, and CDW seeds for each of
`cubic_frustrated`, `cubic_unfrustrated`, and `square`. Each job requests one of
four GPUs for 12 hours in shared QOS, a conservative charge of 3 GPU node-hours.
The nine branches reserve 27.000 node-hours. NERSC shared jobs are charged
by the dominant node fraction; see the [NERSC queue and charge
policy](https://docs.nersc.gov/jobs/policy/#calculating-charges).

Launcher version 1.13.1 requests the general `gpu` constraint for branch
configs with `dmrg.maxdim < 1200`. Configs at `chi >= 1200` retain
`gpu&hbm80g`. This broadens the eligible GPU pool for the current chi-200 and
chi-400 work without weakening the explicit memory guard for the planned
large-bond-dimension campaign.

Launcher version 1.13.1 writes all MPS-bearing branch artifacts to Perlmutter
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
mixer-dependent recurrences are not inherited as physical solutions. The
following `submit` directly launches the scientific branches. Submission is
restart-safe: a partially submitted campaign retries only missing labels.
Manifest validation happens before GPU allocation, while branch selection and
the full budget check share one ledger lock.

Every schema-v5-and-newer checkpoint and terminal state saves the exact initial field in
`fields/initial` and the complete applied and measured MF fields for every
iteration under `history/fields`. Schema v7 also records the same seed
explicitly as `history/fields/seed` with `seed_iteration=0`, so complete-history
plots begin at the actual starting field. Density, energy, residual, and DMRG
histories still begin at completed update 1 because they are undefined for the
unevaluated seed. This restores the legacy analysis history
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
bash slurm/phase1_gpu.sh submit "$RUN_ID"
bash slurm/phase1_gpu.sh status "$RUN_ID"
```

The Stage A plan-only upper bound is `3*3 = 9.000` node-hours. Preparation
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

Only after inspecting the gate record, directly submit Stage B:

```bash
bash slurm/phase1_gpu.sh submit "$CONTROL_RUN"
bash slurm/phase1_gpu.sh status "$CONTROL_RUN"
```

Stage B's first-segment bound is `2*3 = 6.000` node-hours. Stage A plus
conditional Stage B is therefore `15.000` node-hours for first segments. With
the ledger snapshot of `114.500` after the accidental standard campaign, this
would project to `123.500` after Stage A or `129.500` after both, leaving
`276.500` or `270.500` node-hours under the 400-node-hour project cap. The live
Perlmutter ledger is authoritative. The combined four-segment emergency
ceiling is `60.000` node-hours, but continuations are not pre-authorized and
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
or reserve. The first-segment upper bound is `3*3 = 9.000` node-hours.
Only after checking the manifest and live ledger, submit directly:

```bash
bash slurm/phase1_gpu.sh submit "$MATCHED_RUN"
bash slurm/phase1_gpu.sh status "$MATCHED_RUN"
```

The direct action reserves three `3.0`-node-hour segments atomically under the
ledger lock. Each branch performs its own Float64/runtime/linear-algebra
preflight. No continuation is pre-authorized.
Reserve later work for higher bond dimension and length convergence.

## Exploratory square seed/basin pilot

The next square reconnaissance point is `L=64,U=8,V=-0.4,t0=1.4`,
`t_perp=0.1`, density `0.9375`, and chi `200`. It uses six independent
matched-amplitude starts: a pure d-wave control, a legacy-like random
relative-bond pairing control that is constant along the ladder, normal
combined SDW/CDW stripe controls at envelope modes `m=4,5`, and stripe+d-wave
starts at both modes. The legacy-like control follows the actual old fresh-run
structure: `beta=mu_cdw=0`; its `alpha` randomness is already spatially smooth.
At
`L=64`, the stripe harmonics are locked to spin/charge modes `(59,8)` and
`(58,10)`. The charge:spin source-norm ratio is `0.2`; mixed starts use
pairing:spin ratio `1`. The first mode is inspired by the supplied converged
profile and its neighbor is predeclared before inspecting new energies.

Pure pairing and normal-stripe starts are symmetry-subspace controls. The
stripe+d-wave branches are the unrestricted basin test: a nonzero anomalous
source prevents an exactly number-conserving `alpha=0` initialization from
excluding coexistence by construction, while pairing remains free to decay.

The deliberately loose exploratory controls are 12 sweeps, cutoff `1e-10`,
DMRG energy tolerance `1e-6`, inner and outer density tolerances `1e-3`,
chemical-potential bracket step `0.01`, bracket growth `3`, and at most 80 MF
updates. The first 20 are an unmixed raw-map period-one/two probe. Anderson may
follow, while any raw orbit remains separately archived and physically
classified. No branch inherits or resumes fields or an MPS. Energies are at
most a preliminary canonical ranking among accepted branches sharing this one
fingerprint and square geometry.

```bash
cd "$CFS/m4863/MPS-MFT/ladder_mps_mft"
bash slurm/phase1_gpu.sh plan-square-seed-pilot

SQUARE_RUN=20260830_phase1_square_t014_vm04_seed_chi200_loose
bash slurm/phase1_gpu.sh prepare-square-seed-pilot "$SQUARE_RUN"
sed -n '1,7p' "output/phase1_gpu/$SQUARE_RUN/manifest.tsv"
bash slurm/phase1_gpu.sh budget
```

Preparation does not reserve or submit. The first-segment envelope is
`6*3 = 18.000` node-hours. With the synced `135.750` ledger snapshot,
submission would project to `153.750` reserved and `246.250` unreserved; the
live Perlmutter ledger is authoritative. Submit only after inspection:

```bash
bash slurm/phase1_gpu.sh submit "$SQUARE_RUN"
bash slurm/phase1_gpu.sh status "$SQUARE_RUN"
```

The full `t0={1.0,1.2,1.4}` by `V={0,-0.2,-0.4}` square grid remains
conditional and plan-only. This six-branch representative bank plus
provisional three-branch banks at the other eight points would reserve
`90.000` node-hours total, leaving `174.250` from the current synced ledger.
Repeating all six branches everywhere would leave only `102.250` and is not
recommended. Gate each point so that chi and length convergence retain
priority. The evidence and accuracy ladder are recorded in
`docs/SQUARE_SEED_AND_GRID_PLAN_2026-08-30.md`.

## Six-parent tight-five square probe

The loose square pilot establishes qualitative basin collapse but does not
resolve its `2.490e-5`-per-site energy spread. Continue all six accepted states
under one tighter fingerprint for at most five new raw-map evaluations. This is
a short residual/noise diagnostic, not a grid point or a replacement for later
bond-dimension and length convergence.

Preparation must run on Perlmutter because it verifies and rehashes every full
scratch parent. It does not submit or reserve:

```bash
bash slurm/phase1_gpu.sh plan-square-tight5

SOURCE_RUN=20260830_phase1_square_t014_vm04_seed_chi200_loose
TIGHT_RUN=20260831_phase1_square_t014_vm04_chi200_tight5
bash slurm/phase1_gpu.sh prepare-square-tight5 "$SOURCE_RUN" "$TIGHT_RUN"
sed -n '1,7p' "output/phase1_gpu/$TIGHT_RUN/manifest.tsv"
bash slurm/phase1_gpu.sh budget
```

The prepared run pins all six accepted period-one full states and their
SHA-256 values, starts fresh histories, retains `chi=200`, and uses 16 sweeps,
cutoff `1e-11`, DMRG energy tolerance `1e-9`, inner/outer density tolerance
`1e-4`, field gates `1e-7` absolute or `1e-4` relative, and energy-change gate
`1e-7` per site. The raw physical map has threshold zero. The manifest declares
a post-processing floor scan of `0,1e-6,1e-5,1e-4`; see
`docs/PHASE1_NUMERICAL_ERROR_BUDGET.md`.

Each branch requests one of four GPUs for three hours, or `0.75` node-hours.
The complete first-segment envelope is therefore `6*0.75 = 4.500`
node-hours. Recheck the live Perlmutter ledger before direct submission:

```bash
bash slurm/phase1_gpu.sh submit "$TIGHT_RUN"
bash slurm/phase1_gpu.sh status "$TIGHT_RUN"
```

Five records are insufficient for complete period-two validation. Any apparent
two-cycle remains unresolved and its raw phases must remain separate. Anderson
is disabled throughout this probe.

## Square V=0, t0=1.4 six-seed pilot

The next staged reconnaissance reuses the six independent seed classes at
square `L=64,U=8,V=0,t0=1.4,t_perp=0.1`, density `0.9375`, and `chi=200`.
It uses the exact registry value `E_p=-0.14653773091916378`. The first 20
updates remain the raw map, but period two now requires sign-reversing steps
and fixed-point acceptance includes the slow-mode-extrapolated residual.

```bash
cd "$CFS/m4863/MPS-MFT/ladder_mps_mft"
bash slurm/phase1_gpu.sh plan-square-v0-seed-pilot

SQUARE_V0_RUN=20260901_phase1_square_t014_v000_seed_chi200_loose
bash slurm/phase1_gpu.sh prepare-square-v0-seed-pilot "$SQUARE_V0_RUN"
sed -n '1,7p' "output/phase1_gpu/$SQUARE_V0_RUN/manifest.tsv"
bash slurm/phase1_gpu.sh budget
```

Preparation does not submit or reserve. Launcher v1.13.1 directly reserves
`6*3 = 18.000` node-hours for the scientific branches. The already-prepared
v1.12.0 campaign is explicitly compatible: `submit` ignores an existing smoke
row and submits only pending branches. If that smoke is cancelled, wait until
`sacct` shows a terminal state and run `reconcile "$SQUARE_V0_RUN"` so its
unused ceiling is released in project-control accounting. Recheck the live
ledger before submission. The complete scientific and seed contract is in
`docs/SQUARE_V0_T014_SEED_PLAN_2026-09-01.md`.

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
from an approximately 1000-node-hour allocation. Every submission keeps its
original requested upper bound in this file. A separate append-only ledger,
`output/project_budget/additional_node_hours_reconciliations.tsv`, records the
final `sacct` allocation row for terminal jobs and releases only the unused
part of the original ceiling. No reservation row is rewritten or deleted.

On Perlmutter, reconcile one campaign or all recorded campaigns with:

```bash
bash slurm/phase1_gpu.sh reconcile "$RUN_ID"
# or, after inspecting the live ledger:
bash slurm/phase1_gpu.sh reconcile
bash slurm/phase1_gpu.sh budget
```

The action is idempotent by Slurm job ID. It ignores pending/running jobs and
requires a terminal state, integer `ElapsedRaw`, and finalized `End` time. The
measured project amount is `ElapsedRaw/3600` times the effective node fraction
already implied by the reservation (`0.25` for the present one-of-four-GPU
jobs and 64-CPU jobs), capped at the reserved ceiling. The active hard-cap
total is original reservations minus recorded releases. NERSC `sacct` and
allocation reports remain authoritative; the reconciliation ledger is the
project's reproducible accounting mirror.

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
