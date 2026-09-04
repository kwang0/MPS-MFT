# Ladder MPS+MF refactor

This is an isolated replacement path for the ladder mean-field workflow. It lives entirely under `ladder_mps_mft/`; the legacy scripts and result files outside this directory remain untouched. The scientific solver is device-independent, with a pinned CPU environment and a CUDA overlay for production GPU DMRG.

The implementation currently provides:

- a pinned Julia project and modular ITensorMPS solver;
- exact lookup of pair-binding data from the copied `data/E_p_values.csv` registry, plus opt-in bracketed linear interpolation in `t0` with saved endpoints and no extrapolation;
- zero-temperature variational phase comparison with the mean-field double-counting constants and a direct bare-Hamiltonian cross-check;
- a mixer-independent raw-map probe that accepts physical period-two mean-field solutions, followed only when needed by adaptive linear or Anderson fixed-point acceleration;
- immutable final HDF5 states, hash-checked parent/restart lineage, model/numerics/implementation fingerprints, and accepted-only selection;
- charge and spin structure factors, `K_rho`, rung-cut entanglement/central-charge fits, sign-resolved pair correlations, and separate fixed-sector spin/charge/pair-gap calculations;
- a restartable six-sector isolated-ladder backbone plus a zero-field,
  data-driven Stage 1 covariance screen for charge, spin, and Hermitian pairing
  candidates;
- a gated Stage 2 projected-response pilot that orthonormalizes the motivated
  and covariance-selected bank, reuses each finite-field ladder solve across
  all three geometries, preserves fermion parity for pairing probes, and
  validates only the leading measured eigenvectors at a second amplitude;
- a dense-CUDA production backend that preserves the refactored SCF, recurrence, variational, checkpoint, and diagnostic logic while explicitly disabling QN block sparsity; and
- a guarded Perlmutter Phase 1 launcher with direct scientific-branch submission, per-branch GPU preflight, explicit continuations, optional ledgered CPU `E_p` jobs, scratch-resident full MPS artifacts with automatic stateless CFS mirrors, and a conservative 400-additional-node-hour cap.

The first complete Perlmutter CPU matrix (`20260821_phase0_cpu_v2`) is retained as
backend-equivalence and screening evidence. It shortlisted `serial-t1` and
`blocksparse-t4`, but its `chi=64` fixed-`mu=0` workload ran at `n=0.5614`
rather than `n=0.9375`, so it is backend screening only. The available CPU
timings and legacy-GPU timing evidence indicated roughly a two-order-of-magnitude
production disadvantage. Phase 0 is therefore closed without promoting a CPU
production backend; Phase 1 uses the refactored solver on CUDA.

## Project continuity

For a new conversation, device, or collaborator, begin with
`docs/README.md`, then read the short current snapshot in
`docs/PROJECT_STATE.md`, the workflow map in `docs/ARCHITECTURE.md`, and the
current completion sequence in `docs/plans/ACTIVE.md`. These files point to the
relevant stable method and campaign documents without requiring the full
append-only run log in the initial context.

The dated August device handoff remains a historical snapshot. Live user-
reported Perlmutter status and accounting supersede any dated local copy.

## Quick start

All Perlmutter commands in this repository are operator handoff commands for
the user to run. Codex works and validates locally; it does not authenticate to
NERSC, synchronize files, inspect the live scheduler, or submit/cancel jobs.

Instantiate and test from this directory:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

User-run Perlmutter handoff: instantiate the pinned CUDA overlay:

```bash
JULIA_PKG_PRECOMPILE_AUTO=0 julia --project=gpu \
  -e 'using Pkg; Pkg.instantiate(; allow_autoprecomp=false)'
```

User-run Perlmutter handoff: inspect the Phase 1 GPU plan without submitting:

```bash
bash slurm/phase1_gpu.sh plan
```

User-run Perlmutter handoff: inspect the guarded bare-ladder CPU pilot
(`V=0`, `t0=1.4`) without submitting:

```bash
bash slurm/bare_stage1_cpu.sh plan
```

Its six fixed-sector backbone, immutable per-`chi` MPS checkpoints, Stage 1
candidate spectra, and the deliberate stop before finite-field Stage 2 are
documented in `docs/BARE_STAGE1_CPU.md`.

After the Stage 1 results have been synchronized and reviewed, inspect the
separate Stage 2 discovery plan (still without submitting):

```bash
bash slurm/bare_stage2_cpu.sh plan 20260901_bare_t014_v0_stage1
```

The Stage 1 physics/efficiency report is under
`docs/reports/bare_stage1_t014_v0_20260902/`; the Stage 2 scientific gates,
job graph, resource bounds, and user-run handoff are in
`docs/BARE_STAGE2_CPU.md`.

The targeted chi=400 matched-seed convergence pilot has its own read-only plan
and preparation-only action:

```bash
bash slurm/phase1_gpu.sh plan-matched-seed-pilot
bash slurm/phase1_gpu.sh prepare-matched-seed-pilot \
  20260828_phase1_unfrustrated_matched_seed_chi400
```

Preparation does not submit a job or reserve node-hours. See
`docs/PERLMUTTER_PHASE1_GPU.md` before directly submitting the prepared
scientific branches.

Run a configured SCF branch:

```bash
julia --project=. scripts/run_scf.jl configs/example_scf.toml
```

Generate independent SC, SDW, and CDW configurations from a common base:

```bash
julia --project=. scripts/prepare_branch_scan.jl configs/example_scf.toml output/my_branch_scan
```

The default above preserves legacy seed behavior. For an opt-in smooth,
equal-norm, common-product-state protocol and lightweight seed inspection, see
`docs/SEEDING.md`.

Accepted fixed points and validated periodic solutions with identical model, numerical, implementation, and E_p-registry fingerprints can be ranked:

```bash
julia --project=. scripts/compare_branches.jl /path/to/sc/state.h5 /path/to/sdw/state.h5 /path/to/cdw/state.h5
```

See `docs/PHASES_0_TO_4.md` for the staged plan, `docs/LITERATURE_AND_PUBLICATION_GATES.md` for numerical context, and `docs/RUN_LOG.md` for the append-only synchronization record.
See `docs/PERLMUTTER_STORAGE.md` for the full-state scratch layout, compact
analysis mirrors, and the one-time migration procedure for older campaigns.
The v2 data-quality decision and conditional timing observations are recorded
in `docs/PHASE0_V2_AUDIT.md`. The accepted-state audit of the Float64-history
campaign is in `docs/PHASE1_V3_AUDIT.md`. For cross-device continuation, begin
with `docs/README.md` and `docs/PROJECT_STATE.md`, then paste
`docs/NEW_DEVICE_CHAT_PROMPT.md` into the new Codex task. The older
`docs/DEVICE_HANDOFF_2026-08-25.md` is retained as historical context.

## Important interpretation rules

- `completed=true` means either an accepted period-one fixed point or an accepted unmixed periodic solution. Mixer-dependent recurrences remain incomplete candidates.
- A physical period-two solution is stored phase by phase and is never replaced by its field average. This follows the CDW construction of Bollmark, Kohler, and Kantian, Phys. Rev. B 111, 125141 (2025).
- A timing payload is not a converged scientific state.
- `canonical_variational_energy` is a zero-temperature energy, not a finite-temperature free energy.
- Cross-geometry energies describe different Hamiltonians and are not ranked as competing phases by the comparison tool.
- Production GPU states conserve neither `S_z` nor fermion parity at the tensor-block level. The Hamiltonian still has the corresponding symmetries; disabling QNs changes representation/performance, not its terms. Fixed-sector gap and `E_p` calculations remain separate QN-conserving CPU runs.
- The pinned CUDA.jl artifact toolkit is the sole CUDA runtime for Phase 1. The guarded launcher unloads Perlmutter's `cudatoolkit` module and every scientific branch aborts during its preflight if system CUDA runtime libraries are loaded.
- Fixed bond dimension, enhanced pair fields, long correlation lengths, or favorable finite-size pairing do not by themselves establish superconducting long-range order.
