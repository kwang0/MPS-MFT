# Perlmutter Phase 1 GPU workflow

This workflow runs the refactored solver on one dense CUDA GPU. It does **not**
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
bash slurm/phase1_gpu.sh submit "$RUN_ID"
bash slurm/phase1_gpu.sh status "$RUN_ID"
```

`submit` prepares all nine configs and submits only a 30-minute smoke test. It
is not a scientific calculation. The smoke now requires artifact-only CUDA
runtime libraries, exercises a 256-by-256 dense GPU matrix multiplication and
Hermitian eigendecomposition through cuBLAS/cuSOLVER, runs the tiny DMRG, and
round-trips its MPS through HDF5. After `status` reports the smoke job as
`COMPLETED` and `gpu_smoke.h5` exists, inspect the smoke log for the recorded
preflight and submit the branch matrix:

```bash
grep -E 'gpu_smoke_path|linalg_preflight_dimension|CUDA runtime library.*system path' \
  "output/phase1_gpu/$RUN_ID"/logs/smoke-*.out
bash slurm/phase1_gpu.sh submit-matrix "$RUN_ID"
bash slurm/phase1_gpu.sh status "$RUN_ID"
```

The corrected smoke must print `linalg_preflight_dimension=256` and must not
print a `system path` CUDA-runtime warning.

The matrix contains pairing, SDW, and CDW seeds for each of
`cubic_frustrated`, `cubic_unfrustrated`, and `square`. Each job requests one of
four GPUs for 12 hours in shared QOS, a conservative charge of 3 GPU node-hours.
The smoke plus matrix reserves 27.125 node-hours. NERSC shared jobs are charged
by the dominant node fraction; see the [NERSC queue and charge
policy](https://docs.nersc.gov/jobs/policy/#calculating-charges).

Run `20260822_phase1_gpu_v1` is retained only for audit: its smoke passed despite
CUDA.jl warnings, but all nine branches then mixed CUDA.jl artifact libraries
with CUDA 13.2 libraries from the NVIDIA HPC SDK and segfaulted in the first
DMRG eigendecomposition. It produced no result or checkpoint files. Script
version 1.0.1 can display its status but refuses new submissions or
continuations from that run; use a new run ID.

Preparation and matrix submission are restart-safe: a failed smoke submission
can reuse the validated prepared run, and a partially submitted matrix retries
only missing labels. Manifest validation happens before GPU allocation, while
matrix selection and the full budget check share one ledger lock.

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
