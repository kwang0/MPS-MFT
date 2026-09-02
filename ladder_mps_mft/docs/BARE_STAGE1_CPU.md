# Bare-ladder backbone and Stage 1 CPU pilot

This workflow implements the isolated-ladder backbone in
`docs/claude/BARE_SURVEY_AND_LADDER_BACKBONE.md`, followed by Stage 1 of the
hybrid response search. The first production point is

```text
L=64, U=8, t=1, n=0.9375, V=0, t0=1.4, tp=0.1.
```

Stage 2 is intentionally not submitted by this launcher. The Stage 1 spectra
must be inspected first so the covariance candidates and the physically
motivated bank can be pruned before any finite-field DMRG solves are spent.

Codex never authenticates to NERSC/Perlmutter, synchronizes this workflow, or
operates Slurm. The commands below are a handoff for the user, who performs all
transfers, planning against live accounting, submission, status checks, and
result synchronization.

## Scientific payload

The backbone runs six independent, number- and `S_z`-conserving CPU DMRG
sectors:

```text
(N,0), (N,2), (N-1,1), (N-2,0), (N+1,1), (N+2,0).
```

For `L=64` and `n=0.9375`, `N=120`. Each sector uses a spatially distributed
fixed-sector product state, a 15-sweep pre-relaxation at `chi<=200`, and warm
starts through `chi=400,800,1200`. Noise remains at least `1e-8`. An absolute
per-sweep energy stop is enabled only after the final bond dimension has been
held for the configured minimum number of sweeps.

Every completed bond-dimension stage writes a separate immutable MPS
checkpoint. A restarted sector automatically loads its latest checkpoint.
The final sector artifact records every sweep energy, maximum discarded
weight, realized maximum link dimension, last-five-sweep energy spread, and
both numerical and scientific convergence flags. Assembly rejects mixed
configuration or implementation hashes and records the spin gap, charge gap,
hole/particle pair binding, and chemical potential at every bond dimension.
Stage 1 will not read a backbone unless all six final `chi=1200` states pass
their convergence gates.

The zero-field `(N,0)` MPS then supplies:

- density and spin profiles and complete connected real-space covariance
  matrices, resolved into even and odd leg parity;
- charge and spin structure factors, both `K_rho` normalizations,
  entanglement profiles, and the central-charge fit;
- onsite, rung, and both leg singlet-pair matrices. Pair addition and removal
  Gram matrices are computed separately and summed for the Hermitian source
  `Delta+Delta^dagger` used in a finite-field response;
- the exact zero-field raw mean-field map `F(0)` and its norm;
- charge and rung-pair decay fits under three bulk-window choices; and
- the validity ratios `tp/|E_p|`, `tp/Delta_s`, and `tp/Delta_c`.

The pairing transfer calculation caches MPS environments. Its scaling is
quadratic in the number of bond coordinates, rather than performing a fresh
length-`L` MPO contraction for every matrix entry. Pairing classes are
diagonalized separately in Stage 1; their leading vectors are mixed and
orthonormalized together with the physical probe bank in Stage 2.

## User-run Perlmutter handoff

After the user synchronizes the repository to Perlmutter, the launcher is
read-only by default:

```bash
bash slurm/bare_stage1_cpu.sh plan
```

When the user decides to submit, submission is explicit and first invokes the
repository-required Phase 0 CPU calibration plan:

```bash
bash slurm/bare_stage1_cpu.sh submit 20260901_bare_t014_v0_stage1
```

The six sectors are a Slurm array so they can run concurrently. The selected
DMRG topology is the existing `blocksparse-t4` CPU winner: four Julia
block-sparse threads, one BLAS thread, and one Strided thread. Perlmutter
requests two Slurm logical CPUs per Julia thread. Assembly and Stage 1 are
dependency jobs; no finite-field Stage 2 job is present.

The user can inspect scheduler state or the completed eigenvalue table with:

```bash
bash slurm/bare_stage1_cpu.sh status 20260901_bare_t014_v0_stage1
bash slurm/bare_stage1_cpu.sh show 20260901_bare_t014_v0_stage1
```

Full restartable MPS artifacts live under `$PSCRATCH`. A stateless copy with
all MPS objects removed, plus source paths and SHA-256 hashes, is mirrored to
the CFS control directory after Stage 1.

## Local validation boundary

Windows is suitable for compilation, unit tests, and the tiny end-to-end
fixture:

```powershell
julia --startup-file=no --threads=4 --project=. scripts/run_ladder_backbone.jl `
  test/fixtures/phase0_tiny.toml output/bare_stage1_smoke all
julia --startup-file=no --threads=4 --project=. scripts/run_bare_stage1.jl `
  test/fixtures/phase0_tiny.toml output/bare_stage1_smoke/backbone.h5 `
  output/bare_stage1_smoke/stage1.h5
```

The tiny fixture checks software behavior only. It is not physical evidence
for the `L=64` ladder. The Perlmutter artifacts synchronized back by the user
remain authoritative for the pilot spectra.
