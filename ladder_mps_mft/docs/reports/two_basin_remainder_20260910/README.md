# Remaining square two-basin starts: 40 evaluations with channel noise handling

The user authorized proceeding with the fourteen remaining 95%/5% starts at
chi=200, reducing the maximum to 40 and correcting the acceptance issues seen
at `(t0,V)=(1.4,-0.4)`. The four existing anchors keep their original controls.
This directory records local preparation and saved-history replay; it does
not record a new submission or relabel any existing result.

## Revised controls

The new base is
[`phase1_gpu_square_two_basin_chi200_raw40.toml`](../../../configs/phase1_gpu_square_two_basin_chi200_raw40.toml).
The earlier 80-evaluation base remains available for the anchor record.

| Control | Anchors | Remaining fourteen |
|---|---:|---:|
| Maximum MF evaluations | 80 | 40 |
| Minimum before acceptance, including orbits | 50 | 30 |
| Stable fixed-point window | 5 | 10 |
| Channel noise floor | Disabled | 5e-7 t |
| Global field absolute / relative tolerance | 1e-7 t / 1e-4 | Unchanged |
| Corrected canonical energy span per site | 1e-8 t | 2e-8 t |
| Inner DMRG stopping and acceptance tolerance, total energy | 1e-8 t | Unchanged |
| Inner/outer density tolerance per site | 1e-5 | Unchanged |
| Hamiltonian identity / eigenvalue consistency per site | 1e-8 t / 1e-8 t | Unchanged |
| Solver deadline / Slurm ceiling per one-GPU segment | 11.5 h / 12 h | Unchanged |

All updates remain raw, with no Anderson or damping. Every MF evaluation
retains its energy history. Reaching 40 without passing the gates remains
`maximum_iterations`; no automatic continuation is scheduled.

`channel_noise_floor` is an opt-in control, defaulting to zero. With it enabled,
the two-residual extrapolation is used only when **both residual maxima exceed
the noise floor**. A nonzero uniform charge background therefore cannot turn
a 1e-9-scale scalar fluctuation into an infinite convergence penalty. Direct
residuals below 5e-7 may pass; above the floor, the existing stricter extrapolated
absolute/relative criteria apply. Resolved coherent growth still fails.

To keep many small steps from hiding a slow instability, each channel also
has a full-window profile-span gate. Across every applied and measured profile
in the last ten evaluations, compute the componentwise maximum minus minimum.
Its maximum must be at most 5e-7 t, or its L2 norm divided by the largest profile
L2 norm must be at most 1e-4. This catches accumulated drift and intermediate
excursions, including weak spin motion hidden behind larger pairing fields.
The span values and pass flags are saved under
`history/channels/CHANNEL/window_{absolute,relative,passes}`.

Period-two acceptance retains its original recurrence, oscillation, phase
energy, closure, and inner-DMRG checks; the fixed-point profile-span check is
not applied across distinct orbit phases. The new floor is not a relaxation
of period-two recurrence tolerances. The minimum of 30 applies to both kinds
of solution.

## Qualification on saved histories

[`replay_two_basin_convergence.jl`](../../../scripts/replay_two_basin_convergence.jl)
calls the actual Julia convergence functions on immutable saved fields,
energies, densities, and inner-sweep histories. All eight source hashes are
checked before and after the read. See [sources](replay_sources.csv),
[per-evaluation checks](history_gate_replay.csv), and the
[configuration/implementation hashes](replay_contract.toml).

| Saved trajectory | First pass of all available revised history gates |
|---|---:|
| V=-0.4, 95% stripe seed, job 58093799 | 35 |
| V=-0.4, 95% pairing seed, job 58093800 | 30 |
| Three V=0 runs ending at six raw evaluations | None; observation window and other gates fail |
| Three V=0 runs with 21 raw evaluations before mixing | None; spin and other channel gates fail independently of the minimum |

The first two entries demonstrate that these particular paired plateaus fit
within the proposed 40-evaluation budget. They are **not retroactive acceptance**:
per-iteration Hamiltonian-identity and eigenvalue-consistency errors are not
stored in these histories, so those checks cannot be replayed at iterations
30/35. Both original terminal consistency checks pass at iteration 80. The
old V=0 comparisons use only their raw prefixes, never Anderson-suppressed
endpoints. Original acceptance flags and source files remain unchanged.

The focused Julia checks passed 75 assertions: 52 existing raw-basin tests
and 23 noise-handling tests, including a sub-floor-step drift that accumulates
above the window tolerance, noise on a nonzero charge background, growing spin,
energy excursions, failed inner DMRG, minimum observation, period two, and
storage. A local mock-launcher test passed; it never invokes Slurm. No DMRG
or GPU calculation was run locally. This supports the specific threshold
change; a slower instability can still outlive a 40-step observation.

## Scope and cost

The existing `remainder` scope prepares both families at:

| t0 | V |
|---|---|
| 1.0 | -0.4, -0.2, 0.0 |
| 1.2 | -0.4, -0.2, 0.0 |
| 1.4 | -0.2 |

It excludes the four already-submitted starts at `(1.4,-0.4)` and `(1.4,0.0)`.
The user's latest message calls the pending point `(1.0,0.0)`; the supplied
job labels identify 58093802/58093803 as `(1.4,0.0)`. Clarification was requested.
A separately submitted `(1.0,0.0)` job would overlap this prepared scope.

Local preview: `output/seed_previews/20260910_square_two_basin40/final/remainder/`.
Preparation produced exactly 14 configs and derived seeds, verified reference
and seed hashes/readback, nonzero competing fields, target E_p mapping, and
matching fingerprints within each coordinate. Seed-contract iteration values
now come from the actual base settings instead of hard-coded 80/50 values.

The maximum is 560 MF evaluations across the fourteen starts. At the unchanged
12-hour one-GPU ceiling, the upper reservation is **42 fractional node-hours**;
40 iterations is a work limit, not a guaranteed runtime. The launcher reconciles
the existing ledger and uses the same live 400-node-hour hard cap. The completed
anchor pair's actual cost remains 3.763125 node-hours. No new local reservation
or accounting entry has been written.

## Perlmutter handoff — user-run only, after this revision is published

The pending jobs load solver files from the original checkout when they start.
Leave that checkout at its submitted source revision. Fetching changes and
creating a separate worktree preserves those jobs while allowing the new
implementation to run. The wrapper reads the original anchor `run.env` to
retain the exact account, scratch, result, reservation, and reconciliation
locations, then selects the new checkout and 40-evaluation base.

```bash
cd "$CFS/m4863/MPS-MFT/ladder_mps_mft"
git -C .. fetch origin
git -C .. worktree add --detach "$CFS/m4863/MPS-MFT-two-basin40" origin/codex/mps-mft-phase0-refactor
cd "$CFS/m4863/MPS-MFT-two-basin40/ladder_mps_mft"
bash slurm/submit_square_two_basin_remainder.sh
```

Default run ID: `20260910_square_two_basin_95_5_40_remainder`. The wrapper
prepares only the fourteen listed starts, reconciles, then submits. An existing
run ID is refused. To recover a submission interrupted after preparation, use
the existing `phase1_gpu.sh submit RUN_ID` command, which skips recorded jobs.
No old jobs are canceled, modified, or resubmitted by the wrapper.

Progress remains available from the original checkout because results and
ledgers are shared:

```bash
cd "$CFS/m4863/MPS-MFT/ladder_mps_mft"
bash slurm/phase1_gpu.sh status 20260910_square_two_basin_95_5_40_remainder
```

Local reproduction (Windows/PowerShell):

```powershell
julia --startup-file=no --compiled-modules=existing --project=ladder_mps_mft ladder_mps_mft/test/test_raw_basin.jl
julia --startup-file=no --compiled-modules=existing --project=ladder_mps_mft ladder_mps_mft/scripts/replay_two_basin_convergence.jl
python -B -m unittest discover -s ladder_mps_mft/test -p test_two_basin_remainder_launcher.py -v
```
