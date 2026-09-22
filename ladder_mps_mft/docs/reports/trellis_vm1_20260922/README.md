# Four trellis starts at V=-1.0

Prepared locally September 22, 2026 at the user's request to repeat the
[original four trellis runs](../trellis_comparison_20260916/README.md) at
V=-1.0 to better match the intended material parameters. This is a new
independent campaign; there are no submitted job IDs or new scientific
results yet.

| Spatial cell | Stripe-dominant start | Pairing-dominant start |
|---|---|---|
| Skew one-ladder | 95% stripe + 5% pairing | 95% pairing + 5% stripe |
| Rectangular two-ladder A/B | Same template on A and B | Same template on A and B |

The base config changes only V and the unprepared output placeholder.
Retain t=1, U=8, t0=1, tau0=tau1=0.1, L=64 per ladder, n=15/16 on each
ladder, chi=200, r_range=4, and product-state random seed 1404. Each start
uses fresh MPSs with the same established reference correlations (stripe
at (1,0), pairing at (1.4,-0.4)); this is not a continuation of the V=0
endpoints. A/B templates retain the original trellis registration.

The exact registry row at (L,U,V,t0,n)=(64,8,-1,1,0.9375), bare chi=1000,
has **E_p=-0.2713195876256691 t**, so Delta=0.2713195876256691 t. All initial
fields are rebuilt with this target denominator and the appropriate trellis
kernel. No interpolation or new pair-binding calculation is required.
Reference SHA-256:
`e01a1ea7d6be813110870d26377db0df529d1584816c946af78584e1e1fbddc1`.
The [four-branch local receipt](prepared_branches.csv) records configs,
seeds and fingerprints; host-specific paths/hashes are regenerated during
Perlmutter preparation. Local preview configs are not submission configs.

## Numerical controls and measurements

- At most 60 raw simultaneous cell sweeps, at least 40 before acceptance,
  ten stable records, damping=1 and no Anderson. Four jobs contain six
  spatial MPSs, for at most 360 density-targeted ladder solves.
- Field absolute/relative tolerances 1e-7/1e-4, channel floor 5e-7,
  energy-window tolerance 1e-7 t/site, inner-DMRG tolerance 1e-7 t total,
  and density tolerance 1e-5 on each ladder.
- Only stationary cell solutions are accepted. A/B are spatial ladders;
  temporal alternation remains diagnostic. Keep unaccepted endpoints and
  their histories with their original status.
- Full terminal correlations are enabled for every spatial MPS at accepted
  and maximum-iteration endpoints, using the current measurement workflow.
  This includes raw/connected pair-pair, charge/spin, single-particle and
  entanglement diagnostics. Deadline stops retain the existing offline
  measurement option.

Each job requests one GPU, 32 logical CPU cores, shared QOS, one segment,
and a **16-hour allocation ceiling**, with the unchanged **11.5-hour solver
deadline**. As in the recent square A/B campaign, the remaining 4.5 hours
allow for terminal CPU measurements inside the GPU allocation (the synced
square A/B example took about 3.31 hours for both passes). Four jobs reserve
at most **16 node-hours** under the existing 400-additional-node-hour cap.
This is a ceiling, not measured runtime at V=-1. There is no automatic
continuation; a deadline can end the solver before 60 sweeps.

## User-run Perlmutter handoff

Run on **Perlmutter**, from the user-managed checkout:

```bash
cd "$CFS/m4863/MPS-MFT/ladder_mps_mft"
git pull --ff-only
module load julia
bash slurm/submit_trellis_vm1_comparison.sh
```

The wrapper reuses the original anchor account, shared scratch/result roots
and append-only budget ledgers. It prepares exactly four branches under
`20260922_trellis_vm1_two_basin_comparison_60`, reconciles completed
reservations and submits through the existing budget guards. Existing run
IDs are rejected; an optional argument selects another new ID. Keep the
checkout fixed while jobs use it. No old V=0 configs, results or acceptance
flags are modified.

For subsequent user-run status inspection:

```bash
cd "$CFS/m4863/MPS-MFT/ladder_mps_mft"
bash slurm/phase1_gpu.sh status 20260922_trellis_vm1_two_basin_comparison_60
```

## Local validation

All **240 focused trellis assertions and six local launcher tests passed**.
Preparation checks verify both V=0 and V=-1, exact registry
selection, fresh MPS lineage, seed hashes, reconstructed target fields,
unchanged numerical controls and distinct model fingerprints. Local
fake-launcher checks verify config/run selection, allocation ceiling,
shared accounting, custom run IDs and stopping on preparation failure.
No production DMRG, GPU timing, remote connection, transfer or scheduler
action is part of this preparation. Solver and measurement code are unchanged.
