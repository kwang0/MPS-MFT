# Square A/B test at the paired grid points

Prepared September 20, 2026: four fresh runs, two seed families at each of
(t0,V)=(1.4,-0.4) and (1.4,-0.2). This is prepared work, not new Perlmutter
evidence. No job was submitted locally.

## Physical cell and energy

Use `geometry="square", spatial_cell="two_ladder"`. Both ladders have
origin x=i. Every transverse square bond connects opposite legs at the same
rung, without trellis diagonal hopping, half-rung shear, shifted ends or an
extra coordination factor. If K is the original square field map, the new
map is `fields_A=K(correlations_B), fields_B=K(correlations_A)`. The two
physical neighbors of A are copies of B, each connected to its facing leg.

One Jacobi sweep solves both MPSs against frozen incoming fields, then
rebuilds both outgoing fields. This is two two-leg MPSs coupled by mean field,
not an entangled four-leg MPS. Mean density is constrained to 15/16 separately
on each ladder. Spatial charge, spin, pairing and exchange profiles are free;
net transfer between ladders with different average fillings is not tested.

A=B reproduces the original finite-OBC square fields exactly. The cell energy
divided by 4L reproduces the one-ladder energy divided by 2L. This is a nested
spatial extension of the same square model, rather than the rectangular
trellis tau1=0 limit. Energies use current simultaneous correlations and
include the established MF double-counting terms. Stationary acceptance
requires both ladders to pass in the same sweep; temporal cycles stay unaccepted.

## Seeds

Reuse the established reference correlations: stripe at (1,0), pairing at
(1.4,-0.4). S_A is the original stripe template; S_B is displaced by eight
rungs, half the nominal charge period 16. Spin and all other stripe-reference
correlations are displaced together. P is unshifted on both ladders, retaining
the same global pairing sign.

| Family | A template | B template |
|---|---|---|
| Stripe dominated | 0.95 S_A + 0.05 P | 0.95 S_B + 0.05 P |
| Pairing dominated | 0.95 P + 0.05 S_A | 0.95 P + 0.05 S_B |

Initial fields are rebuilt from the opposite ladder's template with target
couplings. Explicit asymmetry prevents the square calculation from staying
in the invariant A=B subspace. Eight rungs is one registration to probe,
not a predicted optimum; profiles may translate or merge during SCF.

The seed displacement is a cyclic site permutation of the finite reference
correlations, including both matrix indices. It preserves particle number
and matrix symmetry, while also relocating the reference's end structure.
It is only a seed operation: physical bonds remain open, the square kernel
never wraps a bond, and no translated boundary or pinning field is imposed.
Fresh MPSs use random/product-state seed 1404 on each ladder. These templates
are initial fields, not inherited converged MPSs.

Reference bundle SHA-256:
`e01a1ea7d6be813110870d26377db0df529d1584816c946af78584e1e1fbddc1`.
Original reference hashes and the displacement are recorded in each seed.
The [local preparation receipt](prepared_branches.csv) is a preview; host
paths and fingerprints are regenerated on Perlmutter.

## Controls, measurements and cost

- L=64 per ladder, 256 sites per cell, U=8, t=1, tp=0.1, t0=1.4,
  chi=200, r_range=4, real fields and dense GPU MPSs.
- Exact |E_p|=0.249624358808660 at V=-0.4 and 0.206800262974070 at V=-0.2.
  No interpolation or new pair-binding run.
- At most 60 cell sweeps, minimum 40 before acceptance, ten stable records.
  Four jobs contain eight spatial MPSs and at most 480 density-targeted ladder
  solves. Raw updates throughout: damping=1, no Anderson.
- Field absolute/relative tolerances 1e-7/1e-4, channel noise floor 5e-7,
  energy-window tolerance 1e-7 t/site, inner-DMRG tolerance 1e-7 t total,
  and density tolerance 1e-5 per ladder. These retain the recent raw-campaign
  gates; no further threshold relaxation is introduced.
- Every sweep stores applied/measured fields, raw correlations, per-ladder
  energies and simultaneous cell energies. Full terminal raw/connected
  pair-pair, charge/spin, density-spin, single-particle, anomalous,
  double-occupancy and entanglement measurements are enabled on A and B for
  accepted and maximum-iteration states. These are intraladder measurements.

Each job requests one GPU and 32 logical CPU cores in shared QOS for at most
**16 hours**, retaining the **11.5-hour SCF deadline** and allowing 4.5 hours
for the two terminal measurement passes. This allowance is not a runtime
benchmark. Four jobs reserve at most **16 GPU node-hours** under the existing
400-additional-node-hour control. Actual charge uses elapsed allocation time;
see [NERSC shared-QOS policy](https://docs.nersc.gov/jobs/policy/).
There is one segment per job and no automatic extension. A solver deadline
may stop before 60 sweeps. Full states are saved before measurements;
deadline stops do not automatically start an expensive measurement pass,
but can be measured explicitly with the existing offline workflow.

## User-run Perlmutter handoff

On **Perlmutter**:

```bash
cd "$CFS/m4863/MPS-MFT/ladder_mps_mft"
git pull --ff-only
module load julia
bash slurm/submit_square_two_ladder.sh
```

The wrapper imports the original anchor run.env to reuse the account,
scratch/result roots and append-only budget ledgers. It sets the current
checkout and new config, prepares exactly four branches, reconciles completed
reservations, and submits through the existing budget/GPU-preference guards.
It refuses an existing run ID. An optional argument selects a different new
ID. Keep this source checkout fixed while the jobs run.

```bash
bash slurm/phase1_gpu.sh status 20260920_square_two_ladder_two_basin_60
```

After syncing results, plot the full histories separately:

```julia
include("plot_phase1_mf_observables.jl")
plot_phase1_mf_profiles_and_middle_histories(state_file; ladder=:A)
plot_phase1_mf_profiles_and_middle_histories(state_file; ladder=:B)
```

These display MF fields. For physical profiles use that ladder's stored
correlations; its incoming field is sourced by the other ladder. New files
use `spatial_cell_mps_mft_state`; complete-cell resumes and compact mirrors
retain both members. Historical trellis files and one-ladder fingerprints
remain compatible.

## Local validation

1107 square assertions passed, covering explicit bonds/ends, A=B reduction,
per-site energy normalization, finite-difference variational derivatives,
four preparations, seed asymmetry, full histories, cap-triggered measurements,
compaction and complete-cell resume on tiny CPU ladders. Existing trellis and
measurement checks also pass; plotting passes 119 assertions including both
square A/B full-history sliders. Nine local launcher checks use fake commands
without Slurm or remote access. No production L64 DMRG calculation or GPU
benchmark was run locally; these checks do not establish scientific convergence.
