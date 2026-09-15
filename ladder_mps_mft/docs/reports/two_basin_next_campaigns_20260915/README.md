# Cubic two-basin grid and short square continuation

Prepared locally on September 15, 2026, following the user's approval of the
[full square-grid analysis](../two_basin_grid_20260915/README.md). No jobs have
been submitted by Codex. The user performs all Perlmutter actions.

## Scope and controls

| Setting | Cubic-unfrustrated grid | Square (t0,V)=(1.4,0) |
|---|---|---|
| Branches | 18: two seeds at each of nine points | 2: continue both original lineages |
| Grid | t0=1.0,1.2,1.4; V=-0.4,-0.2,0 | One point |
| Initial state | Fresh MPS; reciprocal 95%/5% correlation templates | Full saved MPS and latest measured fields |
| chi / ladder length | 200 / L=64 | 200 / L=64 |
| MF evaluation cap | 60 | 20 additional per lineage |
| Minimum evaluations before acceptance | 40 | 10 fresh |
| Stable-history window | 10 | 10 fresh |
| Global relative field tolerance | 1e-4 | 1e-4, unchanged |
| Absolute field tolerance | 3e-7 t | 1e-7 t, unchanged |
| Channel noise floor | 1.5e-6 t | 5e-7 t, as in the square remainder |
| Corrected canonical energy window | 1e-7 t/site | 1e-7 t/site |
| Inner-DMRG stopping and acceptance | 1e-7 t total | 1e-7 t total |
| Density tolerance | 1e-5 | 1e-5 |
| Slurm wall time / solver deadline | 12 h / 11.5 h | 8 h / 7.5 h |
| Allocation ceiling | 54 node-hours | 4 node-hours |

Both use raw F(x) throughout, with no Anderson or adaptive mixing. Pairing,
charge, spin and exchange all evolve. Energy identity and effective-energy
consistency tolerances remain 1e-8 t/site. The full spatial channel-window
and slow-mode extrapolation checks remain active. Physical period-two
solutions retain their separate orbit qualification; they are not relabeled
as fixed points. No automatic continuation is configured.

### Why these settings

The 40-step square grid often identified a basin before profiles stopped
relaxing. Its (1.2,-0.4) pairing start underwent its main collapse only around
iterations 25–35. Sixty cubic evaluations give another twenty steps without
restoring the original 80-step cap; a minimum of forty protects against
short paired transients. This is a starting budget, not a guarantee that all
cubic endpoints will converge.

At fixed tp and E_p, the cubic-unfrustrated density kernel is three times
the square kernel (6g versus 2g, g=tp²/E_p); the same-leg field coefficients
also scale by three. Scaling the absolute field/noise tolerances by three
therefore preserves the square physical-spin resolution. Cubic rung fields
are additionally present, so this is not a claim that every field is simply
three times its square value. The relative tolerances are unchanged.
The seeds are reconstructed with the actual target cubic kernel and exact
target E_p, using the versioned square correlation references.

The new 1e-7 t/site energy window accommodates the tiny energy drift seen
in slowly relaxing square stripes. The inner-DMRG threshold is 1e-7 t for
the **whole ladder**, about 7.8e-10 t/site at 128 sites. It replaces 1e-8 t
total to avoid spending extra sweeps on that last digit at chi=200. These
are separate tolerances. Neither energy change alone can qualify a state:
resolved field motion and coherent spin growth still prevent acceptance.

For square V=0, both seeds continue so their next endpoints share controls.
The stripe lineage starts after 80 evaluations; the pairing lineage after
62. Twenty additional evaluations give maximum cumulative totals of 100
and 82, respectively. The late pairing lineage averaged about 21.6 minutes
per evaluation, motivating an eight-hour allocation despite the short MF
segment. Its collapse may finish within this extension; slow stripe-wall
motion may still fail the unchanged relative/slow-mode checks.

## Checkpoint and plotting continuity

`scripts/prepare_phase1_square_two_basin_finish.jl` pins the two original
compact files, archived configs and their full-artifact links. On Perlmutter
it also requires the full MPS file on the original scratch path, verifies
its SHA-256, model, job ID and restart fields, and checks the E_p registry.
Both new configurations retain the source model fingerprint. The solver
restores the actual chemical potential from the checkpoint.

| Original family | Job | Original iterations | Full checkpoint SHA-256 |
|---|---|---:|---|
| Stripe | 58093802 | 80 | `9ad2d9ea1239727e577be2997a7a9f62e1e6aaeae0653bd8591448f55dc58ea5` |
| Pairing | 58093803 | 62 | `d4b2ef33f8d969e23b519f08642f50eacfd158948c0b6710756613fd89344237` |

The source campaign is `20260908_square_two_basin_95_5_80_anchors`.
An explicit same-model `parent_checkpoint` preserves ancestry for the
existing `plot_phase1_mf_observables.jl` adapter to stitch the original and
continuation histories. Acceptance uses ten **new** records. Every iteration
still saves fields, observables and energy history. The original artifacts
and accepted/status flags remain unchanged.

Only compact artifacts are mirrored locally. `--compact-preview` is a
non-submittable inspection mode: its contract records
`full_sources_verified=false`, and the launcher rejects it. Real preparation
on Perlmutter performs the full-MPS checks before submission.

## Local validation

- All 18 cubic seeds and two square preview configurations passed 182
  preparation assertions, including grid coverage, target-kernel readback,
  controls, ancestry and refusal when full MPS files are absent.
- Replayed the available convergence gates across all 18 immutable square
  histories. Under the square extension settings, the settled V=-0.4
  stripe/pairing controls first pass these gates at 35/27 evaluations.
  Both V=0 histories and both (1.2,-0.4) histories fail at every eligible
  prefix. Tests with the larger cubic tolerances on **unscaled square
  fields** also reject those histories; both settled V=-0.4 controls pass
  at forty. This deliberately more permissive stress test is not a cubic
  convergence forecast. See [the receipt](history_gate_qualification.csv).
- Per-iteration identity/eigenvalue errors are unavailable in the old
  histories: a history-gate pass is not retrospective acceptance. Changing
  the inner-DMRG tolerance may change the newly measured noise. Inspect
  that noise when the new results arrive.
- The existing 75 raw-basin/convergence unit assertions passed. Local fake
  launcher tests verify shared accounting, exact preparation/submit scope,
  one-segment limits, abort on preparation failure, and rejection of compact
  previews by the production validation function. Bash syntax checks pass.
  No DMRG solve, GPU test, scheduler call or remote access was performed.

Reproduce the local qualification from the repository root:

```powershell
julia --startup-file=no --compiled-modules=existing --project=ladder_mps_mft ladder_mps_mft/scripts/qualify_two_basin_next_campaigns.jl
python -m unittest discover -s ladder_mps_mft/test -p 'test_two_basin_next_launchers.py' -v
```

## Perlmutter submission — user runs these commands

The original anchor jobs are complete, so these launchers use the normal
checkout. They reuse the original account, scratch/control roots and shared
budget ledger. Each prepares its complete branch set, reconciles accounting,
then submits it through `phase1_gpu.sh` v1.20.0. The combined reservation
ceiling is **58 node-hours**; actual charges depend on elapsed time. Existing
budget checks apply before submission; no new reservation was made locally.

```bash
cd "$CFS/m4863/MPS-MFT/ladder_mps_mft"
git pull --ff-only
bash slurm/submit_square_two_basin_finish.sh
bash slurm/submit_cubic_unfrustrated_two_basin.sh
```

These submit two and eighteen branches, respectively. Progress:

```bash
bash slurm/phase1_gpu.sh status 20260915_square_t014_v000_two_basin_finish20
bash slurm/phase1_gpu.sh status 20260915_cubic_unfrustrated_two_basin_95_5_60
```

Both wrappers accept one optional alternative run ID and refuse to overwrite
an existing preparation. After a partially successful submission, use
`bash slurm/phase1_gpu.sh submit RUN_ID` to submit only unrecorded branches.
If preparation fails, correct the reported cause and pass a new run ID.
