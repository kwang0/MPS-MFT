# Eight square t_perp scan starts

Prepared locally September 22, 2026 at the user's request. No Perlmutter
connection, transfer, submission or scheduler operation was performed.
The user reports ongoing trellis jobs; their current scheduler state and job
IDs have not been checked locally. Preserve their existing checkout.

| t0 | V | t_perp | Seeds | Exact signed E_p |
|---:|---:|---:|---|---:|
| 1.4 | 0.0 | 0.06 | stripe, pairing | -0.14653773091916378 |
| 1.4 | 0.0 | 0.08 | stripe, pairing | -0.14653773091916378 |
| 1.4 | -0.2 | 0.12 | stripe, pairing | -0.2068002629740704 |
| 1.4 | -0.2 | 0.14 | stripe, pairing | -0.2068002629740704 |

These are eight fresh, independent one-ladder square runs, using the same
protocol as the coarse square grid. They are not parameter continuations or
new two-ladder tests. L=64, t=1, U=8, n=15/16, chi=200 and r_range=4 are fixed.
The existing exact chi=1000 pair-binding registry rows are reused. E_p does
not depend on interladder hopping; no new backbone or interpolation is needed.

Each seed combines 95% of its primary reference correlations with 5% of the
competing reference. All alpha, beta and density fields are rebuilt with
the target t_perp and |E_p|. Fresh MPS initialization uses random seed 1404.
Reference bundle SHA-256:
`e01a1ea7d6be813110870d26377db0df529d1584816c946af78584e1e1fbddc1`.
Labels, manifest rows and HDF5 provenance explicitly record t_perp.
The [local preview receipt](prepared_branches.csv) records hashes and
fingerprints; Perlmutter preparation regenerates host-specific paths/hashes.

## Numerical controls and resources

Retain the existing 60 raw MF evaluation cap, minimum 40 evaluations and ten
stable records, with no damping or Anderson. Existing convergence thresholds
and acceptance flags remain unchanged. This pilot maps qualitative order;
nonaccepted endpoints can still supply phase-profile evidence. The late loss
of pairing at the old V=0, t_perp=0.1 anchor motivates retaining histories.
Central spin, charge profiles and physical pair amplitudes should be inspected
together, since the MF fields themselves scale with t_perp squared.

Full terminal correlations remain enabled under the current measurement
workflow. Each job requests one GPU, 32 logical CPU cores and 16 hours in
shared QOS, with the unchanged 11.5-hour solver deadline and 4.5 hours of
measurement allowance. Deadline endpoints retain the existing offline
measurement option. There is one segment per job, no automatic continuation,
at most 480 MF evaluations and a **32 fractional node-hour reservation ceiling**
(8 x 16 x 0.25), under the existing 400-additional-node-hour cap. This ceiling
is not an actual cost estimate. The user-run launcher reconciles completed
jobs and checks the shared live budget before submitting.

## Isolation from ongoing trellis jobs

The wrapper refuses to run in the original checkout. The handoff fetches
and creates a detached worktree, leaving the original working files and HEAD
unchanged. Solver source, trellis preparer, trellis configs and trellis wrapper
are unchanged by this preparation. Leave both checkouts fixed while their
respective jobs are queued or running.

The new wrapper reads the original anchor run.env for the existing account,
scratch and shared budget/reconciliation locations, then selects the new
source checkout. A unique campaign ID separates full scratch outputs:
`20260922_square_t014_tp_scan_95_5_60`.
Control/log/stateless results use a new `square_tp_scan/` subdirectory below
the inherited PHASE1_RUN_ROOT, so its latest_run.txt does not replace the
trellis pointer. Existing run and scratch directories are refused, and no
trellis job is canceled, modified or resubmitted. Budget updates retain the
existing lock and append-only accounting.

## Perlmutter handoff — user-run only

Do **not** pull into the checkout used by the ongoing trellis jobs. After this
preparation is published on `codex/mps-mft-phase0-refactor`, run:

```bash
cd "$CFS/m4863/MPS-MFT/ladder_mps_mft"
git -C .. fetch origin
git -C .. worktree add --detach "$CFS/m4863/MPS-MFT-square-tp-20260922" origin/codex/mps-mft-phase0-refactor
cd "$CFS/m4863/MPS-MFT-square-tp-20260922/ladder_mps_mft"
bash slurm/submit_square_tp_scan.sh
```

The reference bundle and GPU Manifest are versioned; runtime preferences are
checked before submission. The wrapper prepares exactly the eight listed
branches, reconciles and submits through the existing guarded launcher.
Progress can be checked with the new launcher and the explicit control path:

```bash
cd "$CFS/m4863/MPS-MFT-square-tp-20260922/ladder_mps_mft"
bash slurm/phase1_gpu.sh status "$CFS/m4863/MPS-MFT/ladder_mps_mft/output/phase1_gpu/square_tp_scan/20260922_square_t014_tp_scan_95_5_60"
```

That path assumes the standard inherited control root; the preparation output
prints the authoritative control path if the original run used an override.

## Local validation

`test/test_square_tp_scan.jl` passed 178 assertions: exact coordinate coverage,
exact denominators, fresh lineage, seed hashes and readback, nonzero competing
fields, quadratic scaling of all three field arrays, four distinct model
fingerprints, matched numerical settings, overwrite refusal and unchanged
four-anchor preparation at t_perp=0.1.

Eight focused Python launcher tests passed. They exercise checkout isolation, separate
latest-pointer root, shared accounting, the 16-hour/one-segment envelope,
custom IDs, preparation-failure stop, actual campaign validation and trellis
launcher-version compatibility. They run only local fake launchers and
extracted validation functions, never Slurm. No DMRG, GPU benchmark or full
test suite is required for these preparation and launcher changes.
