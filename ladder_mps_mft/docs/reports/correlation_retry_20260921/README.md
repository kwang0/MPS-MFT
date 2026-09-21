# Correlation backfill recovery — September 21, 2026

The original `20260918_latest_correlations` campaign submitted 56 CPU jobs
for 58 spatial MPSs. All 56 synchronized logs fail before Julia starts:
the spooled batch script tries to source `phase1_gpu.sh` beside itself in
`/var/spool/slurmd/job.../`. No measurement files or receipts are present.
The original manifest, source snapshot, jobs, logs and SCF states remain
unchanged.

The corrected worker dispatches `_run` using its explicit run-directory
argument before loading submission helpers. It retains frozen-source and
manifest hash verification, four Julia threads, single-threaded BLAS and
the existing CPU measurement command. Scientific measurement code is unchanged.

## Prepared retry

`slurm/retry_latest_correlations.sh` defaults to the failed parent above and
new run ID **`20260921_latest_correlations_retry1`**. Its default action is
read-only `plan`.

- Verifies the parent manifest and frozen source hashes, all 56 job records
  and the exact known startup failure in every parent log. Refuses a parent
  with a results directory, since partial measurements need separate review.
- Reuses the parent's shared accounting paths and measurement wall limit.
  Submission reconciles **only the failed measurement campaign** and then
  requires failure accounting for every original job. Missing/running jobs
  stop the retry even if the accounting command itself returns success.
- Uses the existing inventory and source-preparation workflow. Before any
  new job is submitted, the new manifest must match the original byte for
  byte; its parent-manifest hash and path are saved. Changed source-state
  selection stops submission instead of silently measuring different states.
- Creates a separate source snapshot, output directory, jobs.tsv and log
  directory. Resuming an interrupted retry skips its already recorded jobs;
  repeating submission does not automatically rerun failed retry jobs.
- Preserves the 9-node-hour measurement cap and shared 400-additional-node-hour
  project cap. Other campaigns' reservations remain included in that cap.

Scope: 18 latest square-grid branches, 18 cubic branches, 12 square fine-cut
branches, four positive-V square branches, and four trellis branches (six
MPSs). The two superseded square V=0 parents remain excluded.

On September 21 the user reports **two new square A/B jobs still ongoing**.
Their identities and live status were not checked locally. That four-job
campaign is outside the fixed backfill inventory and is neither submitted,
continued, cancelled nor reconciled by this wrapper. Its own automatic
terminal measurements remain independent.

Resources are unchanged: 56 CPU shared jobs, four Julia threads, eight Slurm
logical CPUs, 32 GiB per job; two hours per single-ladder endpoint and four
hours for each two-ladder trellis endpoint. The total requested ceiling is
**8.15625 CPU node-hours**, not an observed runtime or billed-cost estimate.

## User-run Perlmutter commands

Use the existing `codex/mps-mft-phase0-refactor` branch in the original
Perlmutter checkout. A fast-forward pull obtains both launchers and this
handoff; no individual script transfer is needed. Frozen campaign sources
and the ongoing square A/B jobs are unaffected.

```bash
cd "$CFS/m4863/MPS-MFT/ladder_mps_mft"
git pull --ff-only
module load julia
bash slurm/retry_latest_correlations.sh plan
```

Then submit the prepared retry:

```bash
cd "$CFS/m4863/MPS-MFT/ladder_mps_mft"
bash slurm/retry_latest_correlations.sh submit
```

Submission includes the required read-only Phase 0 CPU plan. It verifies
full scratch-MPS availability during preparation and each full-state SHA-256
inside its worker. No DMRG optimization or sector-gap calculation is requested.
If any gate fails, retain the error and preparation directory for review;
do not remove the original campaign or job records to force a retry.

Completion and accounting are separate user-run checks:

```bash
cd "$CFS/m4863/MPS-MFT/ladder_mps_mft"
bash slurm/measure_latest_campaigns.sh status 20260921_latest_correlations_retry1
bash slurm/measure_latest_campaigns.sh reconcile 20260921_latest_correlations_retry1
```

`status` checks receipts and diagnostic hashes; it is not a live queue view.
After completion, synchronize the new campaign's results, manifest, receipts
and logs for analysis. The original submitted manifest SHA-256 is
`6fab4585b0887a17ac310b0d01349bbde0398992aab32c6f4c05cadfa0fbab0f`.

## Validation boundary

Local validation uses fake scheduler/accounting commands and a spooled-script
fixture with no sibling helper. It checks worker dispatch, frozen-source
integrity, exit-status propagation, retry provenance, duplicate prevention,
terminal-accounting requirements, changed-inventory rejection and preservation
of another campaign's reservations. Local compact-state inventory checks do
not verify current Perlmutter scratch availability or live accounting.

Checked locally on September 21:

```powershell
C:/Python313/python.exe -B -m unittest discover -s ladder_mps_mft/test -p test_measurement_launcher.py -v
julia --startup-file=no --compiled-modules=existing --project=ladder_mps_mft ladder_mps_mft/scripts/measure_latest_campaigns.jl plan ladder_mps_mft/output/phase1_gpu --local
```

All nine launcher tests passed in 177 seconds. Inventory completed with
`ready_states=56/56 mps_measurements=58`. No measurement contractions or DMRG
tests were required for these shell-only changes. `git diff --check` passed.

No Perlmutter authentication, transfer, submission or scheduler action is
performed by Codex. Validation results are recorded in `docs/RUN_LOG.md`.
