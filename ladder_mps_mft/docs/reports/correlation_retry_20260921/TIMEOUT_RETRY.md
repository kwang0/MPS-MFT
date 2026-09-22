# Seven missing correlation measurements — September 21, 2026

**Completed September 22:** all seven retries have verified receipts. Together
with retry1, coverage is 56/56 branches and 58/58 MPSs. The [correlation report](../pair_correlations_20260922/README.md)
analyzes the complete set. The commands below are the historical handoff,
not a request to submit another retry.

The user-provided status reports 49/56 branches measured in
`20260921_latest_correlations_retry1` (51/58 spatial MPSs). Accounting shows
the remaining seven jobs hit their two-hour wall limit. Their source states
remain suitable for retrospective measurement; this retry requests no DMRG.

| Parent row | Parent job | Endpoint |
|---|---|---|
| 6 | 58712480 | Square pairing, t0=1, V=-0.2 |
| 11 | 58712491 | Square stripe, t0=1.2, V=-0.2 |
| 31 | 58712512 | Cubic stripe, t0=1.4, V=-0.4 |
| 34 | 58712515 | Cubic pairing, t0=1.4, V=-0.2 |
| 46 | 58712527 | Fine-square pairing, t0=1.3, V=-0.4 |
| 51 | 58712532 | Positive-V square, period-eight seed |
| 53 | 58712534 | One-ladder trellis stripe |

`slurm/complete_missing_correlations.sh` prepares
**`20260921_latest_correlations_retry2`**, with seven single-MPS CPU jobs,
**four hours per job**, four Julia threads, eight Slurm logical CPUs and
32 GiB each. The requested ceiling is **1.96875 CPU node-hours** under the
existing 9/128-node billing convention. Four hours provides extra time; it
is not a measured runtime guarantee.

## Run on Perlmutter

Use the existing `codex/mps-mft-phase0-refactor` checkout:

```bash
cd "$CFS/m4863/MPS-MFT/ladder_mps_mft"
git pull --ff-only
module load julia
bash slurm/complete_missing_correlations.sh plan &&
  bash slurm/complete_missing_correlations.sh submit
```

The default `plan` is read-only. Submission runs the mandatory Phase 0 CPU
plan, reconciles only retry1 measurement jobs, requires TIMEOUT accounting
for all seven selected jobs, and applies the shared 400-node-hour cap.
The two user-reported ongoing square A/B jobs are outside this retry; their
reservations remain included in the shared cap.

The script verifies all 49 successful parent entries through their receipts
and diagnostic hashes and requires precisely the seven listed rows to remain
missing. If that pattern changes, it stops for inspection. It copies the
parent's frozen measurement source and selects the same seven manifest rows,
retaining the full MPS paths, hashes, configurations and convergence flags.
It checks full-state presence and configuration hashes before submission;
each worker verifies the full MPS SHA before contraction.

The new manifest has indices 1–7 because the measurement worker selects a
row by position. `retry_rows.tsv` maps these to original rows and job IDs.
Successful parent outputs and partial timeout files remain untouched. New
measurements go into retry2's fresh results directory. Parent manifest/job
hashes and the copied scientific source are checked on resumed submissions.
Repeated `submit` skips jobs already recorded in retry2, including jobs that
later fail; it does not create another attempt automatically.

## Check completion

```bash
cd "$CFS/m4863/MPS-MFT/ladder_mps_mft"
bash slurm/complete_missing_correlations.sh status
```

This prints parent and retry receipt status, followed by
`combined_complete=56/56 branches` when the entire backfill is available
(58 MPSs). The parent alone will still show 49/56: its files are unchanged.
The combined count matches branch identities and does not double count.

For scheduler status, run on Perlmutter:

```bash
cd "$CFS/m4863/MPS-MFT/ladder_mps_mft"
jobs=$(awk -F'\t' 'NR>1 {print $7}' \
  output/phase1_diagnostics/20260921_latest_correlations_retry2/jobs.tsv | paste -sd, -)
sacct -X -j "$jobs" --format=JobID,State,Elapsed,ExitCode
```

After the jobs finish, reconcile their accounting:

```bash
cd "$CFS/m4863/MPS-MFT/ladder_mps_mft"
bash slurm/complete_missing_correlations.sh reconcile
```

For analysis, synchronize both retry1 and retry2 results, receipts,
manifests and logs, including retry2's `retry_rows.tsv`. The current local
workspace has neither retry campaign's measurement results. Status and
timeout evidence here come from the user's Perlmutter output.

## Local validation boundary

Focused tests use fake scheduler, accounting and status commands. They check
the exact subset and parent-index mapping, four-hour resources, budget caps,
active-job rejection, immutable parent files, fresh output paths, duplicate
submission prevention, missing-state/configuration rejection, frozen-source
integrity and combined status without double counting. The scientific
measurement implementation and worker are reused unchanged; no DMRG or
correlation contractions are needed for this launcher change. Live scheduler
checks, submission and all transfers remain user-run.

September 21 validation: Bash syntax passed; all three focused tests passed
in 95.064 seconds with
`C:/Python313/python.exe -B -m unittest discover -s ladder_mps_mft/test -p test_missing_correlations.py -v`.
