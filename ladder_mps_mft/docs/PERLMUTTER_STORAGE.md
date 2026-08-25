# Perlmutter storage and stateless analysis mirrors

## Storage contract

Launcher version 1.3.0 restores the legacy stateless-data design principle and
makes scratch the authoritative location for heavyweight numerical artifacts:

- Full `state.h5`, rolling checkpoints, orbit files, and pair-binding
  `psi_N_*` sectors live below
  `$PSCRATCH/MPS-MFT/ladder_mps_mft/phase1_gpu/RUN_ID`.
- The CFS campaign directory under `output/phase1_gpu/RUN_ID` retains configs,
  manifests, logs, the budget ledger, and analysis-ready HDF5 mirrors.
- Every HDF5 mirror recursively omits `psi`, orbit-member `psi`, and
  `psi_N_<sector>` objects. It preserves complete applied/measured MF histories,
  seeds, fields, correlations, energy data, diagnostics, and provenance.
- `analysis_storage/full_artifact_path` and
  `analysis_storage/full_artifact_sha256` bind a mirror to its full artifact.
  `stateless_manifest.tsv` records both full and compact hashes and sizes.
- A stateless file can be used by plotting, audits, branch selection, and
  field-only `inherit_from`. It cannot be used as `parent_checkpoint` or
  `resume_checkpoint`; those operations resolve the full scratch artifact.

Each Phase 1 and guarded `E_p` job requests the `scratch,cfs` Slurm licenses.
After the numerical process exits, the worker automatically refreshes the CFS
mirror. If a batch job is killed before that post-processing step, rerun it for
the affected branch:

```bash
PROJECT=/global/cfs/cdirs/m4863/MPS-MFT/ladder_mps_mft
RUN_ID=YOUR_RUN_ID
LABEL=frustrated__pairing_s1
FULL_RUN="$PSCRATCH/MPS-MFT/ladder_mps_mft/phase1_gpu/$RUN_ID"

julia --project="$PROJECT" "$PROJECT/scripts/compact_results.jl" \
  "$FULL_RUN/results/$LABEL" \
  "$PROJECT/output/phase1_gpu/$RUN_ID/results/$LABEL"
```

The scratch path is recorded in both `run.env` and `full_storage_path.txt`, so
continuations and recovery campaigns use full artifacts even though routine
analysis uses the compact CFS tree.

## One-time migration of an existing campaign

Do not migrate a campaign while any job is writing it. Repair any known corrupt
files first, and require every branch to be in a terminal Slurm state. In
particular, wait for the current Float64-history campaign to finish before
moving it.

The recommended migration is one command. It creates `~/.globus` if needed,
submits and waits for the NERSC Globus transfer, verifies every transferred
file against a SHA-256 inventory, builds and verifies the stateless CFS mirror,
records the scratch location, and preserves old absolute parent paths with a
symlink:

```bash
bash slurm/migrate_phase1_to_scratch.sh --prune-cfs 20260823_phase1_gpu_v2
```

Omit `--prune-cfs` to retain the verified full CFS tree under a timestamped
`results.full_cfs.pending-delete.*` name. The script is idempotent after a
completed migration and refuses campaigns with pending, running, or unknown
jobs. The remaining commands in this section document its stages for manual
recovery only.

Set bounded source and destination paths on Perlmutter:

```bash
PROJECT=/global/cfs/cdirs/m4863/MPS-MFT/ladder_mps_mft
RUN_ID=20260823_phase1_gpu_v2                 # repeat for each completed run
CONTROL_RUN="$PROJECT/output/phase1_gpu/$RUN_ID"
FULL_RUN="$PSCRATCH/MPS-MFT/ladder_mps_mft/phase1_gpu/$RUN_ID"
TRANSFER_LIST="$CONTROL_RUN/transfer-to-scratch.txt"

mkdir -p "$FULL_RUN"
printf '%s\n' "$CONTROL_RUN/results" > "$TRANSFER_LIST"
```

Use NERSC Globus—not `rsync`, `scp`, or a Perlmutter login node—for the large
CFS-to-scratch transfer:

```bash
module load globus-tools
transfer_files.py -s dtn -t perlmutter -d "$FULL_RUN" -i "$TRANSFER_LIST" -p
```

Save the printed transfer ID and wait for `SUCCEEDED`:

```bash
module load globus-tools
check_transfer.py -i TRANSFER_ID -p
```

Globus performs transfer integrity checking. For an additional reproducible
inventory, hash the completed, quiescent source tree and verify it at the
scratch destination:

```bash
(
  cd "$CONTROL_RUN/results"
  find . -type f -print0 | sort -z | xargs -0 sha256sum
) > "$CONTROL_RUN/full-results.sha256"

(
  cd "$FULL_RUN/results"
  sha256sum -c "$CONTROL_RUN/full-results.sha256"
)
```

Create and verify a separate compact staging tree before changing CFS:

```bash
julia --project="$PROJECT" "$PROJECT/scripts/compact_results.jl" \
  "$FULL_RUN/results" "$CONTROL_RUN/stateless_results"

julia --project="$PROJECT" "$PROJECT/scripts/verify_stateless_results.jl" \
  "$CONTROL_RUN/stateless_results" --full

du -sh "$CONTROL_RUN/results" "$FULL_RUN/results" "$CONTROL_RUN/stateless_results"
```

For old campaigns, preserve their already-recorded absolute parent paths by
replacing the original CFS `results` directory with a symlink to the verified
scratch copy. The plotting and audit scripts prefer `stateless_results`, so
they do not traverse that full-state symlink for ordinary analysis:

```bash
printf '%s\n' "$FULL_RUN" > "$CONTROL_RUN/full_storage_path.txt"
mv "$CONTROL_RUN/results" "$CONTROL_RUN/results.full_cfs.pending-delete"
ln -s "$FULL_RUN/results" "$CONTROL_RUN/results"

readlink "$CONTROL_RUN/results"
julia --project="$PROJECT" "$PROJECT/scripts/verify_stateless_results.jl" \
  "$CONTROL_RUN/stateless_results" --full
```

Only after the symlink target, all hashes, and a representative plot/audit have
been checked should the explicitly named CFS holding directory be deleted:

```bash
rm -r -- "$CONTROL_RUN/results.full_cfs.pending-delete"
```

That final removal frees CFS space and is destructive. It does not make scratch
durable. For local synchronization, copy `stateless_results`, configs,
manifests, logs, and plots; do not copy or follow the `results` symlink.

## Retention boundary

Perlmutter scratch is temporary, unbacked-up storage. Files not accessed for
eight weeks can be purged. Archive scientifically irreplaceable accepted full
states (and any checkpoint needed for future continuation) to the project's
HPSS location before relying on scratch as the only full copy. The compact CFS
and local mirrors are sufficient for analysis and publication figures, but not
for DMRG restart or MPS-level diagnostics.
