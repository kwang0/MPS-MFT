# Perlmutter storage and stateless analysis mirrors

## Storage contract

Launcher version 1.11.0 retains the legacy stateless-data design principle and
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

The recommended migration is one command. It copies the mounted CFS tree into
a scratch staging directory, verifies every file against a SHA-256 inventory,
atomically installs the verified scratch directory, builds and verifies the
stateless CFS mirror, records the scratch location, and preserves old absolute
parent paths with a symlink:

```bash
bash slurm/migrate_phase1_to_scratch.sh --prune-cfs 20260823_phase1_gpu_v2
```

If an older campaign contains known truncated auxiliary checkpoints or orbit
snapshots, add `--prune-corrupt-auxiliary`. The cleaner scans every HDF5 file,
removes unreadable auxiliary files from both CFS and scratch, and refuses the
entire cleanup if any final `state.h5` is unreadable:

```bash
bash slurm/migrate_phase1_to_scratch.sh --prune-cfs \
  --prune-corrupt-auxiliary 20260823_phase1_gpu_v2
```

Omit `--prune-cfs` to retain the verified full CFS tree under a timestamped
`results.full_cfs.pending-delete.*` name. The script is idempotent after a
completed migration and refuses campaigns with pending, running, or unknown
jobs. The remaining commands in this section document its stages for manual
recovery only.

For the one v2 migration whose Globus task was already submitted, the script
detects the recorded task and waits up to ten minutes for its existing scratch
tree to pass the same exact file-count and SHA-256 gates. It will not start a
concurrent local copy.

The equivalent bounded manual copy is:

```bash
PROJECT=/global/cfs/cdirs/m4863/MPS-MFT/ladder_mps_mft
RUN_ID=20260823_phase1_gpu_v2                 # repeat for each completed run
CONTROL_RUN="$PROJECT/output/phase1_gpu/$RUN_ID"
FULL_RUN="$PSCRATCH/MPS-MFT/ladder_mps_mft/phase1_gpu/$RUN_ID"

mkdir -p "$FULL_RUN"
(
  cd "$CONTROL_RUN/results"
  find . -type f -print0 | sort -z | xargs -0 sha256sum
) > "$CONTROL_RUN/full-results.sha256"

COPY_STAGING="$(mktemp -d "$FULL_RUN/.results.copying.XXXXXX")"
cp -a -- "$CONTROL_RUN/results"/. "$COPY_STAGING"/
(
  cd "$COPY_STAGING"
  sha256sum -c "$CONTROL_RUN/full-results.sha256"
)
mv -- "$COPY_STAGING" "$FULL_RUN/results"
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

## Prune redundant stateless transfer extras

Even MPS-free mirrors can become large because final states, rolling
checkpoints, and detected-orbit snapshots repeat complete field histories and
correlations. `scripts/prune_phase1_stateless_extras.py` removes only the
redundant compact `checkpoint_best.h5`, `checkpoint_latest.h5`, and
`orbit_period_*_iter_*.h5` files. It retains every final `state.h5`, diagnostic,
summary, configuration, log, and control manifest. It never changes the full
scratch tree or any HDF5 file in place.

The default is a read-only, hash-checking plan. It supports both migrated
campaign-wide `stateless_results` manifests and newer branch-level manifests
under `results`:

```bash
python3 scripts/prune_phase1_stateless_extras.py \
  20260823_phase1_gpu_v2 \
  20260824_phase1_gpu_v3_float64_history \
  RUN_ID \
  20260826_phase1_unfrustrated_pairing_recurrence_chi400
```

On Perlmutter, verify every recorded full source before applying the plan:

```bash
python3 scripts/prune_phase1_stateless_extras.py --apply --require-full \
  20260823_phase1_gpu_v2 \
  20260824_phase1_gpu_v3_float64_history \
  RUN_ID \
  20260826_phase1_unfrustrated_pairing_recurrence_chi400
```

For a disposable workstation copy whose scratch paths are not mounted, replace
`--require-full` with the explicit `--local-only` boundary. Each changed
manifest is backed up as `stateless_manifest.before-prune-TIMESTAMP.tsv`; the
active manifest is rewritten atomically and remains compatible with
`verify_stateless_results.jl`. Apply mode runs that Julia verifier both before
and after pruning, in addition to the Python hash and size checks. The utility
is compatible with the older Python 3.6 interpreter available as `python3` on
some Perlmutter login environments. The removed compact extras can be regenerated
from the recorded full scratch artifacts. Do not apply the tool to an active
campaign, and archive irreplaceable full states to HPSS independently.

## Current square V=0 staged campaign

Launcher v1.11.0 adds a preparation-only action for the six-seed square
`V=0,t0=1.4,chi=200` campaign. It preserves the same scratch-first full output
and stateless CFS mirror contract described above:

```bash
bash slurm/phase1_gpu.sh plan-square-v0-seed-pilot
SQUARE_V0_RUN=20260901_phase1_square_t014_v000_seed_chi200_loose
bash slurm/phase1_gpu.sh prepare-square-v0-seed-pilot "$SQUARE_V0_RUN"
```

Neither command submits or reserves. Under launcher v1.13.1, the plan-only
first-segment envelope is `18.000` node-hours (six `3.000` branch
reservations). Recheck the live Perlmutter ledger before direct submission.
If a smoke was already submitted from the v1.12.0-prepared campaign, its
existing reservation remains visible until the terminal job is reconciled.
