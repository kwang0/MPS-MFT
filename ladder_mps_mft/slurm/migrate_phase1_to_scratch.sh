#!/bin/bash

# Move one completed pre-v1.3 Phase 1 result tree from CFS to Perlmutter
# scratch, create its MPS-free CFS analysis mirror, and preserve old absolute
# parent paths with a scratch symlink. The default retains a verified CFS
# backup; --prune-cfs removes that backup only after all gates pass.

set -euo pipefail

script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
project_dir="${PHASE1_PROJECT_DIR:-$(cd "$(dirname "$script_path")/.." && pwd)}"
run_root="${PHASE1_RUN_ROOT:-$project_dir/output/phase1_gpu}"
scratch_base="${PSCRATCH:-${SCRATCH:-}}"
julia_command="${PHASE1_JULIA:-julia}"
prune_cfs=0
run_id=""

die() { echo "error: $*" >&2; exit 1; }

usage() {
  cat <<EOF
Usage: bash $script_path [--prune-cfs] RUN_ID

  RUN_ID       Completed Phase 1 campaign below $run_root
  --prune-cfs  After transfer, compaction, and hash verification, remove the
               explicitly quarantined full CFS results directory.

Without --prune-cfs the full CFS directory is retained under a timestamped
results.full_cfs.pending-delete.* name for manual removal later.
EOF
}

for argument in "$@"; do
  case "$argument" in
    --prune-cfs) prune_cfs=1 ;;
    -h|--help) usage; exit 0 ;;
    --*) die "unknown option: $argument" ;;
    *)
      [[ -z "$run_id" ]] || die "provide exactly one RUN_ID"
      run_id="$argument"
      ;;
  esac
done

[[ -n "$run_id" ]] || { usage; die "RUN_ID is required"; }
[[ "$run_id" =~ ^[A-Za-z0-9_.-]+$ ]] || die "unsafe run ID: $run_id"
[[ -n "$scratch_base" ]] || die "PSCRATCH/SCRATCH is not set; run this on Perlmutter"
[[ "$scratch_base" == /* ]] || die "scratch path must be absolute: $scratch_base"

control_run="$run_root/$run_id"
full_run="$scratch_base/MPS-MFT/ladder_mps_mft/phase1_gpu/$run_id"
source_results="$control_run/results"
stateless_results="$control_run/stateless_results"
transfer_list="$control_run/transfer-to-scratch.txt"
transfer_log="$control_run/transfer-to-scratch.log"
source_hash_manifest="$control_run/full-results.sha256"

[[ -d "$control_run" ]] || die "campaign not found: $control_run"
[[ -f "$control_run/jobs.tsv" ]] || die "campaign has no jobs.tsv: $control_run"
[[ -f "$project_dir/scripts/compact_results.jl" ]] || die "missing compact_results.jl"
[[ -f "$project_dir/scripts/verify_stateless_results.jl" ]] || die "missing stateless verifier"
command -v sacct >/dev/null 2>&1 || die "sacct is required; run this on Perlmutter"

slurm_state() {
  local job_id="$1"
  sacct -n -X -j "$job_id" --format=State -P 2>/dev/null | awk -F'|' \
    'NF {gsub(/[ +].*/, "", $1); print $1; exit}'
}

require_terminal_campaign() {
  local kind label job_id state nonterminal=0
  while IFS=$'\t' read -r kind label _ _ _ _ job_id _; do
    [[ "$kind" == "kind" ]] && continue
    [[ -n "$job_id" ]] || continue
    state="$(slurm_state "$job_id")"
    case "$state" in
      COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|PREEMPTED|NODE_FAIL) ;;
      *)
        printf 'nonterminal_job=%s\t%s\t%s\t%s\n' "$kind" "$label" "$job_id" "${state:-UNKNOWN}" >&2
        nonterminal=1
        ;;
    esac
  done <"$control_run/jobs.tsv"
  (( nonterminal == 0 )) || die "campaign still has nonterminal or unknown jobs"
}

find_status_tool() {
  local candidate
  for candidate in check_transfer.py check_transfers.py; do
    if command -v "$candidate" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  die "globus-tools provides neither check_transfer.py nor check_transfers.py"
}

wait_for_transfer() {
  local status_tool="$1" transfer_id="$2" output status
  while true; do
    output="$($status_tool -i "$transfer_id" -p)"
    printf '%s\n' "$output"
    status="$(printf '%s\n' "$output" | awk -F'|' -v id="$transfer_id" \
      'index(tolower($1), id) {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2}' | tail -1)"
    case "$status" in
      SUCCEEDED) return ;;
      FAILED|CANCELLED) die "Globus transfer $transfer_id ended with $status" ;;
      *) sleep 20 ;;
    esac
  done
}

verify_already_migrated() {
  [[ -L "$source_results" ]] || return 1
  [[ "$(readlink "$source_results")" == "$full_run/results" ]] || return 1
  [[ -f "$stateless_results/stateless_manifest.tsv" ]] || return 1
  "$julia_command" --startup-file=no --project="$project_dir" \
    "$project_dir/scripts/verify_stateless_results.jl" "$stateless_results" --full
}

require_terminal_campaign
mkdir -p "$full_run"

if verify_already_migrated; then
  echo "migration_already_complete=true"
  echo "full_results=$full_run/results"
  echo "stateless_results=$stateless_results"
  exit 0
fi

[[ -d "$source_results" && ! -L "$source_results" ]] || die \
  "expected an unmigrated CFS results directory: $source_results"

# NERSC's globus-tools token adapter does not create its parent directory.
mkdir -p "$HOME/.globus"
chmod 700 "$HOME/.globus"

# The module function exists in Perlmutter login shells.
type module >/dev/null 2>&1 || die "environment modules are unavailable; run from a Perlmutter login shell"
module load globus-tools
command -v transfer_files.py >/dev/null 2>&1 || die "transfer_files.py not found after loading globus-tools"
status_tool="$(find_status_tool)"

printf '%s\n' "$source_results" >"$transfer_list"
echo "Hashing the quiescent CFS result tree before transfer..."
(
  cd "$source_results"
  find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum
) >"$source_hash_manifest"
[[ -s "$source_hash_manifest" ]] || die "CFS result tree contains no files: $source_results"
echo "source_results=$source_results"
echo "full_results=$full_run/results"
echo "Starting NERSC Globus transfer. If prompted, open the URL and paste a NEW one-time code."

set +e
PYTHONUNBUFFERED=1 transfer_files.py -s dtn -t perlmutter \
  -d "$full_run" -i "$transfer_list" -p 2>&1 | tee "$transfer_log"
transfer_status=${PIPESTATUS[0]}
set -e
(( transfer_status == 0 )) || die "transfer_files.py failed; see $transfer_log"

transfer_id="$(grep -Eo '[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}' \
  "$transfer_log" | tail -1 | tr 'A-F' 'a-f' || true)"
[[ -n "$transfer_id" ]] || die "could not parse Globus transfer ID from $transfer_log"
echo "transfer_id=$transfer_id"
wait_for_transfer "$status_tool" "$transfer_id"

[[ -d "$full_run/results" ]] || die "successful transfer did not create $full_run/results"
echo "Verifying every transferred file against the CFS SHA-256 inventory..."
(
  cd "$full_run/results"
  sha256sum -c "$source_hash_manifest"
)
source_file_count="$(wc -l <"$source_hash_manifest" | tr -d '[:space:]')"
destination_file_count="$(find "$full_run/results" -type f | wc -l | tr -d '[:space:]')"
[[ "$destination_file_count" == "$source_file_count" ]] || die \
  "scratch tree has $destination_file_count files; expected exactly $source_file_count"
module load julia

staging="$(mktemp -d "$control_run/.stateless_results.staging.XXXXXX")"
echo "Building MPS-free analysis mirror..."
"$julia_command" --startup-file=no --project="$project_dir" \
  "$project_dir/scripts/compact_results.jl" "$full_run/results" "$staging"
"$julia_command" --startup-file=no --project="$project_dir" \
  "$project_dir/scripts/verify_stateless_results.jl" "$staging" --full

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -e "$stateless_results" ]]; then
  mv "$stateless_results" "$control_run/stateless_results.previous.$timestamp"
fi
mv "$staging" "$stateless_results"
printf '%s\n' "$full_run" >"$control_run/full_storage_path.txt"

quarantine="$control_run/results.full_cfs.pending-delete.$timestamp"
mv "$source_results" "$quarantine"
if ! ln -s "$full_run/results" "$source_results"; then
  mv "$quarantine" "$source_results"
  die "could not install scratch symlink; restored original CFS results"
fi

[[ "$(readlink "$source_results")" == "$full_run/results" ]] || die "scratch symlink verification failed"
"$julia_command" --startup-file=no --project="$project_dir" \
  "$project_dir/scripts/verify_stateless_results.jl" "$stateless_results" --full

if (( prune_cfs == 1 )); then
  [[ -d "$quarantine" && ! -L "$quarantine" ]] || die "refusing to prune unexpected quarantine path: $quarantine"
  rm -r -- "$quarantine"
  echo "cfs_full_results_pruned=true"
else
  echo "cfs_full_results_pruned=false"
  echo "verified_cfs_quarantine=$quarantine"
  echo "To free CFS space after inspection: rm -r -- '$quarantine'"
fi

echo "migration_complete=true"
echo "full_results=$full_run/results"
echo "stateless_results=$stateless_results"
echo "results_symlink=$source_results"
