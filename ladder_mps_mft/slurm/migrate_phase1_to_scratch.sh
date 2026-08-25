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
  --prune-cfs  After copying, compaction, and hash verification, remove the
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

tree_matches_manifest() {
  local directory="$1" expected_count="$2" actual_count
  [[ -d "$directory" ]] || return 1
  actual_count="$(find "$directory" -type f | wc -l | tr -d '[:space:]')"
  [[ "$actual_count" == "$expected_count" ]] || return 1
  (cd "$directory" && sha256sum -c "$source_hash_manifest" >/dev/null 2>&1)
}

wait_for_previous_globus_copy() {
  local expected_count="$1" attempt actual_count
  echo "A Globus task was already submitted for this campaign; waiting for its exact scratch copy."
  for attempt in $(seq 1 30); do
    if tree_matches_manifest "$full_run/results" "$expected_count"; then
      echo "existing_scratch_sha256_verified=true"
      return
    fi
    if [[ -d "$full_run/results" ]]; then
      actual_count="$(find "$full_run/results" -type f | wc -l | tr -d '[:space:]')"
    else
      actual_count=0
    fi
    printf 'existing_copy_poll=%s/30 destination_files=%s/%s\n' \
      "$attempt" "$actual_count" "$expected_count"
    sleep 20
  done
  die "the previously submitted copy did not verify after 10 minutes; do not start a concurrent copy"
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

echo "Hashing the quiescent CFS result tree before copying..."
(
  cd "$source_results"
  find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum
) >"$source_hash_manifest"
[[ -s "$source_hash_manifest" ]] || die "CFS result tree contains no files: $source_results"
source_file_count="$(wc -l <"$source_hash_manifest" | tr -d '[:space:]')"
echo "source_results=$source_results"
echo "full_results=$full_run/results"

previous_transfer_id="$(grep -Eo '[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}' \
  "$transfer_log" 2>/dev/null | tail -1 | tr 'A-F' 'a-f' || true)"
if tree_matches_manifest "$full_run/results" "$source_file_count"; then
  echo "existing_scratch_sha256_verified=true"
elif [[ -n "$previous_transfer_id" ]]; then
  echo "previous_globus_transfer_id=$previous_transfer_id"
  wait_for_previous_globus_copy "$source_file_count"
elif [[ -e "$full_run/results" ]]; then
  die "an unverified scratch results path already exists: $full_run/results"
else
  copy_staging="$(mktemp -d "$full_run/.results.copying.XXXXXX")"
  echo "Copying CFS results directly into scratch staging..."
  cp -a -- "$source_results"/. "$copy_staging"/
  tree_matches_manifest "$copy_staging" "$source_file_count" || die \
    "scratch staging copy failed SHA-256 verification and was retained at $copy_staging"
  mv -- "$copy_staging" "$full_run/results"
  echo "direct_scratch_copy_sha256_verified=true"
fi

[[ -d "$full_run/results" ]] || die "copy did not create $full_run/results"
echo "Verifying every scratch file against the CFS SHA-256 inventory..."
(
  cd "$full_run/results"
  sha256sum -c "$source_hash_manifest"
)
destination_file_count="$(find "$full_run/results" -type f | wc -l | tr -d '[:space:]')"
[[ "$destination_file_count" == "$source_file_count" ]] || die \
  "scratch tree has $destination_file_count files; expected exactly $source_file_count"

# The module function exists in Perlmutter login shells.
type module >/dev/null 2>&1 || die "environment modules are unavailable; run from a Perlmutter login shell"
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
