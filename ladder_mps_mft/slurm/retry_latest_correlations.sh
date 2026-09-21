#!/bin/bash
# User-run recovery of the September 18 measurement startup failure.
# Keeps the original frozen campaign and ongoing SCF jobs unchanged.
set -euo pipefail
(( $# <= 3 )) || { echo "usage: bash $0 plan|submit [OLD_RUN_ID [NEW_RUN_ID]]" >&2; exit 1; }
action="${1:-plan}"
case "$action" in plan|submit) ;; *) echo "expected plan or submit" >&2; exit 1;; esac
old_run="${2:-20260918_latest_correlations}"
new_run="${3:-20260921_latest_correlations_retry1}"
retry_slurm_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PHASE1_PROJECT_DIR="$(cd "$retry_slurm_dir/.." && pwd)"
source "$retry_slurm_dir/phase1_gpu.sh"
validate_new_run_id "$old_run"
validate_new_run_id "$new_run"
[[ "$old_run" != . && "$old_run" != .. && "$new_run" != . && "$new_run" != .. && "$old_run" != "$new_run" ]] || die "retry needs distinct campaign IDs"
export DIAGNOSTICS_ROOT="${DIAGNOSTICS_ROOT:-$project_dir/output/phase1_diagnostics}"
parent="$DIAGNOSTICS_ROOT/$old_run"
launcher="$retry_slurm_dir/measure_latest_campaigns.sh"
[[ -f "$parent/measurement.env" && -f "$parent/jobs.tsv" ]] || die "missing submitted measurement campaign: $parent"
source "$parent/measurement.env"
export PHASE1_LEDGER_PATH PHASE1_RECONCILIATION_PATH PHASE1_BUDGET_ROOT PHASE1_JULIA DIAGNOSTICS_TIME_PER_MPS
export PHASE1_ADDITIONAL_NODE_HOUR_CAP
export DIAGNOSTICS_CAMPAIGN_ROOT="${DIAGNOSTICS_CAMPAIGN_ROOT:-$project_dir/output/phase1_gpu}"
export DIAGNOSTICS_EXPECTED_MANIFEST="$parent/manifest.tsv"
sha256sum -c "$parent/manifest.sha256" >/dev/null
(cd "$parent/source"; sha256sum -c ../source.sha256 >/dev/null)

# This is specifically a complete retry of the known shell-startup failure,
# not a general retry of partial measurements or a running campaign.
[[ ! -d "$parent/results" ]] || die "parent has measurement outputs; inspect them before preparing a full retry"
awk -F'\t' 'NR>1 {if ($1!=NR-1) exit 1; n++; m+=$14} END {if (n!=56 || m!=58) exit 1}' \
  "$parent/manifest.tsv" || die "expected the original 56-branch / 58-MPS manifest"
awk -F'\t' 'NR>1 {if ($1!="diagnostics" || $3!=1 || $4!="cpu" || $2<1 || $2>56 || seen[$2]++ || jobs[$7]++) exit 1; n++} END {if (n!=56) exit 1}' \
  "$parent/jobs.tsv" || die "expected exactly 56 original measurement submissions"
while IFS=$'\t' read -r kind label segment pool wall reserved job_id config; do
  [[ "$kind" == kind ]] && continue
  [[ "$job_id" =~ ^[0-9]+$ ]] || die "invalid original job ID"
  log="$parent/logs/$label-$job_id.out"
  expected="/var/spool/slurmd/job$job_id/slurm_script: line 5: /var/spool/slurmd/job$job_id/phase1_gpu.sh: No such file or directory"
  [[ -f "$log" && "$(cat "$log")" == "$expected" ]] || die "job $job_id does not match the verified startup failure"
done <"$parent/jobs.tsv"

printf 'Measurement retry: %s -> %s\n' "$old_run" "$new_run"
printf 'Same 56 branches / 58 MPSs; new frozen source and job records. Square A/B is excluded.\n'
if [[ "$action" == plan ]]; then
  printf 'Submission will reconcile only the failed measurement campaign and require all its jobs terminal.\n'
  printf 'The new manifest must match the original byte for byte before any submission.\n'
  exec bash "$launcher" plan
fi

bash "$launcher" reconcile "$old_run"
# Reconciliation can succeed while retaining unfinished or missing records.
# Require a terminal record for every original job before retrying it.
while IFS=$'\t' read -r kind label segment pool wall reserved job_id config; do
  [[ "$kind" == kind ]] && continue
  awk -F'\t' -v campaign="$old_run" -v job="$job_id" \
    'NR>1 && $2==campaign && $3=="diagnostics" && $7==job && $14 ~ /^(FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED)$/ {found=1} END {exit !found}' \
    "$PHASE1_RECONCILIATION_PATH" || die "job $job_id lacks reconciled failure accounting; no retry submitted"
done <"$parent/jobs.tsv"
exec bash "$launcher" submit "$new_run"
