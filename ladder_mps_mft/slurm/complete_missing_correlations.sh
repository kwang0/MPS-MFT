#!/bin/bash
# User-run, selective recovery of the seven September 21 measurement timeouts.
set -euo pipefail
(( $# <= 3 )) || { echo "usage: bash $0 plan|submit|status|reconcile [PARENT_RUN [NEW_RUN]]" >&2; exit 1; }
action="${1:-plan}"
case "$action" in plan|submit|status|reconcile) ;; *) echo "expected plan, submit, status or reconcile" >&2; exit 1;; esac
parent_run="${2:-20260921_latest_correlations_retry1}"
new_run="${3:-20260921_latest_correlations_retry2}"
slurm_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PHASE1_PROJECT_DIR="$(cd "$slurm_directory/.." && pwd)"
source "$slurm_directory/phase1_gpu.sh"
validate_new_run_id "$parent_run"
validate_new_run_id "$new_run"
[[ "$parent_run" != . && "$parent_run" != .. && "$new_run" != . && "$new_run" != .. && "$parent_run" != "$new_run" ]] || die "retry needs distinct campaign IDs"
export DIAGNOSTICS_ROOT="${DIAGNOSTICS_ROOT:-$project_dir/output/phase1_diagnostics}"
parent="$DIAGNOSTICS_ROOT/$parent_run"
directory="$DIAGNOSTICS_ROOT/$new_run"
launcher="$slurm_directory/measure_latest_campaigns.sh"
[[ -f "$parent/measurement.env" && -f "$parent/jobs.tsv" ]] || die "missing parent measurement campaign: $parent"
source "$parent/measurement.env"
export PHASE1_LEDGER_PATH PHASE1_RECONCILIATION_PATH PHASE1_BUDGET_ROOT PHASE1_JULIA PHASE1_ADDITIONAL_NODE_HOUR_CAP
ledger_path="$PHASE1_LEDGER_PATH"; reconciliation_path="$PHASE1_RECONCILIATION_PATH"
budget_root="$PHASE1_BUDGET_ROOT"; lock_path="${ledger_path}.lock"
# The old retry's expected manifest describes all 56 rows, not this subset.
export DIAGNOSTICS_EXPECTED_MANIFEST=''
export DIAGNOSTICS_TIME_PER_MPS=04:00:00
sha256sum -c "$parent/manifest.sha256" >/dev/null
(cd "$parent/source"; sha256sum -c ../source.sha256 >/dev/null)
readonly selected='6 11 31 34 46 51 53'
awk -F'\t' 'NR>1 {if (NF!=14 || $1!=NR-1) exit 1; n++; m+=$14} END {if (n!=56 || m!=58) exit 1}' \
  "$parent/manifest.tsv" || die "expected the 56-branch / 58-MPS parent manifest"
awk -F'\t' 'NR>1 {if ($1!="diagnostics" || $3!=1 || $4!="cpu" || $2<1 || $2>56 || $7!~/^[0-9]+$/ || seen[$2]++ || jobs[$7]++) exit 1; n++} END {if (n!=56) exit 1}' \
  "$parent/jobs.tsv" || die "expected one parent submission per manifest row"

retry_manifest() {
  awk -F'\t' -v OFS='\t' -v selected="$selected" '
    BEGIN {split(selected,a," "); for (i in a) wanted[a[i]]=1}
    NR==1 {print; next}
    $1 in wanted {if ($14!=1) exit 1; $1=++n; print}
    END {if (n!=7) exit 1}' "$parent/manifest.tsv"
}
retry_rows() {
  awk -F'\t' -v OFS='\t' -v selected="$selected" '
    BEGIN {split(selected,a," "); for (i in a) wanted[a[i]]=1;
      print "index","parent_index","parent_job_id","campaign","label"}
    NR==FNR {if (FNR>1) jobs[$2]=$7; next}
    FNR>1 && $1 in wanted {print ++n,$1,jobs[$1],$2,$3}' "$parent/jobs.tsv" "$parent/manifest.tsv"
}
verify_retry() {
  [[ -f "$directory/retry.sha256" ]] || die "incomplete retry preparation; inspect $directory"
  sha256sum -c "$directory/retry.sha256" >/dev/null
  sha256sum -c "$directory/manifest.sha256" >/dev/null
  (cd "$directory/source"; sha256sum -c ../source.sha256 >/dev/null)
  cmp -s <(retry_manifest) "$directory/manifest.tsv" || die "retry manifest differs from the seven pinned parent rows"
  cmp -s <(retry_rows) "$directory/retry_rows.tsv" || die "retry parent jobs differ"
  cmp -s "$parent/source.sha256" "$directory/source.sha256" || die "retry scientific source differs from parent"
}

if [[ "$action" == reconcile ]]; then
  verify_retry
  exec bash "$launcher" reconcile "$new_run"
fi

# Reuse the existing receipt + diagnostic-SHA check; Slurm completion alone
# does not establish that a usable diagnostic file was written.
parent_status="$(bash "$launcher" status "$parent_run")"
if [[ "$action" == status ]]; then
  verify_retry
  printf 'Original campaign (successful files stay here):\n%s\n' "$parent_status"
  retry_status="$(bash "$launcher" status "$new_run")"
  printf '\nSeven-entry retry (new indices; see retry_rows.tsv):\n%s\n' "$retry_status"
  # Match by branch identity rather than adding totals, to avoid double counting.
  printf '%s\n%s\n' "$parent_status" "$retry_status" | awk -F'\t' '
    $2=="MEASURED" {done[$3]=1}
    END {for (branch in done) n++; printf "combined_complete=%d/56 branches\n",n}'
  exit 0
fi
awk -F'\t' -v selected="$selected" '
  BEGIN {split(selected,a," "); for (i in a) wanted[a[i]]=1}
  NF==3 {if ($1!=++n || $2!=(($1 in wanted)?"MISSING":"MEASURED")) bad=1}
  END {exit (bad || n!=56)}' <<<"$parent_status" || die "parent no longer has exactly the seven reported missing rows; inspect status before submitting"
printf 'Selective measurement retry: %s -> %s\n' "$parent_run" "$new_run"
retry_rows
printf '7 CPU jobs, four hours each; requested ceiling 1.968750000 CPU node-hours.\n'
printf '49 completed branches stay in the parent; square A/B jobs are outside this retry.\n'
[[ "$PHASE1_QOS" == shared ]] || die "measurement accounting requires shared QOS"
if [[ "$action" == plan ]]; then
  [[ ! -e "$directory" ]] || verify_retry
  check_reservation 1.968750000
  printf 'Submission requires reconciled TIMEOUT accounting for these seven parent jobs.\n'
  exit 0
fi

# Only the parent measurement campaign is reconciled. Other campaigns retain
# their reservations in the shared budget, including ongoing square A/B jobs.
bash "$project_dir/slurm/phase0_calibrate_cpu.sh" plan >/dev/null
require_command sbatch
require_command sacct
bash "$launcher" reconcile "$parent_run"
while IFS=$'\t' read -r index parent_index parent_job campaign label; do
  [[ "$index" == index ]] && continue
  awk -F'\t' -v campaign="$parent_run" -v job="$parent_job" -v row="$parent_index" '
    NR>1 && $2==campaign && $3=="diagnostics" && $4==row && $7==job && $14=="TIMEOUT" {found=1}
    END {exit !found}' "$reconciliation_path" || die "parent job $parent_job lacks reconciled TIMEOUT accounting; no retry submitted"
done < <(retry_rows)
if [[ ! -e "$directory" ]]; then
  check_reservation 1.968750000
  # Pin the existing manifest, never rediscover newer SCF endpoints. Verify
  # full-state presence now; the frozen worker checks the full SHA before use.
  while IFS=$'\t' read -r index campaign label config configsha compact compactsha state hash fingerprint status accepted iteration samples; do
    [[ "$index" == index ]] && continue
    [[ -f "$state" ]] || die "full MPS is unavailable: $state"
    printf '%s  %s\n' "$configsha" "$config" | sha256sum -c - >/dev/null
  done < <(retry_manifest)
  mkdir "$directory"
  retry_manifest >"$directory/manifest.tsv"
  retry_rows >"$directory/retry_rows.tsv"
  mkdir "$directory/logs"
  cp -a "$parent/source" "$directory/source"
  cp "$parent/source.sha256" "$directory/source.sha256"
  sha256sum "$directory/manifest.tsv" >"$directory/manifest.sha256"
  printf 'kind\tlabel\tsegment\tpool\trequested_time\treserved_node_hours\tjob_id\tconfig\n' >"$directory/jobs.tsv"
  {
    printf 'PHASE1_LEDGER_PATH=%q\n' "$ledger_path"
    printf 'PHASE1_RECONCILIATION_PATH=%q\n' "$reconciliation_path"
    printf 'PHASE1_BUDGET_ROOT=%q\n' "$budget_root"
    printf 'PHASE1_JULIA=%q\n' "$PHASE1_JULIA"
    printf 'DIAGNOSTICS_TIME_PER_MPS=04:00:00\nDIAGNOSTICS_EXPECTED_MANIFEST=\n'
  } >"$directory/measurement.env"
  sha256sum "$parent/manifest.tsv" "$parent/jobs.tsv" "$directory/retry_rows.tsv" \
    "$directory/measurement.env" >"$directory/retry.sha256"
fi
verify_retry
# Reuse the existing budget lock, resource requests and duplicate-job guard.
exec bash "$launcher" submit "$new_run"
