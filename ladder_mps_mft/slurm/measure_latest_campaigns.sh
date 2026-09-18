#!/bin/bash
# User-run Perlmutter equal-time measurement backfill. No DMRG or GPU jobs.
set -euo pipefail
measurement_script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
source "$(dirname "$measurement_script")/phase1_gpu.sh"
DIAGNOSTICS_ROOT="${DIAGNOSTICS_ROOT:-$project_dir/output/phase1_diagnostics}"
DIAGNOSTICS_CAMPAIGN_ROOT="${DIAGNOSTICS_CAMPAIGN_ROOT:-$run_root}"
DIAGNOSTICS_TIME_PER_MPS="${DIAGNOSTICS_TIME_PER_MPS:-02:00:00}"
DIAGNOSTICS_MAX_NODE_HOURS="${DIAGNOSTICS_MAX_NODE_HOURS:-9}"
# Four physical cores / eight logical CPUs; 32 GiB memory bills nine cores.
# Same Perlmutter CPU accounting convention as phase0_calibrate_cpu.sh.
readonly DIAGNOSTICS_LOGICAL_CPUS=8 DIAGNOSTICS_MEMORY_MIB=32768
readonly DIAGNOSTICS_PHYSICAL_CORES=9

measurement_reserve() {
  awk -v n="$1" -v s="$(time_to_seconds "$DIAGNOSTICS_TIME_PER_MPS")" \
    'BEGIN {printf "%.9f",n*s/3600*9/128}'
}
measurement_time() {
  local seconds
  seconds=$(( $(time_to_seconds "$DIAGNOSTICS_TIME_PER_MPS") * $1 ))
  printf '%02d:%02d:%02d' "$(( seconds / 3600 ))" "$(( seconds / 60 % 60 ))" "$(( seconds % 60 ))"
}
measurement_plan() {
  [[ "$PHASE1_QOS" == shared ]] || die "measurement accounting requires shared QOS"
  local reserve
  reserve="$(measurement_reserve 58)"
  awk -v n="$reserve" -v cap="$DIAGNOSTICS_MAX_NODE_HOURS" 'BEGIN {exit !(n<=cap)}' || die "measurement ceiling exceeds DIAGNOSTICS_MAX_NODE_HOURS"
  check_reservation "$reserve"
  cat <<EOF
Equal-time MPS measurement backfill: 56 latest branches / 58 spatial MPSs
Sources: square/cubic grids, square fine cuts, square (1.2,+0.2), trellis
CPU jobs: 56; 4 Julia threads, 8 logical CPUs, 32 GiB each; no new DMRG
Wall limit: $DIAGNOSTICS_TIME_PER_MPS per MPS (double for two-ladder trellis)
Requested ceiling: $reserve CPU node-hours; this is not measured runtime.
Project accounting: $(ledger_total) active / $PHASE1_ADDITIONAL_NODE_HOUR_CAP cap
Original states and convergence flags remain unchanged.
EOF
  "$PHASE1_JULIA" --startup-file=no --project="$project_dir" \
    "$project_dir/scripts/measure_latest_campaigns.jl" plan "$DIAGNOSTICS_CAMPAIGN_ROOT"
}

measurement_submit() {
  local run_id="$1" directory index campaign label config configsha compact compactsha source hash fingerprint status accepted iteration samples
  validate_new_run_id "$run_id"
  directory="$DIAGNOSTICS_ROOT/$run_id"
  # Mandatory CPU calibration plan is read-only and submits nothing.
  bash "$project_dir/slurm/phase0_calibrate_cpu.sh" plan >/dev/null
  require_command sbatch
  if [[ ! -d "$directory" ]]; then
    measurement_plan
    "$PHASE1_JULIA" --startup-file=no --project="$project_dir" \
      "$project_dir/scripts/measure_latest_campaigns.jl" prepare "$DIAGNOSTICS_CAMPAIGN_ROOT" "$directory"
    mkdir -p "$directory/logs" "$directory/source"
    cp -a "$project_dir/src" "$project_dir/scripts" "$project_dir/slurm" \
      "$project_dir/Project.toml" "$project_dir/Manifest.toml" "$directory/source/"
    (cd "$directory/source"; find . -type f -print0 | sort -z | xargs -0 sha256sum) >"$directory/source.sha256"
    sha256sum "$directory/manifest.tsv" >"$directory/manifest.sha256"
    printf 'kind\tlabel\tsegment\tpool\trequested_time\treserved_node_hours\tjob_id\tconfig\n' >"$directory/jobs.tsv"
    {
      printf 'PHASE1_LEDGER_PATH=%q\n' "$ledger_path"
      printf 'PHASE1_RECONCILIATION_PATH=%q\n' "$reconciliation_path"
      printf 'PHASE1_BUDGET_ROOT=%q\n' "$budget_root"
      printf 'PHASE1_JULIA=%q\n' "$PHASE1_JULIA"
      printf 'DIAGNOSTICS_TIME_PER_MPS=%q\n' "$DIAGNOSTICS_TIME_PER_MPS"
    } >"$directory/measurement.env"
  fi
  [[ -f "$directory/measurement.env" ]] || die "incomplete preparation; inspect $directory before retrying"
  source "$directory/measurement.env"
  ledger_path="$PHASE1_LEDGER_PATH"
  reconciliation_path="$PHASE1_RECONCILIATION_PATH"
  budget_root="$PHASE1_BUDGET_ROOT"
  sha256sum -c "$directory/manifest.sha256" >/dev/null
  (cd "$directory/source"; sha256sum -c ../source.sha256 >/dev/null)
  lock_path="${ledger_path}.lock"
  acquire_budget_lock
  trap release_budget_lock EXIT
  ensure_ledger
  local total=0 reserve wall raw job_id
  # Check the entire outstanding ceiling before submitting any remaining jobs.
  while IFS=$'\t' read -r index campaign label config configsha compact compactsha source hash fingerprint status accepted iteration samples; do
    [[ "$index" == index ]] && continue
    [[ "$index" =~ ^[0-9]+$ && "$samples" =~ ^[12]$ ]] || die "invalid measurement manifest row"
    awk -F'\t' -v key="$index" 'NR>1 && $2==key {found=1} END {exit !found}' "$directory/jobs.tsv" && continue
    total=$(( total + samples ))
  done <"$directory/manifest.tsv"
  reserve="$(measurement_reserve "$total")"
  awk -v n="$reserve" -v cap="$DIAGNOSTICS_MAX_NODE_HOURS" 'BEGIN {exit !(n<=cap)}' || die "measurement ceiling exceeds cap"
  check_reservation "$reserve"
  while IFS=$'\t' read -r index campaign label config configsha compact compactsha source hash fingerprint status accepted iteration samples; do
    [[ "$index" == index ]] && continue
    awk -F'\t' -v key="$index" 'NR>1 && $2==key {found=1} END {exit !found}' "$directory/jobs.tsv" && continue
    wall="$(measurement_time "$samples")"
    reserve="$(measurement_reserve "$samples")"
    raw="$(sbatch --parsable --account="$PHASE1_EP_ACCOUNT" --constraint=cpu --qos=shared \
      --licenses=scratch,cfs --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32768M \
      --time="$wall" --job-name="lmf-measure-$index" \
      --output="$directory/logs/$index-%j.out" --export=ALL \
      "$directory/source/slurm/measure_latest_campaigns.sh" _run "$directory" "$index")"
    job_id="${raw%%;*}"
    [[ "$job_id" =~ ^[0-9]+$ ]] || die "invalid sbatch response: $raw"
    record_submission "$directory" diagnostics "$index" 1 cpu "$wall" "$reserve" "$job_id" "$directory/manifest.tsv"
    printf 'submitted %s: %s/%s job=%s ceiling=%s node-hours\n' "$index" "$campaign" "$label" "$job_id" "$reserve"
  done <"$directory/manifest.tsv"
  release_budget_lock
  trap - EXIT
  printf 'Measurements: %s/results\n' "$directory"
}

case "${1:-plan}" in
  plan) measurement_plan;;
  submit) measurement_submit "${2:-20260918_latest_correlations}";;
  status)
    directory="$DIAGNOSTICS_ROOT/${2:-20260918_latest_correlations}"
    "$PHASE1_JULIA" --startup-file=no --project="$project_dir" "$project_dir/scripts/measure_latest_campaigns.jl" status "$directory"
    ;;
  reconcile)
    directory="$DIAGNOSTICS_ROOT/${2:-20260918_latest_correlations}"
    source "$directory/measurement.env"
    ledger_path="$PHASE1_LEDGER_PATH"; reconciliation_path="$PHASE1_RECONCILIATION_PATH"
    lock_path="${ledger_path}.lock"
    reconcile_ledger "${2:-20260918_latest_correlations}"
    ;;
  _run)
    directory="$2"; index="$3"
    source "$directory/measurement.env"
    module load julia
    export JULIA_NUM_THREADS=4 JULIA_PKG_PRECOMPILE_AUTO=0
    export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
    sha256sum -c "$directory/manifest.sha256" >/dev/null
    (cd "$directory/source"; sha256sum -c ../source.sha256 >/dev/null)
    srun --ntasks=1 --cpus-per-task=8 --cpu-bind=cores \
      "$PHASE1_JULIA" --startup-file=no --project="$directory/source" \
      "$directory/source/scripts/measure_latest_campaigns.jl" run "$directory/manifest.tsv" "$index"
    ;;
  *) die "usage: bash $0 plan|submit|status|reconcile [RUN_ID]";;
esac
