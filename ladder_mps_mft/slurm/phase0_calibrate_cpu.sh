#!/bin/bash

# Guarded Perlmutter CPU calibration for the ladder MPS+MF DMRG payload.
# No job is submitted by the default `plan` action.

set -euo pipefail

readonly PHASE0_SCRIPT_VERSION="1.3.1"
readonly MIB_PER_LOGICAL_CPU=1952
readonly PHYSICAL_CORES_PER_NODE=128

script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
project_dir="${PHASE0_PROJECT_DIR:-$(cd "$(dirname "$script_path")/.." && pwd)}"
run_root="${PHASE0_RUN_ROOT:-$project_dir/output/phase0_calibration}"

PHASE0_ACCOUNT="${PHASE0_ACCOUNT:-m4863}"
PHASE0_QOS="${PHASE0_QOS:-shared}"
PHASE0_JULIA="${PHASE0_JULIA:-julia}"
PHASE0_MAX_NODE_HOURS="${PHASE0_MAX_NODE_HOURS:-3.0}"
PHASE0_BENCH_MEMORY="${PHASE0_BENCH_MEMORY:-32G}"
PHASE0_BENCH_TIME="${PHASE0_BENCH_TIME:-04:00:00}"
PHASE0_SEED_MEMORY="${PHASE0_SEED_MEMORY:-8G}"
PHASE0_SEED_TIME="${PHASE0_SEED_TIME:-00:15:00}"
PHASE0_REPORT_TIME="${PHASE0_REPORT_TIME:-00:15:00}"
PHASE0_REPETITIONS="${PHASE0_REPETITIONS:-2}"
PHASE0_SEED_THREADS="${PHASE0_SEED_THREADS:-2}"

die() { echo "error: $*" >&2; exit 1; }
require_command() { command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"; }
ceil_div() { echo $(( ($1 + $2 - 1) / $2 )); }

candidate_rows() {
  printf '%s\t%s\t%s\t%s\n' \
    serial-t1 1 serial 2 \
    blocksparse-t4 4 blocksparse 8
}

memory_to_mib() {
  local raw
  raw="$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')"
  raw="${raw//IB/}"
  raw="${raw//B/}"
  case "$raw" in
    *G) echo $(( ${raw%G} * 1024 ));;
    *M) echo "${raw%M}";;
    *) [[ "$raw" =~ ^[0-9]+$ ]] || die "unsupported memory: $1"; echo "$raw";;
  esac
}

time_to_seconds() {
  local first second third
  IFS=: read -r first second third <<<"$1"
  [[ -n "${third:-}" ]] || { third="$second"; second="$first"; first=0; }
  echo $(( 10#$first * 3600 + 10#$second * 60 + 10#$third ))
}

upper_bound_node_hours() {
  local logical="$1" memory_mib="$2" seconds="$3"
  local memory_logical unavailable physical
  memory_logical="$(ceil_div "$memory_mib" "$MIB_PER_LOGICAL_CPU")"
  unavailable="$logical"
  (( memory_logical > unavailable )) && unavailable="$memory_logical"
  physical="$(ceil_div "$unavailable" 2)"
  awk -v s="$seconds" -v p="$physical" -v c="$PHYSICAL_CORES_PER_NODE" \
    'BEGIN { printf "%.9f", s / 3600.0 * p / c }'
}

phase0_upper_bound() {
  local total=0 memory_mib seconds seed_memory_mib seed_seconds label threads backend logical contribution
  memory_mib="$(memory_to_mib "$PHASE0_BENCH_MEMORY")"
  seconds="$(time_to_seconds "$PHASE0_BENCH_TIME")"
  while IFS=$'\t' read -r label threads backend logical; do
    contribution="$(upper_bound_node_hours "$logical" "$memory_mib" "$seconds")"
    total="$(awk -v a="$total" -v b="$contribution" 'BEGIN { printf "%.9f", a+b }')"
  done < <(candidate_rows)
  seed_memory_mib="$(memory_to_mib "$PHASE0_SEED_MEMORY")"
  seed_seconds="$(time_to_seconds "$PHASE0_SEED_TIME")"
  contribution="$(upper_bound_node_hours $(( 2 * PHASE0_SEED_THREADS )) "$seed_memory_mib" "$seed_seconds")"
  total="$(awk -v a="$total" -v b="$contribution" 'BEGIN { printf "%.9f", a+b }')"
  contribution="$(upper_bound_node_hours 2 2048 "$(time_to_seconds "$PHASE0_REPORT_TIME")")"
  awk -v a="$total" -v b="$contribution" 'BEGIN { printf "%.9f", a+b }'
}

validate_project() {
  [[ -f "$project_dir/Project.toml" ]] || die "missing $project_dir/Project.toml"
  [[ -f "$project_dir/configs/phase0_timing.toml" ]] || die "missing Phase 0 timing config"
  [[ -f "$project_dir/configs/phase0_validation.toml" ]] || die "missing Phase 0 validation config"
  [[ "$PHASE0_QOS" == "shared" ]] || die "Phase 0 is budgeted only for the shared QOS"
  (( PHASE0_REPETITIONS >= 2 )) || die "at least two repetitions are required"
}

print_plan() {
  validate_project
  local bound count
  bound="$(phase0_upper_bound)"
  count="$(candidate_rows | awk 'NF {n++} END {print n+0}')"
  cat <<EOF
Ladder MPS+MF Phase 0 focused CPU calibration

Timing payload:      L=64, chi=200, 6-sweep fixed-mu DMRG, cubic_frustrated
Candidate jobs:      $count (${PHASE0_REPETITIONS} identical DMRG solves each)
Shared seed:         one immutable fixed-mu chi<=64 warm-start MPS and MF-field state
Backends:            serial-t1 and block-sparse-t4, shortlisted by the v2 matrix
Candidate request:   ${PHASE0_BENCH_MEMORY}, ${PHASE0_BENCH_TIME}, shared QOS
Worst-case reserve:  $bound node-hours
Enforced Phase 0 cap: $PHASE0_MAX_NODE_HOURS node-hours

The matrix times only run_dmrg_ground at fixed mu=1.8. MPO construction, compilation, MPS
copying, GC, density measurement, and chemical-potential search are outside the timed region.
Every repetition starts from the same MPS, and candidates must reproduce the serial energy and
density. Use \`submit-seed\` followed by \`submit-matrix\` to inspect the seed before allocating
the two-job comparison.
Run \`bash $script_path submit [RUN_ID]\` to perform the external state change.
EOF
  if awk -v bound="$bound" -v cap="$PHASE0_MAX_NODE_HOURS" 'BEGIN {exit !(bound > cap)}'; then
    die "worst-case reservation exceeds the Phase 0 cap"
  fi
}

write_environment() {
  local path="$1"
  {
    # Use a distinct persisted name: PHASE0_SCRIPT_VERSION is readonly in
    # every worker process and therefore cannot safely be reassigned by
    # sourcing this file.
    printf 'PHASE0_RUN_SCRIPT_VERSION=%q\n' "$PHASE0_SCRIPT_VERSION"
    printf 'PHASE0_PROJECT_DIR=%q\n' "$project_dir"
    printf 'PHASE0_ACCOUNT=%q\n' "$PHASE0_ACCOUNT"
    printf 'PHASE0_QOS=%q\n' "$PHASE0_QOS"
    printf 'PHASE0_JULIA=%q\n' "$PHASE0_JULIA"
    printf 'PHASE0_MAX_NODE_HOURS=%q\n' "$PHASE0_MAX_NODE_HOURS"
    printf 'PHASE0_BENCH_MEMORY=%q\n' "$PHASE0_BENCH_MEMORY"
    printf 'PHASE0_BENCH_TIME=%q\n' "$PHASE0_BENCH_TIME"
    printf 'PHASE0_SEED_MEMORY=%q\n' "$PHASE0_SEED_MEMORY"
    printf 'PHASE0_SEED_TIME=%q\n' "$PHASE0_SEED_TIME"
    printf 'PHASE0_REPORT_TIME=%q\n' "$PHASE0_REPORT_TIME"
    printf 'PHASE0_REPETITIONS=%q\n' "$PHASE0_REPETITIONS"
    printf 'PHASE0_SEED_THREADS=%q\n' "$PHASE0_SEED_THREADS"
  } >"$path"
}

resolve_run_dir() {
  local requested="${1:-}"
  if [[ -n "$requested" && -d "$requested" ]]; then cd "$requested" && pwd; return; fi
  if [[ -n "$requested" && -d "$run_root/$requested" ]]; then cd "$run_root/$requested" && pwd; return; fi
  if [[ -z "$requested" && -f "$run_root/latest_run.txt" ]]; then
    local latest; latest="$(<"$run_root/latest_run.txt")"
    [[ -d "$latest" ]] || die "latest run directory does not exist: $latest"
    cd "$latest" && pwd; return
  fi
  die "cannot resolve Phase 0 run: ${requested:-latest}"
}

load_environment() {
  local run_dir="$1"
  [[ -f "$run_dir/run.env" ]] || die "missing run.env"
  if grep -q '^PHASE0_SCRIPT_VERSION=' "$run_dir/run.env"; then
    die "legacy run.env cannot be loaded safely; preserve this failed run and submit a new run ID"
  fi
  # Generated by write_environment with shell escaping.
  # shellcheck disable=SC1090
  source "$run_dir/run.env"
  local run_script_version="${PHASE0_RUN_SCRIPT_VERSION:-missing}"
  if [[ "$run_script_version" != "$PHASE0_SCRIPT_VERSION" ]]; then
    if [[ "$run_script_version" == "1.3.0" && "$PHASE0_SCRIPT_VERSION" == "1.3.1" ]]; then
      echo "warning: continuing compatible Phase 0 v1.3.0 run with launcher fix v1.3.1" >&2
    else
      die "run.env script version $run_script_version differs from worker version $PHASE0_SCRIPT_VERSION"
    fi
  fi
  project_dir="$PHASE0_PROJECT_DIR"
}

set_thread_environment() {
  local threads="$1" backend="$2"
  export JULIA_NUM_THREADS="$threads"
  export JULIA_PKG_PRECOMPILE_AUTO=0
  export OMP_PROC_BIND=spread
  export OMP_PLACES=cores
  export PHASE0_REPETITIONS
  if [[ "$backend" == "blas" ]]; then
    export OMP_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
  else
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  fi
}

run_payload() {
  local run_dir="$1" label="$2" threads="$3" backend="$4" config="$5" metric="$6" time_file="$7"
  local logical="${SLURM_CPUS_PER_TASK:-$(( 2 * threads ))}"
  set_thread_environment "$threads" "$backend"
  export PHASE0_SEED_STATE="$run_dir/seed_state.h5"
  require_command srun
  require_command /usr/bin/time
  srun --ntasks=1 --cpus-per-task="$logical" --cpu-bind=cores \
    /usr/bin/time -v -o "$time_file" \
    "$PHASE0_JULIA" --startup-file=no --threads="$threads" --project="$project_dir" \
    "$project_dir/scripts/phase0_payload.jl" "$config" "$metric" "$label" "$backend"
}

initialize_phase0_run() {
  local run_id="$1"
  [[ "$run_id" =~ ^[A-Za-z0-9_.-]+$ ]] || die "unsafe run id: $run_id"
  local run_dir="$run_root/$run_id"
  [[ ! -e "$run_dir" ]] || die "run directory already exists: $run_dir"
  mkdir -p "$run_dir/logs" "$run_dir/metrics"
  write_environment "$run_dir/run.env"
  printf 'label\tjulia_threads\tbackend\tslurm_logical_cpus\n' >"$run_dir/candidates.tsv"
  candidate_rows >>"$run_dir/candidates.tsv"
  printf 'kind\tlabel\tbackend\tjulia_threads\tslurm_logical_cpus\tmemory\tjob_id\n' >"$run_dir/jobs.tsv"
  printf '%s\n' "$(phase0_upper_bound)" >"$run_dir/worst_case_node_hours.txt"
  mkdir -p "$run_root"
  printf '%s\n' "$run_dir" >"$run_root/latest_run.txt"
  printf '%s\n' "$run_dir"
}

submit_seed_job() {
  local run_dir="$1"
  local seed_logical seed_raw seed_id
  seed_logical=$(( 2 * PHASE0_SEED_THREADS ))
  seed_raw="$(sbatch --parsable --account="$PHASE0_ACCOUNT" --constraint=cpu --qos="$PHASE0_QOS" \
    --nodes=1 --ntasks=1 --cpus-per-task="$seed_logical" --mem="$PHASE0_SEED_MEMORY" \
    --time="$PHASE0_SEED_TIME" --job-name=lmf0-seed \
    --output="$run_dir/logs/seed-%j.out" --export=ALL \
    "$script_path" _seed "$run_dir")"
  seed_id="${seed_raw%%;*}"
  printf 'seed\tseed\tblocksparse\t%s\t%s\t%s\t%s\n' "$PHASE0_SEED_THREADS" "$seed_logical" "$PHASE0_SEED_MEMORY" "$seed_id" >>"$run_dir/jobs.tsv"
  printf '%s\n' "$seed_id"
}

submit_matrix_jobs() {
  local run_dir="$1" seed_id="$2" seed_state="${3:-pending}"
  if awk -F'\t' '$1 == "benchmark" || ($1 == "report" && $2 == "report") {found=1} END {exit !found}' "$run_dir/jobs.tsv"; then
    die "benchmark matrix or primary report already recorded in $run_dir/jobs.tsv"
  fi
  local -a seed_dependency_args=()
  case "$seed_state" in
    pending) seed_dependency_args=(--dependency="afterok:$seed_id" --kill-on-invalid-dep=yes);;
    completed) ;;
    *) die "unknown seed state for matrix submission: $seed_state";;
  esac
  local -a job_ids=()
  local label threads backend logical raw job_id
  while IFS=$'\t' read -r label threads backend logical; do
    raw="$(sbatch --parsable --account="$PHASE0_ACCOUNT" --constraint=cpu --qos="$PHASE0_QOS" \
      --nodes=1 --ntasks=1 --cpus-per-task="$logical" --mem="$PHASE0_BENCH_MEMORY" \
      --time="$PHASE0_BENCH_TIME" --job-name="lmf0-$label" \
      ${seed_dependency_args[@]+"${seed_dependency_args[@]}"} \
      --output="$run_dir/logs/${label}-%j.out" --export=ALL \
      "$script_path" _bench "$run_dir" "$label" "$threads" "$backend")"
    job_id="${raw%%;*}"
    job_ids+=("$job_id")
    printf 'benchmark\t%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$backend" "$threads" "$logical" "$PHASE0_BENCH_MEMORY" "$job_id" >>"$run_dir/jobs.tsv"
  done < <(candidate_rows)

  local dependency report_raw report_id
  dependency="$(IFS=:; echo "${job_ids[*]}")"
  report_raw="$(sbatch --parsable --account="$PHASE0_ACCOUNT" --constraint=cpu --qos="$PHASE0_QOS" \
    --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=2G --time="$PHASE0_REPORT_TIME" \
    --job-name=lmf0-report --dependency="afterany:$dependency" --kill-on-invalid-dep=yes \
    --output="$run_dir/logs/report-%j.out" --export=ALL "$script_path" _report "$run_dir")"
  report_id="${report_raw%%;*}"
  printf 'report\treport\tserial\t1\t2\t2G\t%s\n' "$report_id" >>"$run_dir/jobs.tsv"
  echo "Submitted matrix ${job_ids[*]} and report $report_id after seed $seed_id"
}

submit_phase0() {
  print_plan
  require_command sbatch
  local run_id="${1:-$(date -u +%Y%m%dT%H%M%SZ)}" run_dir seed_id
  run_dir="$(initialize_phase0_run "$run_id")"
  seed_id="$(submit_seed_job "$run_dir")"
  submit_matrix_jobs "$run_dir" "$seed_id" pending
  echo "Submitted Phase 0 run $run_id with seed $seed_id"
  echo "Monitor: bash $script_path status $run_id"
}

submit_seed_only() {
  print_plan
  require_command sbatch
  local run_id="${1:-$(date -u +%Y%m%dT%H%M%SZ)}" run_dir seed_id
  run_dir="$(initialize_phase0_run "$run_id")"
  seed_id="$(submit_seed_job "$run_dir")"
  echo "Submitted seed-only preflight $seed_id for Phase 0 run $run_id"
  echo "Monitor: bash $script_path status $run_id"
  echo "Inspect after completion: bash $script_path show-seed $run_id"
}

submit_matrix_phase0() {
  require_command sbatch
  require_command sacct
  local run_dir seed_id state
  run_dir="$(resolve_run_dir "${1:-}")"
  load_environment "$run_dir"
  print_plan
  seed_id="$(awk -F'\t' '$1 == "seed" {print $7; exit}' "$run_dir/jobs.tsv")"
  [[ -n "$seed_id" ]] || die "no seed job recorded in $run_dir/jobs.tsv"
  state="$(sacct -X -j "$seed_id" --noheader --parsable2 --format=State | awk -F'|' 'NF {print $1; exit}')"
  [[ "$state" == COMPLETED* ]] || die "seed job $seed_id is not complete (state: ${state:-unknown})"
  [[ -f "$run_dir/seed_state.h5" ]] || die "completed seed job did not produce $run_dir/seed_state.h5"
  # The artifact and successful terminal state have already been verified.
  # Do not attach afterok here: Slurm may reject dependencies on completed jobs
  # after they age out of the controller's active record.
  submit_matrix_jobs "$run_dir" "$seed_id" completed
  echo "Monitor: bash $script_path status $(basename "$run_dir")"
}

benchmark_worker() {
  local run_dir="$1" label="$2" threads="$3" backend="$4"
  load_environment "$run_dir"
  [[ -n "${SLURM_JOB_ID:-}" ]] || die "_bench must run in Slurm"
  run_payload "$run_dir" "$label" "$threads" "$backend" \
    "$project_dir/configs/phase0_validation.toml" "$run_dir/metrics/$label.toml" "$run_dir/metrics/$label.time"
}

seed_worker() {
  local run_dir="$1"
  load_environment "$run_dir"
  [[ -n "${SLURM_JOB_ID:-}" ]] || die "_seed must run in Slurm"
  set_thread_environment "$PHASE0_SEED_THREADS" blocksparse
  local logical="${SLURM_CPUS_PER_TASK:-$(( 2 * PHASE0_SEED_THREADS ))}"
  srun --ntasks=1 --cpus-per-task="$logical" --cpu-bind=cores \
    /usr/bin/time -v -o "$run_dir/metrics/seed.time" \
    "$PHASE0_JULIA" --startup-file=no --threads="$PHASE0_SEED_THREADS" --project="$project_dir" \
    "$project_dir/scripts/phase0_prepare_seed.jl" "$project_dir/configs/phase0_validation.toml" "$run_dir/seed_state.h5"
}

report_worker() {
  local run_dir="$1"
  load_environment "$run_dir"
  srun --ntasks=1 --cpus-per-task=2 "$PHASE0_JULIA" --startup-file=no --threads=1 \
    --project="$project_dir" "$project_dir/scripts/phase0_report.jl" "$run_dir"
}

status_phase0() {
  local run_dir; run_dir="$(resolve_run_dir "${1:-}")"
  printf '%-12s %-22s %-12s %-16s\n' KIND LABEL JOB_ID STATE
  local kind label backend threads logical memory job_id state
  while IFS=$'\t' read -r kind label backend threads logical memory job_id; do
    [[ "$kind" == kind ]] && continue
    state="$(squeue --noheader --jobs="$job_id" --format='%T' 2>/dev/null | head -n1 || true)"
    if [[ -z "$state" ]]; then
      state="$(sacct -X -j "$job_id" --noheader --parsable2 --format=State 2>/dev/null | awk -F'|' 'NF {print $1; exit}' || true)"
    fi
    printf '%-12s %-22s %-12s %-16s\n' "$kind" "$label" "$job_id" "${state:-UNKNOWN}"
  done <"$run_dir/jobs.tsv"
}

show_seed_phase0() {
  local run_dir seed_id log_path
  run_dir="$(resolve_run_dir "${1:-}")"
  seed_id="$(awk -F'\t' '$1 == "seed" {print $7; exit}' "$run_dir/jobs.tsv")"
  [[ -n "$seed_id" ]] || die "no seed job recorded in $run_dir/jobs.tsv"
  log_path="$run_dir/logs/seed-$seed_id.out"
  [[ -f "$log_path" ]] || die "missing seed log: $log_path"
  cat "$log_path"
  if [[ -f "$run_dir/seed_state.h5" ]]; then
    require_command sha256sum
    sha256sum "$run_dir/seed_state.h5"
  fi
}

show_phase0() {
  local run_dir; run_dir="$(resolve_run_dir "${1:-}")"
  [[ -f "$run_dir/recommendation.md" ]] || die "no recommendation yet; inspect status/logs"
  cat "$run_dir/recommendation.md"
}

manual_report() {
  local run_dir; run_dir="$(resolve_run_dir "${1:-}")"
  load_environment "$run_dir"
  "$PHASE0_JULIA" --startup-file=no --threads=1 --project="$project_dir" \
    "$project_dir/scripts/phase0_report.jl" "$run_dir"
}

case "${1:-plan}" in
  plan) print_plan;;
  submit) submit_phase0 "${2:-}";;
  submit-seed) submit_seed_only "${2:-}";;
  submit-matrix) submit_matrix_phase0 "${2:-}";;
  status) status_phase0 "${2:-}";;
  show-seed) show_seed_phase0 "${2:-}";;
  show) show_phase0 "${2:-}";;
  report) manual_report "${2:-}";;
  _bench) benchmark_worker "$2" "$3" "$4" "$5";;
  _seed) seed_worker "$2";;
  _report) report_worker "$2";;
  *) die "usage: $0 {plan|submit [RUN_ID]|submit-seed [RUN_ID]|submit-matrix [RUN_ID]|status [RUN_ID]|show-seed [RUN_ID]|show [RUN_ID]|report [RUN_ID]}";;
esac
