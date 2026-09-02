#!/bin/bash

# Guarded Perlmutter CPU launcher for the isolated-ladder backbone and Stage 1
# equal-time covariance screen. The default action is read-only `plan`.

set -euo pipefail

readonly SCRIPT_VERSION="1.0.0"
readonly PHYSICAL_CORES_PER_NODE=128
readonly MIB_PER_LOGICAL_CPU=1952

script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
project_dir="${BARE_PROJECT_DIR:-$(cd "$(dirname "$script_path")/.." && pwd)}"
control_root="${BARE_CONTROL_ROOT:-$project_dir/output/bare_stage1}"
full_root="${BARE_FULL_ROOT:-${PSCRATCH:-$project_dir/output}/MPS-MFT/ladder_mps_mft/bare_stage1}"

BARE_ACCOUNT="${BARE_ACCOUNT:-m4863}"
BARE_QOS="${BARE_QOS:-shared}"
BARE_CONFIG="${BARE_CONFIG:-$project_dir/configs/bare_stage1_t014_v0.toml}"
BARE_JULIA="${BARE_JULIA:-julia}"
BARE_JULIA_THREADS="${BARE_JULIA_THREADS:-4}"
BARE_LOGICAL_CPUS="${BARE_LOGICAL_CPUS:-8}"
BARE_SECTOR_MEMORY="${BARE_SECTOR_MEMORY:-48G}"
BARE_SECTOR_TIME="${BARE_SECTOR_TIME:-24:00:00}"
BARE_ASSEMBLE_MEMORY="${BARE_ASSEMBLE_MEMORY:-64G}"
BARE_ASSEMBLE_TIME="${BARE_ASSEMBLE_TIME:-02:00:00}"
BARE_STAGE1_MEMORY="${BARE_STAGE1_MEMORY:-64G}"
BARE_STAGE1_TIME="${BARE_STAGE1_TIME:-16:00:00}"
BARE_MAX_NODE_HOURS="${BARE_MAX_NODE_HOURS:-24.0}"

die() { echo "error: $*" >&2; exit 1; }
require_command() { command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"; }
ceil_div() { echo $(( ($1 + $2 - 1) / $2 )); }

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

reservation_node_hours() {
  local jobs="$1" logical="$2" memory="$3" wall="$4"
  local memory_logical unavailable physical seconds
  memory_logical="$(ceil_div "$(memory_to_mib "$memory")" "$MIB_PER_LOGICAL_CPU")"
  unavailable="$logical"
  (( memory_logical > unavailable )) && unavailable="$memory_logical"
  physical="$(ceil_div "$unavailable" 2)"
  seconds="$(time_to_seconds "$wall")"
  awk -v j="$jobs" -v p="$physical" -v s="$seconds" -v c="$PHYSICAL_CORES_PER_NODE" \
    'BEGIN { printf "%.9f", j*p/c*s/3600.0 }'
}

upper_bound_node_hours() {
  local sectors assemble stage1 total
  sectors="$(reservation_node_hours 6 "$BARE_LOGICAL_CPUS" "$BARE_SECTOR_MEMORY" "$BARE_SECTOR_TIME")"
  assemble="$(reservation_node_hours 1 2 "$BARE_ASSEMBLE_MEMORY" "$BARE_ASSEMBLE_TIME")"
  stage1="$(reservation_node_hours 1 "$BARE_LOGICAL_CPUS" "$BARE_STAGE1_MEMORY" "$BARE_STAGE1_TIME")"
  total="$(awk -v a="$sectors" -v b="$assemble" -v c="$stage1" 'BEGIN {printf "%.9f", a+b+c}')"
  printf '%s\t%s\t%s\t%s\n' "$sectors" "$assemble" "$stage1" "$total"
}

validate() {
  [[ -f "$BARE_CONFIG" ]] || die "missing config: $BARE_CONFIG"
  [[ -f "$project_dir/Project.toml" ]] || die "missing Julia project: $project_dir"
  [[ "$BARE_QOS" == "shared" ]] || die "the pilot is budgeted only for shared QOS"
  (( BARE_JULIA_THREADS >= 1 )) || die "BARE_JULIA_THREADS must be positive"
  (( BARE_LOGICAL_CPUS >= 2 * BARE_JULIA_THREADS )) || die \
    "request at least two Slurm logical CPUs per Julia thread on Perlmutter"
}

print_plan() {
  validate
  local sectors assemble stage1 total
  IFS=$'\t' read -r sectors assemble stage1 total < <(upper_bound_node_hours)
  cat <<EOF
Bare-ladder backbone + Stage 1 CPU pilot

Model config:       $BARE_CONFIG
Sector jobs:        6 independent QN-DMRG array tasks
Per sector:         ${BARE_JULIA_THREADS} Julia threads, ${BARE_LOGICAL_CPUS} Slurm CPUs, ${BARE_SECTOR_MEMORY}, ${BARE_SECTOR_TIME}
DMRG threading:     block-sparse only; BLAS=1, Strided=1
Assembly:           ${BARE_ASSEMBLE_MEMORY}, ${BARE_ASSEMBLE_TIME}
Stage 1:            ${BARE_JULIA_THREADS} Julia threads, ${BARE_STAGE1_MEMORY}, ${BARE_STAGE1_TIME}
Worst-case reserve: sectors=${sectors}, assembly=${assemble}, stage1=${stage1}, total=${total} node-hours
Enforced cap:       ${BARE_MAX_NODE_HOURS} node-hours

The six sectors run independently, so no allocated core waits for another DMRG
sector. Stage 1 starts only after the immutable six-sector backbone is assembled.
The full MPS artifacts live on scratch; a stateless analysis mirror is written to CFS.
EOF
  if awk -v total="$total" -v cap="$BARE_MAX_NODE_HOURS" 'BEGIN {exit !(total > cap)}'; then
    die "worst-case reservation exceeds BARE_MAX_NODE_HOURS"
  fi
}

set_thread_environment() {
  export JULIA_NUM_THREADS="$BARE_JULIA_THREADS"
  export JULIA_PKG_PRECOMPILE_AUTO=0
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  export OMP_PROC_BIND=spread OMP_PLACES=cores
}

initialize_run() {
  local run_id="$1" control_dir="$control_root/$run_id" full_dir="$full_root/$run_id"
  [[ "$run_id" =~ ^[A-Za-z0-9_.-]+$ ]] || die "unsafe run id: $run_id"
  [[ ! -e "$control_dir" ]] || die "control directory already exists: $control_dir"
  [[ ! -e "$full_dir" ]] || die "scratch directory already exists: $full_dir"
  mkdir -p "$control_dir/logs" "$full_dir"
  cp "$BARE_CONFIG" "$control_dir/config.toml"
  cat >"$control_dir/run.env" <<EOF
BARE_RUN_SCRIPT_VERSION='$SCRIPT_VERSION'
BARE_PROJECT_DIR='$project_dir'
BARE_CONTROL_DIR='$control_dir'
BARE_FULL_DIR='$full_dir'
BARE_CONFIG_COPY='$control_dir/config.toml'
BARE_ACCOUNT='$BARE_ACCOUNT'
BARE_QOS='$BARE_QOS'
BARE_JULIA='$BARE_JULIA'
BARE_JULIA_THREADS='$BARE_JULIA_THREADS'
BARE_LOGICAL_CPUS='$BARE_LOGICAL_CPUS'
EOF
  printf 'kind\tlabel\tjob_id\n' >"$control_dir/jobs.tsv"
  printf '%s\n' "$full_dir" >"$control_dir/full_run_path.txt"
  mkdir -p "$control_root"
  printf '%s\n' "$control_dir" >"$control_root/latest_run.txt"
  printf '%s\t%s\n' "$control_dir" "$full_dir"
}

resolve_control_dir() {
  local requested="${1:-}"
  if [[ -n "$requested" && -d "$requested" ]]; then cd "$requested" && pwd; return; fi
  if [[ -n "$requested" && -d "$control_root/$requested" ]]; then cd "$control_root/$requested" && pwd; return; fi
  if [[ -z "$requested" && -f "$control_root/latest_run.txt" ]]; then
    local latest; latest="$(<"$control_root/latest_run.txt")"
    [[ -d "$latest" ]] || die "latest run does not exist: $latest"
    cd "$latest" && pwd; return
  fi
  die "cannot resolve run: ${requested:-latest}"
}

load_run() {
  local control_dir="$1"
  [[ -f "$control_dir/run.env" ]] || die "missing run.env in $control_dir"
  # Generated locally by initialize_run from validated paths and scalar values.
  # shellcheck disable=SC1090
  source "$control_dir/run.env"
  [[ "$BARE_RUN_SCRIPT_VERSION" == "$SCRIPT_VERSION" ]] || die \
    "run script version $BARE_RUN_SCRIPT_VERSION differs from $SCRIPT_VERSION"
  project_dir="$BARE_PROJECT_DIR"
}

submit_run() {
  print_plan
  # Repository policy requires this read-only calibration plan before any CPU submission.
  "$project_dir/slurm/phase0_calibrate_cpu.sh" plan >/dev/null
  require_command sbatch
  local run_id="${1:-$(date -u +%Y%m%dT%H%M%SZ)}" initialized control_dir full_dir
  initialized="$(initialize_run "$run_id")"
  IFS=$'\t' read -r control_dir full_dir <<<"$initialized"
  local sector_raw sector_id assemble_raw assemble_id stage1_raw stage1_id
  sector_raw="$(sbatch --parsable --account="$BARE_ACCOUNT" --constraint=cpu --qos="$BARE_QOS" \
    --licenses=scratch,cfs \
    --nodes=1 --ntasks=1 --cpus-per-task="$BARE_LOGICAL_CPUS" --mem="$BARE_SECTOR_MEMORY" \
    --time="$BARE_SECTOR_TIME" --array=1-6 --job-name=lmf-backbone \
    --output="$control_dir/logs/sector-%A_%a.out" --export=ALL \
    "$script_path" _sector "$control_dir")"
  sector_id="${sector_raw%%;*}"
  printf 'sector_array\tsectors_1_to_6\t%s\n' "$sector_id" >>"$control_dir/jobs.tsv"

  assemble_raw="$(sbatch --parsable --account="$BARE_ACCOUNT" --constraint=cpu --qos="$BARE_QOS" \
    --licenses=scratch,cfs \
    --nodes=1 --ntasks=1 --cpus-per-task=2 --mem="$BARE_ASSEMBLE_MEMORY" \
    --time="$BARE_ASSEMBLE_TIME" --job-name=lmf-backbone-assemble \
    --dependency="afterok:$sector_id" --kill-on-invalid-dep=yes \
    --output="$control_dir/logs/assemble-%j.out" --export=ALL \
    "$script_path" _assemble "$control_dir")"
  assemble_id="${assemble_raw%%;*}"
  printf 'assemble\tbackbone\t%s\n' "$assemble_id" >>"$control_dir/jobs.tsv"

  stage1_raw="$(sbatch --parsable --account="$BARE_ACCOUNT" --constraint=cpu --qos="$BARE_QOS" \
    --licenses=scratch,cfs \
    --nodes=1 --ntasks=1 --cpus-per-task="$BARE_LOGICAL_CPUS" --mem="$BARE_STAGE1_MEMORY" \
    --time="$BARE_STAGE1_TIME" --job-name=lmf-stage1 \
    --dependency="afterok:$assemble_id" --kill-on-invalid-dep=yes \
    --output="$control_dir/logs/stage1-%j.out" --export=ALL \
    "$script_path" _stage1 "$control_dir")"
  stage1_id="${stage1_raw%%;*}"
  printf 'stage1\tcovariance_screen\t%s\n' "$stage1_id" >>"$control_dir/jobs.tsv"
  echo "submitted run=$run_id sectors=$sector_id assembly=$assemble_id stage1=$stage1_id"
  echo "monitor: $script_path status $run_id"
}

sector_worker() {
  local control_dir="$1"
  load_run "$control_dir"
  [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || die "_sector requires a Slurm array task"
  set_thread_environment
  srun --ntasks=1 --cpus-per-task="$BARE_LOGICAL_CPUS" --cpu-bind=cores \
    "$BARE_JULIA" --startup-file=no --threads="$BARE_JULIA_THREADS" --project="$project_dir" \
    "$project_dir/scripts/run_ladder_backbone.jl" "$BARE_CONFIG_COPY" "$BARE_FULL_DIR" \
    "$SLURM_ARRAY_TASK_ID"
}

assemble_worker() {
  local control_dir="$1"
  load_run "$control_dir"
  export JULIA_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  srun --ntasks=1 --cpus-per-task=2 \
    "$BARE_JULIA" --startup-file=no --threads=1 --project="$project_dir" \
    "$project_dir/scripts/run_ladder_backbone.jl" "$BARE_CONFIG_COPY" "$BARE_FULL_DIR" assemble
}

stage1_worker() {
  local control_dir="$1"
  load_run "$control_dir"
  set_thread_environment
  srun --ntasks=1 --cpus-per-task="$BARE_LOGICAL_CPUS" --cpu-bind=cores \
    "$BARE_JULIA" --startup-file=no --threads="$BARE_JULIA_THREADS" --project="$project_dir" \
    "$project_dir/scripts/run_bare_stage1.jl" "$BARE_CONFIG_COPY" \
    "$BARE_FULL_DIR/backbone.h5" "$BARE_FULL_DIR/stage1.h5"
  srun --ntasks=1 --cpus-per-task="$BARE_LOGICAL_CPUS" \
    "$BARE_JULIA" --startup-file=no --threads=1 --project="$project_dir" \
    "$project_dir/scripts/compact_results.jl" "$BARE_FULL_DIR" "$control_dir/stateless_results"
}

status_run() {
  require_command squeue
  local control_dir; control_dir="$(resolve_control_dir "${1:-}")"
  printf '%-16s %-24s %-14s %-16s\n' KIND LABEL JOB_ID STATE
  local kind label job_id state
  while IFS=$'\t' read -r kind label job_id; do
    [[ "$kind" == kind ]] && continue
    state="$(squeue --noheader --jobs="$job_id" --format='%T' 2>/dev/null | head -n1 || true)"
    if [[ -z "$state" ]] && command -v sacct >/dev/null 2>&1; then
      state="$(sacct -X -j "$job_id" --noheader --parsable2 --format=State 2>/dev/null | awk -F'|' 'NF {print $1; exit}' || true)"
    fi
    printf '%-16s %-24s %-14s %-16s\n' "$kind" "$label" "$job_id" "${state:-UNKNOWN}"
  done <"$control_dir/jobs.tsv"
}

show_run() {
  local control_dir; control_dir="$(resolve_control_dir "${1:-}")"
  load_run "$control_dir"
  if [[ -f "$BARE_FULL_DIR/stage1_summary.tsv" ]]; then
    cat "$BARE_FULL_DIR/stage1_summary.tsv"
  else
    status_run "$control_dir"
  fi
}

case "${1:-plan}" in
  plan) print_plan;;
  submit) submit_run "${2:-}";;
  status) status_run "${2:-}";;
  show) show_run "${2:-}";;
  _sector) sector_worker "$2";;
  _assemble) assemble_worker "$2";;
  _stage1) stage1_worker "$2";;
  *) die "usage: $0 {plan|submit [RUN_ID]|status [RUN_ID]|show [RUN_ID]}";;
esac
