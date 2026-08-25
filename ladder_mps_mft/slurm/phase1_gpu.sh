#!/bin/bash

# Guarded Perlmutter GPU launcher for refactored Phase 1 MPS+MF branches.
# The default action is read-only. Scientific jobs are staged after a GPU smoke test.

set -euo pipefail

readonly PHASE1_SCRIPT_VERSION="1.4.0"
script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
project_dir="${PHASE1_PROJECT_DIR:-$(cd "$(dirname "$script_path")/.." && pwd)}"
repo_root="${PHASE1_REPO_ROOT:-$(cd "$project_dir/.." && pwd)}"
run_root="${PHASE1_RUN_ROOT:-$project_dir/output/phase1_gpu}"
scratch_root="${PHASE1_SCRATCH_ROOT:-}"
budget_root="${PHASE1_BUDGET_ROOT:-$project_dir/output/project_budget}"
ledger_path="${PHASE1_LEDGER_PATH:-$budget_root/additional_node_hours.tsv}"
lock_path="${ledger_path}.lock"

PHASE1_ACCOUNT="${PHASE1_ACCOUNT:-m4863_g}"
PHASE1_EP_ACCOUNT="${PHASE1_EP_ACCOUNT:-m4863}"
PHASE1_QOS="${PHASE1_QOS:-shared}"
PHASE1_JULIA="${PHASE1_JULIA:-julia}"
PHASE1_GPU_TIME="${PHASE1_GPU_TIME:-12:00:00}"
PHASE1_GPU_CPUS="${PHASE1_GPU_CPUS:-32}"
PHASE1_SMOKE_TIME="${PHASE1_SMOKE_TIME:-00:30:00}"
PHASE1_MAX_SEGMENTS="${PHASE1_MAX_SEGMENTS:-4}"
PHASE1_ADDITIONAL_NODE_HOUR_CAP="${PHASE1_ADDITIONAL_NODE_HOUR_CAP:-400}"
PHASE1_DECLARED_ALLOCATION_NODE_HOURS="${PHASE1_DECLARED_ALLOCATION_NODE_HOURS:-1000}"
PHASE1_DECLARED_USED_NODE_HOURS="${PHASE1_DECLARED_USED_NODE_HOURS:-277}"
PHASE1_BASE_CONFIG="${PHASE1_BASE_CONFIG:-$project_dir/configs/phase1_gpu_base.toml}"
PHASE1_RECURRENCE_CONFIG="${PHASE1_RECURRENCE_CONFIG:-$project_dir/configs/phase1_gpu_recurrence_chi400.toml}"

die() { echo "error: $*" >&2; exit 1; }
require_command() { command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"; }

resolve_scratch_root() {
  if [[ -z "$scratch_root" ]]; then
    local scratch_base="${PSCRATCH:-${SCRATCH:-}}"
    [[ -n "$scratch_base" ]] || die \
      "Perlmutter scratch is unavailable; set PSCRATCH/SCRATCH or PHASE1_SCRATCH_ROOT"
    scratch_root="$scratch_base/MPS-MFT/ladder_mps_mft/phase1_gpu"
  fi
  [[ "$scratch_root" == /* ]] || die "PHASE1_SCRATCH_ROOT must be an absolute path"
}

time_to_seconds() {
  local first second third
  IFS=: read -r first second third <<<"$1"
  [[ -n "${third:-}" ]] || { third="$second"; second="$first"; first=0; }
  echo $(( 10#$first * 3600 + 10#$second * 60 + 10#$third ))
}

gpu_node_hours() {
  awk -v seconds="$(time_to_seconds "$1")" 'BEGIN {printf "%.9f", seconds / 3600.0 / 4.0}'
}

ep_node_hours() {
  # submit_E_p_ladder.sh requests 64 Slurm CPUs = 32 of 128 physical CPU cores.
  awk 'BEGIN {printf "%.9f", 48.0 / 4.0}'
}

ledger_total() {
  [[ -f "$ledger_path" ]] || { printf '0.000000000\n'; return; }
  awk -F'\t' 'NR > 1 {sum += $7} END {printf "%.9f\n", sum + 0}' "$ledger_path"
}

validate_project() {
  [[ "$PHASE1_QOS" == "shared" ]] || die "the budget model is valid only for shared QOS"
  [[ -f "$project_dir/Project.toml" ]] || die "missing refactored Julia project"
  [[ -f "$project_dir/gpu/Project.toml" ]] || die "missing GPU overlay environment"
  [[ -f "$PHASE1_BASE_CONFIG" ]] || die "missing Phase 1 base config: $PHASE1_BASE_CONFIG"
  [[ -f "$PHASE1_RECURRENCE_CONFIG" ]] || die \
    "missing Phase 1 recurrence config: $PHASE1_RECURRENCE_CONFIG"
  [[ -f "$project_dir/scripts/run_scf_gpu.jl" ]] || die "missing GPU SCF entry point"
  [[ -f "$project_dir/scripts/prepare_phase1_recurrence.jl" ]] || die \
    "missing recurrence preparation entry point"
  [[ -f "$project_dir/scripts/gpu_smoke.jl" ]] || die "missing GPU smoke test"
  [[ -f "$project_dir/scripts/validate_gpu_smoke.jl" ]] || die "missing GPU smoke validator"
  [[ -f "$project_dir/scripts/compact_results.jl" ]] || die "missing stateless-result compactor"
  (( PHASE1_MAX_SEGMENTS >= 1 )) || die "PHASE1_MAX_SEGMENTS must be positive"
}

print_recurrence_plan() {
  validate_project
  local segment smoke targeted maximum
  segment="$(gpu_node_hours "$PHASE1_GPU_TIME")"
  smoke="$(gpu_node_hours "$PHASE1_SMOKE_TIME")"
  targeted="$(awk -v s="$segment" -v p="$smoke" 'BEGIN {printf "%.9f", 3*s+p}')"
  maximum="$(awk -v s="$segment" -v p="$smoke" -v m="$PHASE1_MAX_SEGMENTS" \
    'BEGIN {printf "%.9f", 3*m*s+p}')"
  cat <<EOF
Ladder MPS+MF targeted unfrustrated-pairing recurrence campaign

Representative point: L=64, U=8, V=-0.2, t0=1.1, t_perp=0.1, density=0.9375
Numerical control:    chi=400, 16 sweeps, cutoff=1e-11, energy_tol=1e-9
Branches:             v3 orbit phases 001 and 002 plus independent pairing seed s2 (3 jobs)
Physics controls:     20-update unmixed period-1/2 probe; stop before Anderson acceleration
Parent contract:      full v3 scratch state plus SHA-256 and explicit orbit-member index
GPU request:          one of four GPUs, ${PHASE1_GPU_TIME}, ${PHASE1_GPU_CPUS} CPUs, shared QOS
Per-segment reserve:  ${segment} GPU node-hours
Smoke-test reserve:   ${smoke} GPU node-hours
Targeted staged total: ${targeted} node-hours
Four-segment ceiling: ${maximum} node-hours for these three branches
Hard project cap:     ${PHASE1_ADDITIONAL_NODE_HOUR_CAP} additional node-hours

Preparation does not submit or reserve:
  bash $script_path prepare-recurrence 20260824_phase1_gpu_v3_float64_history RUN_ID

Submission remains explicitly staged:
  bash $script_path submit RUN_ID
  bash $script_path status RUN_ID
  bash $script_path submit-matrix RUN_ID
EOF
  print_budget
  awk -v current="$(ledger_total)" -v requested="$targeted" -v cap="$PHASE1_ADDITIONAL_NODE_HOUR_CAP" \
    'BEGIN {exit !((current+requested) > cap)}' && die "targeted recurrence campaign would exceed the hard cap"
  return 0
}

print_budget() {
  local committed remaining declared_remaining
  committed="$(ledger_total)"
  remaining="$(awk -v cap="$PHASE1_ADDITIONAL_NODE_HOUR_CAP" -v used="$committed" 'BEGIN {printf "%.9f", cap-used}')"
  declared_remaining="$(awk -v total="$PHASE1_DECLARED_ALLOCATION_NODE_HOURS" -v used="$PHASE1_DECLARED_USED_NODE_HOURS" 'BEGIN {printf "%.3f", total-used}')"
  cat <<EOF
User-reported allocation snapshot: ${PHASE1_DECLARED_ALLOCATION_NODE_HOURS} total, ${PHASE1_DECLARED_USED_NODE_HOURS} used, ${declared_remaining} remaining
Project additional hard cap:      ${PHASE1_ADDITIONAL_NODE_HOUR_CAP} node-hours
Conservatively reserved by ledger: ${committed} node-hours
Unreserved project allowance:      ${remaining} node-hours
Ledger:                            ${ledger_path}

The hard cap counts requested upper bounds, not actual elapsed charge, and never
reclaims early completion. CPU and GPU charges are summed here as a conservative
project control even though NERSC accounts them in separate allocation pools.
EOF
}

print_plan() {
  validate_project
  local segment smoke initial maximum
  segment="$(gpu_node_hours "$PHASE1_GPU_TIME")"
  smoke="$(gpu_node_hours "$PHASE1_SMOKE_TIME")"
  initial="$(awk -v s="$segment" -v p="$smoke" 'BEGIN {printf "%.9f", 9*s+p}')"
  maximum="$(awk -v s="$segment" -v p="$smoke" -v m="$PHASE1_MAX_SEGMENTS" 'BEGIN {printf "%.9f", 9*m*s+p}')"
  cat <<EOF
Ladder MPS+MF Phase 1 refactored GPU campaign

Representative point: L=64, U=8, V=-0.2, t0=1.1, t_perp=0.1, density=0.9375, chi=200
Branches:             pairing, SDW, and CDW seeds for all three transverse geometries (9 jobs)
Physics controls:     unmixed raw-map cycle probe, then Anderson mixing if needed
Tensor backend:       dense Float64 CUDA, with S_z and fermion-parity QNs explicitly disabled
CUDA runtime:         pinned CUDA.jl artifacts only; system toolkit libraries are rejected
E_p policy:           exact lookup first; bracketed linear interpolation in t0; no extrapolation
GPU request:          one of four GPUs, ${PHASE1_GPU_TIME}, ${PHASE1_GPU_CPUS} CPUs, shared QOS
Full-result storage:  \$PSCRATCH/MPS-MFT/ladder_mps_mft/phase1_gpu/RUN_ID
CFS/local storage:    MPS-free analysis mirrors under output/phase1_gpu/RUN_ID/results
Per-segment reserve:  ${segment} GPU node-hours
Smoke-test reserve:   ${smoke} GPU node-hours
Initial staged total: ${initial} node-hours
Four-segment ceiling: ${maximum} node-hours for these nine branches
Hard project cap:     ${PHASE1_ADDITIONAL_NODE_HOUR_CAP} additional node-hours

Submission is staged:
  1. bash $script_path submit RUN_ID
  2. bash $script_path status RUN_ID
  3. bash $script_path submit-matrix RUN_ID

To recover nine immutable Float32 branch states into the corrected Float64
solver while inspecting the generated controls before allocation:
  bash $script_path prepare-recovery SOURCE_RUN_ID NEW_RUN_ID
  bash $script_path submit NEW_RUN_ID

submit-recovery SOURCE_RUN_ID NEW_RUN_ID remains the one-command equivalent
that prepares the controls and submits only the smoke job.

Before step 1, instantiate CUDA once on a Perlmutter login node:
  JULIA_PKG_PRECOMPILE_AUTO=0 $PHASE1_JULIA --project="$project_dir/gpu" \\
    -e 'using Pkg; Pkg.instantiate(; allow_autoprecomp=false)'

Do not import CUDA on the GPU-less login node. The smoke allocation performs
the first CUDA import and precompile on a real GPU. Scratch is temporary and
purgeable: archive scientifically irreplaceable full states to HPSS separately.
EOF
  print_budget
  awk -v current="$(ledger_total)" -v requested="$initial" -v cap="$PHASE1_ADDITIONAL_NODE_HOUR_CAP" \
    'BEGIN {exit !((current+requested) > cap)}' && die "initial campaign would exceed the hard cap"
  return 0
}

write_environment() {
  local path="$1" scratch_run_dir="$2"
  {
    printf 'PHASE1_RUN_SCRIPT_VERSION=%q\n' "$PHASE1_SCRIPT_VERSION"
    printf 'PHASE1_PROJECT_DIR=%q\n' "$project_dir"
    printf 'PHASE1_REPO_ROOT=%q\n' "$repo_root"
    printf 'PHASE1_ACCOUNT=%q\n' "$PHASE1_ACCOUNT"
    printf 'PHASE1_EP_ACCOUNT=%q\n' "$PHASE1_EP_ACCOUNT"
    printf 'PHASE1_QOS=%q\n' "$PHASE1_QOS"
    printf 'PHASE1_JULIA=%q\n' "$PHASE1_JULIA"
    printf 'PHASE1_GPU_TIME=%q\n' "$PHASE1_GPU_TIME"
    printf 'PHASE1_GPU_CPUS=%q\n' "$PHASE1_GPU_CPUS"
    printf 'PHASE1_SMOKE_TIME=%q\n' "$PHASE1_SMOKE_TIME"
    printf 'PHASE1_MAX_SEGMENTS=%q\n' "$PHASE1_MAX_SEGMENTS"
    printf 'PHASE1_ADDITIONAL_NODE_HOUR_CAP=%q\n' "$PHASE1_ADDITIONAL_NODE_HOUR_CAP"
    printf 'PHASE1_DECLARED_ALLOCATION_NODE_HOURS=%q\n' "$PHASE1_DECLARED_ALLOCATION_NODE_HOURS"
    printf 'PHASE1_DECLARED_USED_NODE_HOURS=%q\n' "$PHASE1_DECLARED_USED_NODE_HOURS"
    printf 'PHASE1_BASE_CONFIG=%q\n' "$PHASE1_BASE_CONFIG"
    printf 'PHASE1_RECURRENCE_CONFIG=%q\n' "$PHASE1_RECURRENCE_CONFIG"
    printf 'PHASE1_RUN_ROOT=%q\n' "$run_root"
    printf 'PHASE1_SCRATCH_ROOT=%q\n' "$scratch_root"
    printf 'PHASE1_RUN_SCRATCH_DIR=%q\n' "$scratch_run_dir"
    printf 'PHASE1_BUDGET_ROOT=%q\n' "$budget_root"
    printf 'PHASE1_LEDGER_PATH=%q\n' "$ledger_path"
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
  die "cannot resolve Phase 1 run: ${requested:-latest}"
}

load_environment() {
  local run_dir="$1"
  [[ -f "$run_dir/run.env" ]] || die "missing $run_dir/run.env"
  # Generated by write_environment with shell escaping.
  # shellcheck disable=SC1090
  source "$run_dir/run.env"
  case "${PHASE1_RUN_SCRIPT_VERSION:-missing}" in
    1.0.0|1.0.1|1.1.0|1.2.0|1.3.0|1.4.0) ;;
    *) die "unsupported run script version ${PHASE1_RUN_SCRIPT_VERSION:-missing}; current version is $PHASE1_SCRIPT_VERSION";;
  esac
  project_dir="$PHASE1_PROJECT_DIR"
  repo_root="$PHASE1_REPO_ROOT"
  run_root="$PHASE1_RUN_ROOT"
  scratch_root="${PHASE1_SCRATCH_ROOT:-$scratch_root}"
  budget_root="$PHASE1_BUDGET_ROOT"
  ledger_path="$PHASE1_LEDGER_PATH"
  lock_path="${ledger_path}.lock"
}

full_run_directory_from_control() {
  local run_dir="$1" locator="$run_dir/full_storage_path.txt" resolved
  if [[ -f "$locator" ]]; then
    resolved="$(<"$locator")"
  else
    resolved="$({
      unset PHASE1_RUN_SCRATCH_DIR
      # Generated by write_environment with shell escaping.
      # shellcheck disable=SC1090
      source "$run_dir/run.env"
      printf '%s' "${PHASE1_RUN_SCRATCH_DIR:-}"
    })"
  fi
  if [[ -n "$resolved" ]]; then
    printf '%s\n' "$resolved"
  else
    printf '%s\n' "$run_dir"
  fi
}

require_current_run_version() {
  [[ "${PHASE1_RUN_SCRIPT_VERSION:-missing}" == "$PHASE1_SCRIPT_VERSION" ]] || die \
    "run was prepared with script ${PHASE1_RUN_SCRIPT_VERSION:-missing}; preserve it for audit and start a new run with $PHASE1_SCRIPT_VERSION"
}

require_worker_compatible_run_version() {
  case "${PHASE1_RUN_SCRIPT_VERSION:-missing}" in
    1.2.0|1.3.0|1.4.0) ;;
    *) die "queued worker cannot execute run script ${PHASE1_RUN_SCRIPT_VERSION:-missing} with launcher $PHASE1_SCRIPT_VERSION";;
  esac
}

initialize_run() {
  local run_id="$1" source_run_dir="${2:-}" campaign_kind="${3:-standard}"
  [[ "$run_id" =~ ^[A-Za-z0-9_.-]+$ ]] || die "unsafe run ID: $run_id"
  [[ -f "$project_dir/gpu/Manifest.toml" ]] || die \
    "GPU environment is not instantiated; run the command printed by plan"
  require_command sha256sum
  resolve_scratch_root
  local run_dir="$run_root/$run_id" scratch_run_dir="$scratch_root/$run_id"
  [[ ! -e "$run_dir" ]] || die "run directory already exists: $run_dir"
  [[ ! -e "$scratch_run_dir" ]] || die "scratch run directory already exists: $scratch_run_dir"
  mkdir -p "$run_dir/logs" "$run_dir/ep" || die "could not create run directory: $run_dir"
  mkdir -p "$scratch_run_dir/results" "$scratch_run_dir/ep" || die \
    "could not create scratch run directory: $scratch_run_dir"
  write_environment "$run_dir/run.env" "$scratch_run_dir"
  printf '%s\n' "$scratch_run_dir" >"$run_dir/full_storage_path.txt"
  printf 'kind\tlabel\tsegment\tpool\trequested_time\treserved_node_hours\tjob_id\tconfig\n' >"$run_dir/jobs.tsv"
  printf '%s\n' "$PHASE1_DECLARED_ALLOCATION_NODE_HOURS" >"$run_dir/declared_allocation_node_hours.txt"
  printf '%s\n' "$PHASE1_DECLARED_USED_NODE_HOURS" >"$run_dir/declared_used_node_hours.txt"
  local -a prepare_args=()
  local prepare_script
  case "$campaign_kind" in
    standard)
      prepare_script="$project_dir/scripts/prepare_phase1_gpu.jl"
      prepare_args=("$PHASE1_BASE_CONFIG" "$run_dir" "$scratch_run_dir" "$run_id")
      if [[ -n "$source_run_dir" ]]; then
        prepare_args+=("$source_run_dir" "$(full_run_directory_from_control "$source_run_dir")/results")
      fi
      ;;
    recurrence)
      [[ -n "$source_run_dir" ]] || die "recurrence preparation requires a source run"
      prepare_script="$project_dir/scripts/prepare_phase1_recurrence.jl"
      prepare_args=("$PHASE1_RECURRENCE_CONFIG" "$source_run_dir" "$run_dir" "$scratch_run_dir" "$run_id")
      ;;
    *) die "unknown Phase 1 campaign kind: $campaign_kind";;
  esac
  "$PHASE1_JULIA" --startup-file=no --project="$project_dir" \
    "$prepare_script" "${prepare_args[@]}" >&2 || \
    die "Phase 1 $campaign_kind configuration preparation failed; use a new run ID after correcting the cause"
  awk 'END {print NR-1}' "$run_dir/manifest.tsv" >"$run_dir/branch_count.txt"
  cp "$project_dir/gpu/Manifest.toml" "$run_dir/gpu-Manifest.toml" || die "could not copy GPU manifest"
  sha256sum "$run_dir/gpu-Manifest.toml" >"$run_dir/gpu-Manifest.toml.sha256" || \
    die "could not hash GPU manifest"
  validate_initialized_run "$run_dir"
  mkdir -p "$run_root"
  printf '%s\n' "$run_dir" >"$run_root/latest_run.txt"
  printf '%s\n' "$run_dir"
}

validate_initialized_run() {
  local run_dir="$1" manifest_rows config_count expected_count
  [[ -f "$run_dir/run.env" ]] || die "prepared run is missing run.env"
  [[ -f "$run_dir/jobs.tsv" ]] || die "prepared run is missing jobs.tsv"
  [[ -f "$run_dir/manifest.tsv" ]] || die "prepared run is missing manifest.tsv"
  [[ -f "$run_dir/gpu-Manifest.toml" ]] || die "prepared run is missing its GPU manifest"
  [[ -f "$run_dir/gpu-Manifest.toml.sha256" ]] || die "prepared run is missing its GPU-manifest hash"
  if [[ "${PHASE1_RUN_SCRIPT_VERSION:-$PHASE1_SCRIPT_VERSION}" =~ ^1\.(3|4)\.0$ ]]; then
    [[ -d "$(full_run_directory_from_control "$run_dir")/results" ]] || die \
      "prepared run is missing its full-result scratch directory"
  fi
  expected_count=9
  [[ ! -f "$run_dir/branch_count.txt" ]] || expected_count="$(<"$run_dir/branch_count.txt")"
  [[ "$expected_count" =~ ^[1-9][0-9]*$ ]] || die "invalid prepared branch count: $expected_count"
  manifest_rows="$(awk 'END {print NR-1}' "$run_dir/manifest.tsv")"
  [[ "$manifest_rows" == "$expected_count" ]] || die \
    "prepared manifest has $manifest_rows branches instead of $expected_count"
  config_count="$(find "$run_dir/configs" -type f -name '*.segment-001.toml' | wc -l | tr -d ' ')"
  [[ "$config_count" == "$expected_count" ]] || die \
    "prepared run has $config_count initial configs instead of $expected_count"
  sha256sum -c "$run_dir/gpu-Manifest.toml.sha256" >/dev/null || die "prepared GPU-manifest hash failed"
}

acquire_budget_lock() {
  mkdir -p "$budget_root"
  local attempt
  for attempt in $(seq 1 200); do
    if mkdir "$lock_path" 2>/dev/null; then return; fi
    sleep 0.1
  done
  die "could not acquire budget ledger lock: $lock_path"
}

release_budget_lock() { rmdir "$lock_path" 2>/dev/null || true; }

ensure_ledger() {
  if [[ ! -f "$ledger_path" ]]; then
    printf 'submitted_utc\tcampaign\tkind\tlabel\tsegment\tpool\treserved_node_hours\tjob_id\trequested_time\tgit_commit\n' >"$ledger_path"
  fi
}

check_reservation() {
  local requested="$1" current
  current="$(ledger_total)"
  awk -v current="$current" -v requested="$requested" -v cap="$PHASE1_ADDITIONAL_NODE_HOUR_CAP" \
    'BEGIN {exit !((current+requested) <= cap+1e-12)}' || die \
    "reservation $requested would exceed cap $PHASE1_ADDITIONAL_NODE_HOUR_CAP (already reserved $current)"
}

record_submission() {
  local run_dir="$1" kind="$2" label="$3" segment="$4" pool="$5" requested_time="$6" reserved="$7" job_id="$8" config="$9"
  local campaign commit timestamp
  campaign="$(basename "$run_dir")"
  commit="$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || printf unknown)"
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$timestamp" "$campaign" "$kind" "$label" "$segment" "$pool" "$reserved" "$job_id" "$requested_time" "$commit" >>"$ledger_path"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$kind" "$label" "$segment" "$pool" "$requested_time" "$reserved" "$job_id" "$config" >>"$run_dir/jobs.tsv"
}

submit_smoke_job() {
  local run_dir="$1" reserved raw job_id
  awk -F'\t' '$1 == "smoke" {found=1} END {exit !found}' "$run_dir/jobs.tsv" && die "GPU smoke already submitted"
  reserved="$(gpu_node_hours "$PHASE1_SMOKE_TIME")"
  acquire_budget_lock
  trap release_budget_lock EXIT
  ensure_ledger
  check_reservation "$reserved"
  raw="$(sbatch --parsable --account="$PHASE1_ACCOUNT" --constraint='gpu&hbm80g' --qos="$PHASE1_QOS" \
    --licenses=scratch,cfs \
    --nodes=1 --ntasks=1 --cpus-per-task="$PHASE1_GPU_CPUS" --gpus-per-task=1 \
    --time="$PHASE1_SMOKE_TIME" --job-name=lmf1-gpu-smoke \
    --output="$run_dir/logs/smoke-%j.out" --export=ALL \
    "$script_path" _smoke "$run_dir")"
  job_id="${raw%%;*}"
  record_submission "$run_dir" smoke smoke 0 gpu "$PHASE1_SMOKE_TIME" "$reserved" "$job_id" "$project_dir/scripts/gpu_smoke.jl"
  release_budget_lock
  trap - EXIT
  printf '%s\n' "$job_id"
}

submit_gpu_job_locked() {
  local run_dir="$1" label="$2" segment="$3" config="$4" reserved raw job_id
  reserved="$(gpu_node_hours "$PHASE1_GPU_TIME")"
  raw="$(sbatch --parsable --account="$PHASE1_ACCOUNT" --constraint='gpu&hbm80g' --qos="$PHASE1_QOS" \
    --licenses=scratch,cfs \
    --nodes=1 --ntasks=1 --cpus-per-task="$PHASE1_GPU_CPUS" --gpus-per-task=1 \
    --time="$PHASE1_GPU_TIME" --job-name="lmf1-${label:0:40}" \
    --output="$run_dir/logs/${label}.s${segment}-%j.out" --export=ALL \
    "$script_path" _run "$run_dir" "$label" "$segment" "$config")"
  job_id="${raw%%;*}"
  record_submission "$run_dir" branch "$label" "$segment" gpu "$PHASE1_GPU_TIME" "$reserved" "$job_id" "$config"
  printf '%s\n' "$job_id"
}

submit_gpu_job() {
  local run_dir="$1" label="$2" segment="$3" config="$4" reserved job_id
  reserved="$(gpu_node_hours "$PHASE1_GPU_TIME")"
  acquire_budget_lock
  trap release_budget_lock EXIT
  ensure_ledger
  check_reservation "$reserved"
  job_id="$(submit_gpu_job_locked "$run_dir" "$label" "$segment" "$config")"
  release_budget_lock
  trap - EXIT
  printf '%s\n' "$job_id"
}

slurm_state() {
  local job_id="$1"
  sacct -n -X -j "$job_id" --format=State -P 2>/dev/null | awk -F'|' 'NF {gsub(/[ +].*/, "", $1); print $1; exit}'
}

require_completed_smoke() {
  local run_dir="$1" job_id state
  job_id="$(awk -F'\t' '$1 == "smoke" {id=$7} END {print id}' "$run_dir/jobs.tsv")"
  [[ -n "$job_id" ]] || die "submit the GPU smoke job first"
  state="$(slurm_state "$job_id")"
  [[ "$state" == "COMPLETED" ]] || die "GPU smoke job $job_id is ${state:-unknown}, not COMPLETED"
  [[ -f "$run_dir/gpu_smoke.h5" ]] || die "GPU smoke artifact is missing"
  "$PHASE1_JULIA" --startup-file=no --project="$project_dir" \
    "$project_dir/scripts/validate_gpu_smoke.jl" "$run_dir/gpu_smoke.h5" >&2 || die \
    "GPU smoke artifact failed Float64/runtime/preflight validation"
}

submit_matrix_jobs() {
  local run_dir="$1"
  require_completed_smoke "$run_dir"
  local label config matrix_reservation index
  local -a pending_labels=() pending_configs=()
  acquire_budget_lock
  trap release_budget_lock EXIT
  ensure_ledger
  while IFS=$'\t' read -r label _ _ _ _ config _; do
    [[ "$label" == "label" ]] && continue
    if ! awk -F'\t' -v wanted="$label" \
      '$1 == "branch" && $2 == wanted {found=1} END {exit !found}' "$run_dir/jobs.tsv"; then
      pending_labels+=("$label")
      pending_configs+=("$config")
    fi
  done <"$run_dir/manifest.tsv"
  (( ${#pending_labels[@]} > 0 )) || die "all prepared Phase 1 branches are already submitted"
  matrix_reservation="$(awk \
    -v segment="$(gpu_node_hours "$PHASE1_GPU_TIME")" \
    -v count="${#pending_labels[@]}" \
    'BEGIN {printf "%.9f", count*segment}')"
  check_reservation "$matrix_reservation"
  for ((index=0; index<${#pending_labels[@]}; index++)); do
    submit_gpu_job_locked "$run_dir" "${pending_labels[index]}" 1 "${pending_configs[index]}" >/dev/null
  done
  release_budget_lock
  trap - EXIT
  echo "submitted ${#pending_labels[@]} prepared Phase 1 GPU branches"
}

latest_source_state() {
  local run_dir="$1" label="$2" state checkpoint result_root
  result_root="$(full_run_directory_from_control "$run_dir")/results/$label"
  state="$(ls -1t "$result_root"/*/*/state.h5 "$result_root"/*/state.h5 2>/dev/null | head -1 || true)"
  [[ -n "$state" ]] && { printf '%s\n' "$state"; return; }
  checkpoint="$(ls -1t "$result_root"/*/*/checkpoint_latest.h5 "$result_root"/*/checkpoint_latest.h5 2>/dev/null | head -1 || true)"
  [[ -n "$checkpoint" ]] && { printf '%s\n' "$checkpoint"; return; }
  die "no state or checkpoint found for $label"
}

latest_analysis_state() {
  local run_dir="$1" label="$2" state checkpoint result_root
  if [[ -d "$run_dir/stateless_results" ]]; then
    result_root="$run_dir/stateless_results/$label"
  else
    result_root="$run_dir/results/$label"
  fi
  state="$(ls -1t "$result_root"/*/*/state.h5 "$result_root"/*/state.h5 2>/dev/null | head -1 || true)"
  [[ -n "$state" ]] && { printf '%s\n' "$state"; return; }
  checkpoint="$(ls -1t "$result_root"/*/*/checkpoint_latest.h5 "$result_root"/*/checkpoint_latest.h5 2>/dev/null | head -1 || true)"
  [[ -n "$checkpoint" ]] && { printf '%s\n' "$checkpoint"; return; }
  return 1
}

continue_branch() {
  local run_dir="$1" label="$2" latest_row segment job_id state source previous_config next_segment next_config
  [[ "$label" =~ ^[A-Za-z0-9_.-]+$ ]] || die "unsafe branch label: $label"
  latest_row="$(awk -F'\t' -v label="$label" '$1 == "branch" && $2 == label {row=$0} END {print row}' "$run_dir/jobs.tsv")"
  [[ -n "$latest_row" ]] || die "no submitted branch named $label"
  IFS=$'\t' read -r _ _ segment _ _ _ job_id previous_config <<<"$latest_row"
  (( segment < PHASE1_MAX_SEGMENTS )) || die "$label already reached the $PHASE1_MAX_SEGMENTS-segment ceiling"
  state="$(slurm_state "$job_id")"
  case "$state" in
    COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY) ;;
    *) die "latest job $job_id is ${state:-unknown}; wait for a terminal state";;
  esac
  source="$(latest_source_state "$run_dir" "$label")"
  next_segment=$(( segment + 1 ))
  next_config="$run_dir/configs/$label.segment-$(printf '%03d' "$next_segment").toml"
  "$PHASE1_JULIA" --startup-file=no --project="$project_dir" \
    "$project_dir/scripts/prepare_phase1_resume.jl" "$source" "$previous_config" "$next_config"
  submit_gpu_job "$run_dir" "$label" "$next_segment" "$next_config"
}

submit_ep_job() {
  local run_dir="$1" label="$2" L="$3" U="$4" V="$5" t0="$6" density="$7"
  shift 7
  [[ "$label" =~ ^[A-Za-z0-9_.-]+$ ]] || die "unsafe E_p label: $label"
  local arg
  for arg in "$@"; do
    [[ "$arg" != --outfile && "$arg" != --outfile=* ]] || die "the guarded launcher owns --outfile"
  done
  local reserved outfile compact_outfile raw job_id
  reserved="$(ep_node_hours)"
  outfile="$PHASE1_RUN_SCRATCH_DIR/ep/$label.h5"
  compact_outfile="$run_dir/ep/$label.h5"
  [[ ! -e "$outfile" && ! -e "$compact_outfile" ]] || die "E_p output already exists for label: $label"
  acquire_budget_lock
  trap release_budget_lock EXIT
  ensure_ledger
  check_reservation "$reserved"
  raw="$(sbatch --parsable --account="$PHASE1_EP_ACCOUNT" --constraint=cpu --qos="$PHASE1_QOS" \
    --licenses=scratch,cfs \
    --nodes=1 --ntasks=1 --cpus-per-task=64 --time=48:00:00 \
    --job-name="lmf-ep-${label:0:40}" --chdir="$repo_root" \
    --output="$run_dir/logs/ep-${label}-%j.out" --export=ALL \
    "$script_path" _ep "$run_dir" "$label" "$L" "$U" "$V" "$t0" "$density" "$@")"
  job_id="${raw%%;*}"
  record_submission "$run_dir" ep "$label" 1 cpu 48:00:00 "$reserved" "$job_id" "$outfile"
  release_budget_lock
  trap - EXIT
  printf '%s\n' "$job_id"
}

show_status() {
  local run_dir="$1"
  printf '%-8s %-35s %-7s %-12s %-10s\n' KIND LABEL SEGMENT JOB_ID STATE
  local kind label segment _ _ _ job_id _ state
  while IFS=$'\t' read -r kind label segment _ _ _ job_id _; do
    [[ "$kind" == "kind" ]] && continue
    state="$(slurm_state "$job_id")"
    printf '%-8s %-35s %-7s %-12s %-10s\n' "$kind" "$label" "$segment" "$job_id" "${state:-UNKNOWN}"
  done <"$run_dir/jobs.tsv"
  echo
  printf 'LATEST STATE\tSTATUS\tACCEPTED\tPERIOD\tENERGY\tSCALAR\n'
  local -a states=()
  local branch
  while IFS= read -r branch; do
    state="$(latest_analysis_state "$run_dir" "$branch" 2>/dev/null || true)"
    [[ -n "$state" ]] && states+=("$state")
  done < <(awk -F'\t' '$1 == "branch" {seen[$2]=1} END {for (label in seen) print label}' "$run_dir/jobs.tsv" | sort)
  if (( ${#states[@]} > 0 )); then
    "$PHASE1_JULIA" --startup-file=no --project="$project_dir" \
      "$project_dir/scripts/phase1_state_status.jl" "${states[@]}"
  else
    echo "none"
  fi
  echo
  print_budget
}

sanitize_cuda_runtime_environment() {
  local cuda_root="${CUDA_HOME:-${CUDA_PATH:-${CUDA_ROOT:-}}}"
  local path_entry
  local -a original_entries=() retained_entries=()
  IFS=: read -r -a original_entries <<<"${LD_LIBRARY_PATH:-}"
  for path_entry in "${original_entries[@]}"; do
    [[ -n "$path_entry" ]] || continue
    [[ "$path_entry" == /opt/nvidia/hpc_sdk/* ]] && continue
    [[ "$path_entry" == /usr/local/cuda* ]] && continue
    [[ -n "$cuda_root" && "$path_entry" == "$cuda_root"* ]] && continue
    retained_entries+=("$path_entry")
  done
  if (( ${#retained_entries[@]} > 0 )); then
    local joined
    printf -v joined '%s:' "${retained_entries[@]}"
    export LD_LIBRARY_PATH="${joined%:}"
  else
    unset LD_LIBRARY_PATH
  fi
  unset CUDA_HOME CUDA_PATH CUDA_ROOT
}

run_gpu_environment() {
  export JULIA_LOAD_PATH="$project_dir/gpu:$project_dir:@stdlib"
  export JULIA_PKG_PRECOMPILE_AUTO=0
  export JULIA_NUM_THREADS=1
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  export SLURM_CPU_BIND=cores
  module load julia
  module unload cudatoolkit >/dev/null 2>&1 || true
  sanitize_cuda_runtime_environment
}

worker_smoke() {
  local run_dir="$1"
  load_environment "$run_dir"
  require_worker_compatible_run_version
  run_gpu_environment
  cd "$project_dir"
  srun --ntasks=1 --cpus-per-task="$PHASE1_GPU_CPUS" --cpu-bind=cores \
    "$PHASE1_JULIA" --startup-file=no --project="$project_dir/gpu" \
    "$project_dir/scripts/gpu_smoke.jl" "$run_dir/gpu_smoke.h5"
}

worker_run() {
  local run_dir="$1" label="$2" segment="$3" config="$4"
  load_environment "$run_dir"
  require_worker_compatible_run_version
  run_gpu_environment
  cd "$project_dir"
  echo "phase1_label=$label"
  echo "phase1_segment=$segment"
  echo "phase1_config=$config"
  if [[ "$PHASE1_RUN_SCRIPT_VERSION" == "1.2.0" ]]; then
    echo "warning: completing already-submitted v1.2 worker with legacy CFS result storage"
    srun --ntasks=1 --cpus-per-task="$PHASE1_GPU_CPUS" --cpu-bind=cores \
      "$PHASE1_JULIA" --startup-file=no --project="$project_dir/gpu" \
      "$project_dir/scripts/run_scf_gpu.jl" "$config"
    return
  fi
  local solver_status compact_status=0 full_branch compact_branch
  full_branch="$PHASE1_RUN_SCRATCH_DIR/results/$label"
  compact_branch="$run_dir/results/$label"
  set +e
  srun --ntasks=1 --cpus-per-task="$PHASE1_GPU_CPUS" --cpu-bind=cores \
    "$PHASE1_JULIA" --startup-file=no --project="$project_dir/gpu" \
    "$project_dir/scripts/run_scf_gpu.jl" "$config"
  solver_status=$?
  set -e
  if [[ -d "$full_branch" ]]; then
    "$PHASE1_JULIA" --startup-file=no --project="$project_dir" \
      "$project_dir/scripts/compact_results.jl" "$full_branch" "$compact_branch" || compact_status=$?
  fi
  (( compact_status == 0 )) || die \
    "stateless CFS mirror failed with status $compact_status; full data remain at $full_branch"
  exit "$solver_status"
}

worker_ep() {
  local run_dir="$1" label="$2" L="$3" U="$4" V="$5" t0="$6" density="$7"
  shift 7
  load_environment "$run_dir"
  require_current_run_version
  module load julia
  mkdir -p "$PHASE1_RUN_SCRATCH_DIR/ep" "$run_dir/ep"
  local full_outfile="$PHASE1_RUN_SCRATCH_DIR/ep/$label.h5" solver_status compact_status=0
  cd "$repo_root"
  set +e
  "$repo_root/submit_E_p_ladder.sh" "$L" "$U" "$V" "$t0" "$density" "$@" --outfile "$full_outfile"
  solver_status=$?
  set -e
  if [[ -f "$full_outfile" ]]; then
    "$PHASE1_JULIA" --startup-file=no --project="$project_dir" \
      "$project_dir/scripts/compact_results.jl" "$PHASE1_RUN_SCRATCH_DIR/ep" "$run_dir/ep" || compact_status=$?
  fi
  (( compact_status == 0 )) || die \
    "stateless E_p mirror failed with status $compact_status; full data remain at $full_outfile"
  exit "$solver_status"
}

usage() {
  cat <<EOF
Usage: bash $script_path ACTION [ARGS]

Read-only:
  plan
  plan-recurrence
  budget
  status [RUN_ID]
  show [RUN_ID]

Preparation only (no Slurm submission or budget reservation):
  prepare-recovery SOURCE_RUN NEW_RUN   Prepare Float64 warm-start controls
  prepare-recurrence SOURCE_RUN NEW_RUN Prepare phase-resolved chi=400 recurrence controls

Submissions:
  submit RUN_ID                         Prepare campaign and submit GPU smoke only
  submit-recovery SOURCE_RUN NEW_RUN    Prepare Float64 warm-start recovery and submit smoke
  submit-matrix RUN_ID                  Submit all prepared branches after smoke completes
  continue RUN_ID LABEL                 Submit an explicit same-model continuation
  submit-ep RUN_ID LABEL L U V t0 n ... Submit a guarded legacy CPU E_p calculation

Storage:
  Full MPS checkpoints are written below \$PSCRATCH; CFS results and E_p files
  are automatically mirrored without MPS tensors for analysis and local sync.
EOF
}

action="${1:-plan}"
case "$action" in
  plan) print_plan;;
  plan-recurrence) print_recurrence_plan;;
  budget) print_budget;;
  submit)
    [[ $# == 2 ]] || die "submit requires RUN_ID"
    require_command sbatch
    if [[ -d "$run_root/$2" ]]; then
      run_dir="$(resolve_run_dir "$2")"
      load_environment "$run_dir"
      require_current_run_version
      validate_initialized_run "$run_dir"
    else
      check_reservation "$(gpu_node_hours "$PHASE1_SMOKE_TIME")"
      run_dir="$(initialize_run "$2")"
    fi
    submit_smoke_job "$run_dir"
    ;;
  prepare-recovery)
    [[ $# == 3 ]] || die "prepare-recovery requires SOURCE_RUN_ID NEW_RUN_ID"
    source_run_dir="$(resolve_run_dir "$2")"
    [[ ! -e "$run_root/$3" ]] || die "new recovery run already exists: $run_root/$3"
    initialize_run "$3" "$source_run_dir"
    ;;
  prepare-recurrence)
    [[ $# == 3 ]] || die "prepare-recurrence requires SOURCE_RUN_ID NEW_RUN_ID"
    source_run_dir="$(resolve_run_dir "$2")"
    [[ ! -e "$run_root/$3" ]] || die "new recurrence run already exists: $run_root/$3"
    initialize_run "$3" "$source_run_dir" recurrence
    ;;
  submit-recovery)
    [[ $# == 3 ]] || die "submit-recovery requires SOURCE_RUN_ID NEW_RUN_ID"
    require_command sbatch
    source_run_dir="$(resolve_run_dir "$2")"
    [[ ! -e "$run_root/$3" ]] || die "new recovery run already exists: $run_root/$3"
    check_reservation "$(gpu_node_hours "$PHASE1_SMOKE_TIME")"
    run_dir="$(initialize_run "$3" "$source_run_dir")"
    submit_smoke_job "$run_dir"
    ;;
  submit-matrix)
    [[ $# == 2 ]] || die "submit-matrix requires RUN_ID"
    require_command sbatch; require_command sacct
    run_dir="$(resolve_run_dir "$2")"; load_environment "$run_dir"; require_current_run_version; submit_matrix_jobs "$run_dir"
    ;;
  continue)
    [[ $# == 3 ]] || die "continue requires RUN_ID LABEL"
    require_command sbatch; require_command sacct
    run_dir="$(resolve_run_dir "$2")"; load_environment "$run_dir"; require_current_run_version; continue_branch "$run_dir" "$3"
    ;;
  submit-ep)
    (( $# >= 8 )) || die "submit-ep requires RUN_ID LABEL L U V t0 density [options]"
    require_command sbatch
    run_dir="$(resolve_run_dir "$2")"; load_environment "$run_dir"
    require_current_run_version
    submit_ep_job "$run_dir" "$3" "$4" "$5" "$6" "$7" "$8" "${@:9}"
    ;;
  status|show)
    require_command sacct
    run_dir="$(resolve_run_dir "${2:-}")"; load_environment "$run_dir"; show_status "$run_dir"
    ;;
  _smoke)
    [[ $# == 2 ]] || die "internal smoke action requires RUN_DIR"
    worker_smoke "$2"
    ;;
  _run)
    [[ $# == 5 ]] || die "internal run action requires RUN_DIR LABEL SEGMENT CONFIG"
    worker_run "$2" "$3" "$4" "$5"
    ;;
  _ep)
    (( $# >= 8 )) || die "internal E_p action requires RUN_DIR LABEL L U V t0 density [options]"
    worker_ep "$2" "$3" "$4" "$5" "$6" "$7" "$8" "${@:9}"
    ;;
  -h|--help|help) usage;;
  *) usage; die "unknown action: $action";;
esac
