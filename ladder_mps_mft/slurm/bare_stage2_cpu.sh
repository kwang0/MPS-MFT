#!/bin/bash

# Guarded Perlmutter CPU launcher for Stage 2 of the bare-ladder hybrid
# response search. The default action is the read-only discovery plan.

set -euo pipefail

readonly SCRIPT_VERSION="1.0.1"
readonly PHYSICAL_CORES_PER_NODE=128
readonly MIB_PER_LOGICAL_CPU=1952
readonly NORMAL_PROBES=9
readonly PAIR_PROBES=3
readonly VALIDATION_PROBES=3

script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
project_dir="${BARE_STAGE2_PROJECT_DIR:-$(cd "$(dirname "$script_path")/.." && pwd)}"
control_root="${BARE_STAGE2_CONTROL_ROOT:-$project_dir/output/bare_stage2}"
full_root="${BARE_STAGE2_FULL_ROOT:-${PSCRATCH:-$project_dir/output}/MPS-MFT/ladder_mps_mft/bare_stage2}"
source_control_root="${BARE_STAGE2_SOURCE_ROOT:-$project_dir/output/bare_stage1}"

BARE_STAGE2_ACCOUNT="${BARE_STAGE2_ACCOUNT:-m4863}"
BARE_STAGE2_QOS="${BARE_STAGE2_QOS:-shared}"
BARE_STAGE2_CONFIG="${BARE_STAGE2_CONFIG:-$project_dir/configs/bare_stage2_t014_v0.toml}"
BARE_STAGE2_JULIA="${BARE_STAGE2_JULIA:-julia}"
BARE_STAGE2_JULIA_THREADS="${BARE_STAGE2_JULIA_THREADS:-4}"
BARE_STAGE2_LOGICAL_CPUS="${BARE_STAGE2_LOGICAL_CPUS:-8}"
BARE_STAGE2_PREP_MEMORY="${BARE_STAGE2_PREP_MEMORY:-8G}"
BARE_STAGE2_PREP_TIME="${BARE_STAGE2_PREP_TIME:-02:00:00}"
BARE_STAGE2_NORMAL_MEMORY="${BARE_STAGE2_NORMAL_MEMORY:-48G}"
BARE_STAGE2_NORMAL_TIME="${BARE_STAGE2_NORMAL_TIME:-12:00:00}"
BARE_STAGE2_PAIR_MEMORY="${BARE_STAGE2_PAIR_MEMORY:-64G}"
BARE_STAGE2_PAIR_TIME="${BARE_STAGE2_PAIR_TIME:-12:00:00}"
BARE_STAGE2_ASSEMBLE_MEMORY="${BARE_STAGE2_ASSEMBLE_MEMORY:-8G}"
BARE_STAGE2_ASSEMBLE_TIME="${BARE_STAGE2_ASSEMBLE_TIME:-02:00:00}"
BARE_STAGE2_MAX_NODE_HOURS="${BARE_STAGE2_MAX_NODE_HOURS:-24.0}"
BARE_STAGE2_VALIDATION_MEMORY="${BARE_STAGE2_VALIDATION_MEMORY:-64G}"
BARE_STAGE2_VALIDATION_TIME="${BARE_STAGE2_VALIDATION_TIME:-24:00:00}"
BARE_STAGE2_VALIDATION_MAX_NODE_HOURS="${BARE_STAGE2_VALIDATION_MAX_NODE_HOURS:-12.0}"

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
  local prepare normal_reference normal pair_reference pair_probes assemble total
  prepare="$(reservation_node_hours 1 2 "$BARE_STAGE2_PREP_MEMORY" "$BARE_STAGE2_PREP_TIME")"
  normal_reference="$(reservation_node_hours 1 "$BARE_STAGE2_LOGICAL_CPUS" "$BARE_STAGE2_NORMAL_MEMORY" "$BARE_STAGE2_NORMAL_TIME")"
  normal="$(reservation_node_hours "$NORMAL_PROBES" "$BARE_STAGE2_LOGICAL_CPUS" "$BARE_STAGE2_NORMAL_MEMORY" "$BARE_STAGE2_NORMAL_TIME")"
  pair_reference="$(reservation_node_hours 1 "$BARE_STAGE2_LOGICAL_CPUS" "$BARE_STAGE2_PAIR_MEMORY" "$BARE_STAGE2_PAIR_TIME")"
  pair_probes="$(reservation_node_hours "$PAIR_PROBES" "$BARE_STAGE2_LOGICAL_CPUS" "$BARE_STAGE2_PAIR_MEMORY" "$BARE_STAGE2_PAIR_TIME")"
  assemble="$(reservation_node_hours 1 2 "$BARE_STAGE2_ASSEMBLE_MEMORY" "$BARE_STAGE2_ASSEMBLE_TIME")"
  total="$(awk -v a="$prepare" -v b="$normal_reference" -v c="$normal" -v d="$pair_reference" -v e="$pair_probes" -v f="$assemble" \
    'BEGIN {printf "%.9f", a+b+c+d+e+f}')"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$prepare" "$normal_reference" "$normal" "$pair_reference" "$pair_probes" "$assemble" "$total"
}

validation_upper_bound_node_hours() {
  local probes assemble total
  probes="$(reservation_node_hours "$VALIDATION_PROBES" "$BARE_STAGE2_LOGICAL_CPUS" "$BARE_STAGE2_VALIDATION_MEMORY" "$BARE_STAGE2_VALIDATION_TIME")"
  assemble="$(reservation_node_hours 1 2 "$BARE_STAGE2_ASSEMBLE_MEMORY" "$BARE_STAGE2_ASSEMBLE_TIME")"
  total="$(awk -v a="$probes" -v b="$assemble" 'BEGIN {printf "%.9f", a+b}')"
  printf '%s\t%s\t%s\n' "$probes" "$assemble" "$total"
}

resolve_source_control_dir() {
  local requested="${1:-}"
  [[ -n "$requested" ]] || die "a completed Stage 1 source run is required"
  if [[ -d "$requested" ]]; then cd "$requested" && pwd; return; fi
  if [[ -d "$source_control_root/$requested" ]]; then cd "$source_control_root/$requested" && pwd; return; fi
  die "cannot resolve Stage 1 source run: $requested"
}

source_full_directory() {
  local source_control_dir="$1"
  [[ -f "$source_control_dir/full_run_path.txt" ]] || die \
    "missing Stage 1 full_run_path.txt in $source_control_dir"
  local source_full_dir
  source_full_dir="$(<"$source_control_dir/full_run_path.txt")"
  [[ -d "$source_full_dir" ]] || die "Stage 1 scratch directory does not exist: $source_full_dir"
  [[ -f "$source_full_dir/backbone.h5" ]] || die "missing Stage 1 backbone: $source_full_dir/backbone.h5"
  [[ -f "$source_full_dir/stage1.h5" ]] || die "missing Stage 1 covariance artifact: $source_full_dir/stage1.h5"
  printf '%s\n' "$source_full_dir"
}

validate() {
  [[ -f "$BARE_STAGE2_CONFIG" ]] || die "missing Stage 2 config: $BARE_STAGE2_CONFIG"
  [[ -f "$project_dir/Project.toml" ]] || die "missing Julia project: $project_dir"
  [[ "$BARE_STAGE2_QOS" == "shared" ]] || die "the Stage 2 pilot is budgeted only for shared QOS"
  (( BARE_STAGE2_JULIA_THREADS >= 1 )) || die "BARE_STAGE2_JULIA_THREADS must be positive"
  (( BARE_STAGE2_LOGICAL_CPUS >= 2 * BARE_STAGE2_JULIA_THREADS )) || die \
    "request at least two Slurm logical CPUs per Julia thread on Perlmutter"
}

print_plan() {
  validate
  local source_control_dir source_full_dir prepare normal_reference normal pair_reference pair_probes assemble total
  source_control_dir="$(resolve_source_control_dir "$1")"
  source_full_dir="$(source_full_directory "$source_control_dir")"
  IFS=$'\t' read -r prepare normal_reference normal pair_reference pair_probes assemble total < <(upper_bound_node_hours)
  cat <<EOF
Bare-ladder Stage 2 projected-response discovery

Stage 1 control:    $source_control_dir
Stage 1 full data:  $source_full_dir
Stage 2 config:     $BARE_STAGE2_CONFIG
Named candidates:   14 (11 motivated + 3 Stage-1 covariance additions)
Independent probes: $NORMAL_PROBES normal + $PAIR_PROBES pairing = $((NORMAL_PROBES + PAIR_PROBES))
Probe amplitude:    1e-4 per physical ladder site
DMRG topology:      ${BARE_STAGE2_JULIA_THREADS} Julia block-sparse threads, ${BARE_STAGE2_LOGICAL_CPUS} Slurm logical CPUs
Normal probe job:   ${BARE_STAGE2_NORMAL_MEMORY}, ${BARE_STAGE2_NORMAL_TIME}
Pairing jobs:       ${BARE_STAGE2_PAIR_MEMORY}, ${BARE_STAGE2_PAIR_TIME}
Worst-case reserve: prepare=${prepare}, normal-reference=${normal_reference}, normal=${normal}, pair-reference=${pair_reference}, pair-probes=${pair_probes}, assembly=${assemble}, total=${total} node-hours
Enforced cap:       ${BARE_STAGE2_MAX_NODE_HOURS} node-hours

The preparation job verifies the Stage 1/backbone hashes once and constructs
the orthonormal candidate bank. A stricter number-conserving zero-field solve
removes residual backbone relaxation before any finite difference is taken.
The nine normal probes and parity-only zero-field pairing reference then run in
parallel. The three pairing probes start only after that reference passes.
Assembly checks reciprocity,
diagonalizes the full projected response for all three geometries, and writes
the three proposed h/2 validation modes. Validation is a separate decision gate.
EOF
  if awk -v total="$total" -v cap="$BARE_STAGE2_MAX_NODE_HOURS" 'BEGIN {exit !(total > cap)}'; then
    die "worst-case reservation exceeds BARE_STAGE2_MAX_NODE_HOURS"
  fi
}

print_validation_plan() {
  validate
  local control_dir probes assemble total
  control_dir="$(resolve_control_dir "$1")"
  load_run "$control_dir"
  [[ -f "$BARE_STAGE2_FULL_DIR/stage2_discovery.h5" ]] || die \
    "Stage 2 discovery is not complete: $BARE_STAGE2_FULL_DIR/stage2_discovery.h5"
  [[ -f "$BARE_STAGE2_FULL_DIR/stage2_discovery_gates.tsv" ]] || die \
    "Stage 2 discovery gate summary is missing"
  awk -F'\t' 'NR == 2 {seen=1; exit !($1 == "true")} END {if (!seen) exit 1}' \
    "$BARE_STAGE2_FULL_DIR/stage2_discovery_gates.tsv" || die \
    "Stage 2 discovery did not pass its scientific gates; validation is not authorized"
  [[ -f "$BARE_STAGE2_FULL_DIR/candidate_bank.h5" ]] || die "missing candidate bank"
  [[ -f "$BARE_STAGE2_FULL_DIR/normal_reference.h5" ]] || die "missing normal reference"
  [[ -f "$BARE_STAGE2_FULL_DIR/pair_reference.h5" ]] || die "missing pairing reference"
  IFS=$'\t' read -r probes assemble total < <(validation_upper_bound_node_hours)
  cat <<EOF
Bare-ladder Stage 2 h/h2 validation

Discovery run:      $control_dir
Validation modes:   $VALIDATION_PROBES selected eigenvectors
Per mode:           two sequential DMRG solves at h=1e-4 and h/2=5e-5
DMRG topology:      ${BARE_STAGE2_JULIA_THREADS} Julia block-sparse threads, ${BARE_STAGE2_LOGICAL_CPUS} Slurm logical CPUs
Per validation job: ${BARE_STAGE2_VALIDATION_MEMORY}, ${BARE_STAGE2_VALIDATION_TIME}
Worst-case reserve: probes=${probes}, assembly=${assemble}, total=${total} node-hours
Enforced cap:       ${BARE_STAGE2_VALIDATION_MAX_NODE_HOURS} node-hours

This action is intentionally separate from discovery. Run it only after the
synced discovery matrices and their reciprocity gate have been reviewed.
EOF
  if awk -v total="$total" -v cap="$BARE_STAGE2_VALIDATION_MAX_NODE_HOURS" 'BEGIN {exit !(total > cap)}'; then
    die "worst-case validation reservation exceeds BARE_STAGE2_VALIDATION_MAX_NODE_HOURS"
  fi
}

set_thread_environment() {
  export JULIA_NUM_THREADS="$BARE_STAGE2_JULIA_THREADS"
  export JULIA_PKG_PRECOMPILE_AUTO=0
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  export OMP_PROC_BIND=spread OMP_PLACES=cores
}

initialize_run() {
  local run_id="$1" source_control_dir="$2" source_full_dir="$3"
  local control_dir="$control_root/$run_id" full_dir="$full_root/$run_id"
  [[ "$run_id" =~ ^[A-Za-z0-9_.-]+$ ]] || die "unsafe run id: $run_id"
  [[ ! -e "$control_dir" ]] || die "control directory already exists: $control_dir"
  [[ ! -e "$full_dir" ]] || die "scratch directory already exists: $full_dir"
  mkdir -p "$control_dir/logs" "$full_dir/probes"
  cp "$BARE_STAGE2_CONFIG" "$control_dir/config.toml"
  cat >"$control_dir/run.env" <<EOF
BARE_STAGE2_RUN_SCRIPT_VERSION='$SCRIPT_VERSION'
BARE_STAGE2_PROJECT_DIR='$project_dir'
BARE_STAGE2_CONTROL_DIR='$control_dir'
BARE_STAGE2_FULL_DIR='$full_dir'
BARE_STAGE2_CONFIG_COPY='$control_dir/config.toml'
BARE_STAGE2_SOURCE_CONTROL_DIR='$source_control_dir'
BARE_STAGE2_SOURCE_FULL_DIR='$source_full_dir'
BARE_STAGE2_ACCOUNT='$BARE_STAGE2_ACCOUNT'
BARE_STAGE2_QOS='$BARE_STAGE2_QOS'
BARE_STAGE2_JULIA='$BARE_STAGE2_JULIA'
BARE_STAGE2_JULIA_THREADS='$BARE_STAGE2_JULIA_THREADS'
BARE_STAGE2_LOGICAL_CPUS='$BARE_STAGE2_LOGICAL_CPUS'
EOF
  printf 'kind\tlabel\tjob_id\n' >"$control_dir/jobs.tsv"
  printf '%s\n' "$full_dir" >"$control_dir/full_run_path.txt"
  printf '%s\n' "$source_control_dir" >"$control_dir/source_stage1_control_path.txt"
  printf '%s\n' "$source_full_dir" >"$control_dir/source_stage1_full_path.txt"
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
    [[ -d "$latest" ]] || die "latest Stage 2 run does not exist: $latest"
    cd "$latest" && pwd; return
  fi
  die "cannot resolve Stage 2 run: ${requested:-latest}"
}

load_run() {
  local control_dir="$1"
  [[ -f "$control_dir/run.env" ]] || die "missing run.env in $control_dir"
  # Generated by initialize_run from validated paths and scalar values.
  # shellcheck disable=SC1090
  source "$control_dir/run.env"
  [[ "$BARE_STAGE2_RUN_SCRIPT_VERSION" == "$SCRIPT_VERSION" ]] || die \
    "run script version $BARE_STAGE2_RUN_SCRIPT_VERSION differs from $SCRIPT_VERSION"
  project_dir="$BARE_STAGE2_PROJECT_DIR"
}

submit_discovery() {
  local run_id="$1" source_requested="$2"
  print_plan "$source_requested"
  "$project_dir/slurm/phase0_calibrate_cpu.sh" plan >/dev/null
  require_command sbatch
  local source_control_dir source_full_dir initialized control_dir full_dir
  source_control_dir="$(resolve_source_control_dir "$source_requested")"
  source_full_dir="$(source_full_directory "$source_control_dir")"
  initialized="$(initialize_run "$run_id" "$source_control_dir" "$source_full_dir")"
  IFS=$'\t' read -r control_dir full_dir <<<"$initialized"

  local prep_raw prep_id normal0_raw normal0_id normal_raw normal_id pair0_raw pair0_id pair_raw pair_id assemble_raw assemble_id
  prep_raw="$(sbatch --parsable --account="$BARE_STAGE2_ACCOUNT" --constraint=cpu --qos="$BARE_STAGE2_QOS" \
    --licenses=scratch,cfs --nodes=1 --ntasks=1 --cpus-per-task=2 --mem="$BARE_STAGE2_PREP_MEMORY" \
    --time="$BARE_STAGE2_PREP_TIME" --job-name=lmf-s2-prep \
    --output="$control_dir/logs/prepare-%j.out" --export=ALL \
    "$script_path" _prepare "$control_dir")"
  prep_id="${prep_raw%%;*}"
  printf 'prepare\tcandidate_bank\t%s\n' "$prep_id" >>"$control_dir/jobs.tsv"

  normal0_raw="$(sbatch --parsable --account="$BARE_STAGE2_ACCOUNT" --constraint=cpu --qos="$BARE_STAGE2_QOS" \
    --licenses=scratch,cfs --nodes=1 --ntasks=1 --cpus-per-task="$BARE_STAGE2_LOGICAL_CPUS" \
    --mem="$BARE_STAGE2_NORMAL_MEMORY" --time="$BARE_STAGE2_NORMAL_TIME" --job-name=lmf-s2-zero \
    --dependency="afterok:$prep_id" --kill-on-invalid-dep=yes \
    --output="$control_dir/logs/normal-reference-%j.out" --export=ALL \
    "$script_path" _normal_reference "$control_dir")"
  normal0_id="${normal0_raw%%;*}"
  printf 'normal_reference\tstrict_zero_field\t%s\n' "$normal0_id" >>"$control_dir/jobs.tsv"

  normal_raw="$(sbatch --parsable --account="$BARE_STAGE2_ACCOUNT" --constraint=cpu --qos="$BARE_STAGE2_QOS" \
    --licenses=scratch,cfs --nodes=1 --ntasks=1 --cpus-per-task="$BARE_STAGE2_LOGICAL_CPUS" \
    --mem="$BARE_STAGE2_NORMAL_MEMORY" --time="$BARE_STAGE2_NORMAL_TIME" --array=1-"$NORMAL_PROBES" \
    --job-name=lmf-s2-normal --dependency="afterok:$normal0_id" --kill-on-invalid-dep=yes \
    --output="$control_dir/logs/normal-%A_%a.out" --export=ALL \
    "$script_path" _normal "$control_dir")"
  normal_id="${normal_raw%%;*}"
  printf 'normal_array\tprojected_normal_1_to_%s\t%s\n' "$NORMAL_PROBES" "$normal_id" >>"$control_dir/jobs.tsv"

  pair0_raw="$(sbatch --parsable --account="$BARE_STAGE2_ACCOUNT" --constraint=cpu --qos="$BARE_STAGE2_QOS" \
    --licenses=scratch,cfs --nodes=1 --ntasks=1 --cpus-per-task="$BARE_STAGE2_LOGICAL_CPUS" \
    --mem="$BARE_STAGE2_PAIR_MEMORY" --time="$BARE_STAGE2_PAIR_TIME" --job-name=lmf-s2-pair0 \
    --dependency="afterok:$normal0_id" --kill-on-invalid-dep=yes \
    --output="$control_dir/logs/pair-reference-%j.out" --export=ALL \
    "$script_path" _pair_reference "$control_dir")"
  pair0_id="${pair0_raw%%;*}"
  printf 'pair_reference\tparity_only_zero_field\t%s\n' "$pair0_id" >>"$control_dir/jobs.tsv"

  pair_raw="$(sbatch --parsable --account="$BARE_STAGE2_ACCOUNT" --constraint=cpu --qos="$BARE_STAGE2_QOS" \
    --licenses=scratch,cfs --nodes=1 --ntasks=1 --cpus-per-task="$BARE_STAGE2_LOGICAL_CPUS" \
    --mem="$BARE_STAGE2_PAIR_MEMORY" --time="$BARE_STAGE2_PAIR_TIME" --array=1-"$PAIR_PROBES" \
    --job-name=lmf-s2-pair --dependency="afterok:$pair0_id" --kill-on-invalid-dep=yes \
    --output="$control_dir/logs/pair-%A_%a.out" --export=ALL \
    "$script_path" _pair "$control_dir")"
  pair_id="${pair_raw%%;*}"
  printf 'pair_array\tprojected_pair_1_to_%s\t%s\n' "$PAIR_PROBES" "$pair_id" >>"$control_dir/jobs.tsv"

  assemble_raw="$(sbatch --parsable --account="$BARE_STAGE2_ACCOUNT" --constraint=cpu --qos="$BARE_STAGE2_QOS" \
    --licenses=scratch,cfs --nodes=1 --ntasks=1 --cpus-per-task=2 --mem="$BARE_STAGE2_ASSEMBLE_MEMORY" \
    --time="$BARE_STAGE2_ASSEMBLE_TIME" --job-name=lmf-s2-assemble \
    --dependency="afterok:$normal_id:$pair_id" --kill-on-invalid-dep=yes \
    --output="$control_dir/logs/assemble-%j.out" --export=ALL \
    "$script_path" _assemble "$control_dir")"
  assemble_id="${assemble_raw%%;*}"
  printf 'assemble\tdiscovery_and_compact_mirror\t%s\n' "$assemble_id" >>"$control_dir/jobs.tsv"
  echo "submitted Stage 2 discovery run=$run_id prepare=$prep_id normal_reference=$normal0_id normal=$normal_id pair_reference=$pair0_id pair=$pair_id assembly=$assemble_id"
  echo "monitor: $script_path status $run_id"
}

submit_validation() {
  local requested="$1" control_dir
  control_dir="$(resolve_control_dir "$requested")"
  print_validation_plan "$control_dir"
  "$project_dir/slurm/phase0_calibrate_cpu.sh" plan >/dev/null
  require_command sbatch
  load_run "$control_dir"
  [[ ! -e "$BARE_STAGE2_FULL_DIR/stage2_validation.h5" ]] || die \
    "Stage 2 validation already exists: $BARE_STAGE2_FULL_DIR/stage2_validation.h5"
  ! awk -F'\t' '$1 == "validation_array" || $1 == "validation_assemble" {found=1} END {exit found ? 0 : 1}' \
    "$control_dir/jobs.tsv" || die "validation jobs are already recorded in $control_dir/jobs.tsv"
  mkdir -p "$BARE_STAGE2_FULL_DIR/validation"
  local validation_raw validation_id assemble_raw assemble_id
  validation_raw="$(sbatch --parsable --account="$BARE_STAGE2_ACCOUNT" --constraint=cpu --qos="$BARE_STAGE2_QOS" \
    --licenses=scratch,cfs --nodes=1 --ntasks=1 --cpus-per-task="$BARE_STAGE2_LOGICAL_CPUS" \
    --mem="$BARE_STAGE2_VALIDATION_MEMORY" --time="$BARE_STAGE2_VALIDATION_TIME" \
    --array=1-"$VALIDATION_PROBES" --job-name=lmf-s2-valid \
    --output="$control_dir/logs/validation-%A_%a.out" --export=ALL \
    "$script_path" _validation "$control_dir")"
  validation_id="${validation_raw%%;*}"
  printf 'validation_array\th_and_h2_modes_1_to_%s\t%s\n' "$VALIDATION_PROBES" "$validation_id" >>"$control_dir/jobs.tsv"
  assemble_raw="$(sbatch --parsable --account="$BARE_STAGE2_ACCOUNT" --constraint=cpu --qos="$BARE_STAGE2_QOS" \
    --licenses=scratch,cfs --nodes=1 --ntasks=1 --cpus-per-task=2 --mem="$BARE_STAGE2_ASSEMBLE_MEMORY" \
    --time="$BARE_STAGE2_ASSEMBLE_TIME" --job-name=lmf-s2-val-asm \
    --dependency="afterok:$validation_id" --kill-on-invalid-dep=yes \
    --output="$control_dir/logs/validation-assemble-%j.out" --export=ALL \
    "$script_path" _validation_assemble "$control_dir")"
  assemble_id="${assemble_raw%%;*}"
  printf 'validation_assemble\tlinearity_and_richardson\t%s\n' "$assemble_id" >>"$control_dir/jobs.tsv"
  echo "submitted Stage 2 validation run=$(basename "$control_dir") validation=$validation_id assembly=$assemble_id"
  echo "monitor: $script_path status $(basename "$control_dir")"
}

prepare_worker() {
  local control_dir="$1"
  load_run "$control_dir"
  export JULIA_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  srun --ntasks=1 --cpus-per-task=2 \
    "$BARE_STAGE2_JULIA" --startup-file=no --threads=1 --project="$project_dir" \
    "$project_dir/scripts/run_bare_stage2.jl" prepare "$BARE_STAGE2_CONFIG_COPY" \
    "$BARE_STAGE2_SOURCE_FULL_DIR/backbone.h5" "$BARE_STAGE2_SOURCE_FULL_DIR/stage1.h5" \
    "$BARE_STAGE2_FULL_DIR/candidate_bank.h5"
}

normal_reference_worker() {
  local control_dir="$1"
  load_run "$control_dir"
  set_thread_environment
  srun --ntasks=1 --cpus-per-task="$BARE_STAGE2_LOGICAL_CPUS" --cpu-bind=cores \
    "$BARE_STAGE2_JULIA" --startup-file=no --threads="$BARE_STAGE2_JULIA_THREADS" --project="$project_dir" \
    "$project_dir/scripts/run_bare_stage2.jl" normal-reference "$BARE_STAGE2_CONFIG_COPY" \
    "$BARE_STAGE2_FULL_DIR/candidate_bank.h5" "$BARE_STAGE2_FULL_DIR/normal_reference.h5"
}

normal_worker() {
  local control_dir="$1"
  load_run "$control_dir"
  [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || die "_normal requires a Slurm array task"
  set_thread_environment
  local output
  output="$(printf '%s/probes/normal_%03d.h5' "$BARE_STAGE2_FULL_DIR" "$SLURM_ARRAY_TASK_ID")"
  srun --ntasks=1 --cpus-per-task="$BARE_STAGE2_LOGICAL_CPUS" --cpu-bind=cores \
    "$BARE_STAGE2_JULIA" --startup-file=no --threads="$BARE_STAGE2_JULIA_THREADS" --project="$project_dir" \
    "$project_dir/scripts/run_bare_stage2.jl" probe "$BARE_STAGE2_CONFIG_COPY" \
    "$BARE_STAGE2_FULL_DIR/candidate_bank.h5" "$BARE_STAGE2_FULL_DIR/normal_reference.h5" \
    - normal "$SLURM_ARRAY_TASK_ID" "$output"
}

pair_reference_worker() {
  local control_dir="$1"
  load_run "$control_dir"
  set_thread_environment
  srun --ntasks=1 --cpus-per-task="$BARE_STAGE2_LOGICAL_CPUS" --cpu-bind=cores \
    "$BARE_STAGE2_JULIA" --startup-file=no --threads="$BARE_STAGE2_JULIA_THREADS" --project="$project_dir" \
    "$project_dir/scripts/run_bare_stage2.jl" pair-reference "$BARE_STAGE2_CONFIG_COPY" \
    "$BARE_STAGE2_FULL_DIR/candidate_bank.h5" "$BARE_STAGE2_FULL_DIR/normal_reference.h5" \
    "$BARE_STAGE2_FULL_DIR/pair_reference.h5"
}

pair_worker() {
  local control_dir="$1"
  load_run "$control_dir"
  [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || die "_pair requires a Slurm array task"
  set_thread_environment
  local output
  output="$(printf '%s/probes/pair_%03d.h5' "$BARE_STAGE2_FULL_DIR" "$SLURM_ARRAY_TASK_ID")"
  srun --ntasks=1 --cpus-per-task="$BARE_STAGE2_LOGICAL_CPUS" --cpu-bind=cores \
    "$BARE_STAGE2_JULIA" --startup-file=no --threads="$BARE_STAGE2_JULIA_THREADS" --project="$project_dir" \
    "$project_dir/scripts/run_bare_stage2.jl" probe "$BARE_STAGE2_CONFIG_COPY" \
    "$BARE_STAGE2_FULL_DIR/candidate_bank.h5" "$BARE_STAGE2_FULL_DIR/normal_reference.h5" \
    "$BARE_STAGE2_FULL_DIR/pair_reference.h5" \
    pair "$SLURM_ARRAY_TASK_ID" "$output"
}

assemble_worker() {
  local control_dir="$1"
  load_run "$control_dir"
  export JULIA_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  srun --ntasks=1 --cpus-per-task=2 \
    "$BARE_STAGE2_JULIA" --startup-file=no --threads=1 --project="$project_dir" \
    "$project_dir/scripts/run_bare_stage2.jl" assemble "$BARE_STAGE2_CONFIG_COPY" \
    "$BARE_STAGE2_FULL_DIR/candidate_bank.h5" "$BARE_STAGE2_FULL_DIR/probes" \
    "$BARE_STAGE2_FULL_DIR/stage2_discovery.h5"
  srun --ntasks=1 --cpus-per-task=2 \
    "$BARE_STAGE2_JULIA" --startup-file=no --threads=1 --project="$project_dir" \
    "$project_dir/scripts/compact_results.jl" "$BARE_STAGE2_FULL_DIR" \
    "$control_dir/stateless_results"
}

validation_worker() {
  local control_dir="$1"
  load_run "$control_dir"
  [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || die "_validation requires a Slurm array task"
  set_thread_environment
  local output
  output="$(printf '%s/validation/validation_%03d.h5' "$BARE_STAGE2_FULL_DIR" "$SLURM_ARRAY_TASK_ID")"
  srun --ntasks=1 --cpus-per-task="$BARE_STAGE2_LOGICAL_CPUS" --cpu-bind=cores \
    "$BARE_STAGE2_JULIA" --startup-file=no --threads="$BARE_STAGE2_JULIA_THREADS" --project="$project_dir" \
    "$project_dir/scripts/run_bare_stage2.jl" validate "$BARE_STAGE2_CONFIG_COPY" \
    "$BARE_STAGE2_FULL_DIR/candidate_bank.h5" "$BARE_STAGE2_FULL_DIR/stage2_discovery.h5" \
    "$BARE_STAGE2_FULL_DIR/normal_reference.h5" "$BARE_STAGE2_FULL_DIR/pair_reference.h5" \
    "$SLURM_ARRAY_TASK_ID" "$output"
}

validation_assemble_worker() {
  local control_dir="$1"
  load_run "$control_dir"
  export JULIA_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  srun --ntasks=1 --cpus-per-task=2 \
    "$BARE_STAGE2_JULIA" --startup-file=no --threads=1 --project="$project_dir" \
    "$project_dir/scripts/run_bare_stage2.jl" assemble-validation "$BARE_STAGE2_CONFIG_COPY" \
    "$BARE_STAGE2_FULL_DIR/stage2_discovery.h5" "$BARE_STAGE2_FULL_DIR/validation" \
    "$BARE_STAGE2_FULL_DIR/stage2_validation.h5" "$VALIDATION_PROBES"
  srun --ntasks=1 --cpus-per-task=2 \
    "$BARE_STAGE2_JULIA" --startup-file=no --threads=1 --project="$project_dir" \
    "$project_dir/scripts/compact_results.jl" "$BARE_STAGE2_FULL_DIR" \
    "$control_dir/stateless_results"
}

status_run() {
  require_command squeue
  local control_dir; control_dir="$(resolve_control_dir "${1:-}")"
  printf '%-18s %-32s %-14s %-16s\n' KIND LABEL JOB_ID STATE
  local kind label job_id state
  while IFS=$'\t' read -r kind label job_id; do
    [[ "$kind" == kind ]] && continue
    state="$(squeue --noheader --jobs="$job_id" --format='%T' 2>/dev/null | head -n1 || true)"
    if [[ -z "$state" ]] && command -v sacct >/dev/null 2>&1; then
      state="$(sacct -X -j "$job_id" --noheader --parsable2 --format=State 2>/dev/null | awk -F'|' 'NF {print $1; exit}' || true)"
    fi
    printf '%-18s %-32s %-14s %-16s\n' "$kind" "$label" "$job_id" "${state:-UNKNOWN}"
  done <"$control_dir/jobs.tsv"
}

show_run() {
  local control_dir; control_dir="$(resolve_control_dir "${1:-}")"
  load_run "$control_dir"
  if [[ -f "$control_dir/stateless_results/stage2_validation_summary.tsv" ]]; then
    cat "$control_dir/stateless_results/stage2_validation_summary.tsv"
  elif [[ -f "$control_dir/stateless_results/stage2_discovery_summary.tsv" ]]; then
    if [[ -f "$control_dir/stateless_results/stage2_discovery_gates.tsv" ]]; then
      cat "$control_dir/stateless_results/stage2_discovery_gates.tsv"
      echo
    fi
    cat "$control_dir/stateless_results/stage2_discovery_summary.tsv"
  else
    status_run "$control_dir"
  fi
}

case "${1:-}" in
  ""|plan) [[ -n "${2:-}" ]] || die "usage: $0 plan STAGE1_RUN"; print_plan "$2";;
  submit-discovery)
    [[ -n "${2:-}" && -n "${3:-}" ]] || die "usage: $0 submit-discovery RUN_ID STAGE1_RUN"
    submit_discovery "$2" "$3";;
  plan-validation) [[ -n "${2:-}" ]] || die "usage: $0 plan-validation RUN_ID"; print_validation_plan "$2";;
  submit-validation) [[ -n "${2:-}" ]] || die "usage: $0 submit-validation RUN_ID"; submit_validation "$2";;
  status) status_run "${2:-}";;
  show) show_run "${2:-}";;
  _prepare) prepare_worker "$2";;
  _normal_reference) normal_reference_worker "$2";;
  _normal) normal_worker "$2";;
  _pair_reference) pair_reference_worker "$2";;
  _pair) pair_worker "$2";;
  _assemble) assemble_worker "$2";;
  _validation) validation_worker "$2";;
  _validation_assemble) validation_assemble_worker "$2";;
  *) die "usage: $0 {plan STAGE1_RUN|submit-discovery RUN_ID STAGE1_RUN|plan-validation RUN_ID|submit-validation RUN_ID|status [RUN_ID]|show [RUN_ID]}";;
esac
