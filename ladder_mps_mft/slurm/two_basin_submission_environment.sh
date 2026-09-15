#!/bin/bash
# Sourced by the two September 15 entry points; no scheduler action here.
new_project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
original_project="${TWO_BASIN_ORIGINAL_PROJECT:-${CFS:?CFS must be set}/m4863/MPS-MFT/ladder_mps_mft}"
anchor_env="$original_project/output/phase1_gpu/20260908_square_two_basin_95_5_80_anchors/run.env"
[[ -f "$anchor_env" ]] || { echo "error: missing original accounting environment: $anchor_env" >&2; exit 1; }
set -a
# shellcheck disable=SC1090
source "$anchor_env"
set +a
# These describe the old campaign, not shared account settings. Let the new
# launcher stamp its own version and scratch run directory during preparation.
unset PHASE1_RUN_SCRIPT_VERSION PHASE1_RUN_SCRATCH_DIR
export PHASE1_PROJECT_DIR="$new_project"
export PHASE1_REPO_ROOT="$(cd "$new_project/.." && pwd)"
export PHASE1_MAX_SEGMENTS=1
launcher="$new_project/slurm/phase1_gpu.sh"
