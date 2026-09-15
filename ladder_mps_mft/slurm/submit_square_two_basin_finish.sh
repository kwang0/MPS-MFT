#!/bin/bash
# User-run Perlmutter entry point: 20 more raw updates for each square V=0 seed.
set -euo pipefail
(( $# <= 1 )) || { echo "usage: bash $0 [NEW_RUN_ID]" >&2; exit 1; }
run_id="${1:-20260915_square_t014_v000_two_basin_finish20}"
source "$(dirname "${BASH_SOURCE[0]}")/two_basin_submission_environment.sh"
export PHASE1_GPU_TIME=08:00:00
export PHASE1_SQUARE_TWO_BASIN_FINISH_CONFIG="$new_project/configs/phase1_gpu_square_two_basin_finish20.toml"
source_run="$PHASE1_RUN_ROOT/20260908_square_two_basin_95_5_80_anchors"
bash "$launcher" prepare-square-two-basin-finish "$source_run" "$run_id"
bash "$launcher" reconcile
exec bash "$launcher" submit "$run_id"
