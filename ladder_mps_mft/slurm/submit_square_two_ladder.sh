#!/bin/bash
# User-run Perlmutter handoff: two seeds at each of the two paired square points.
set -euo pipefail
(( $# <= 1 )) || { echo "usage: bash $0 [NEW_RUN_ID]" >&2; exit 1; }
run_id="${1:-20260920_square_two_ladder_two_basin_60}"
source "$(dirname "${BASH_SOURCE[0]}")/two_basin_submission_environment.sh"
# Keep the 11.5-hour SCF deadline; leave 4.5 hours for both terminal MPS measurements.
export PHASE1_GPU_TIME=16:00:00
export PHASE1_SQUARE_TWO_LADDER_CONFIG="$new_project/configs/phase1_gpu_square_two_ladder_chi200_raw60.toml"
bash "$launcher" prepare-square-two-ladder "$new_project/data/two_basin_references.h5" "$run_id"
bash "$launcher" reconcile
exec bash "$launcher" submit "$run_id"
