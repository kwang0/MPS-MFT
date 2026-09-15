#!/bin/bash
# User-run Perlmutter entry point: all 18 cubic 95%/5%, chi=200 raw starts.
set -euo pipefail
(( $# <= 1 )) || { echo "usage: bash $0 [NEW_RUN_ID]" >&2; exit 1; }
run_id="${1:-20260915_cubic_unfrustrated_two_basin_95_5_60}"
source "$(dirname "${BASH_SOURCE[0]}")/two_basin_submission_environment.sh"
export PHASE1_GPU_TIME=12:00:00
export PHASE1_CUBIC_TWO_BASIN_CONFIG="$new_project/configs/phase1_gpu_cubic_unfrustrated_two_basin_chi200_raw60.toml"
bash "$launcher" prepare-cubic-unfrustrated-two-basin-raw "$new_project/data/two_basin_references.h5" "$run_id"
bash "$launcher" reconcile
exec bash "$launcher" submit "$run_id"
