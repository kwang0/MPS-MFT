#!/bin/bash
# User-run Perlmutter entry point: six square coordinates, two seeds each.
set -euo pipefail
(( $# <= 1 )) || { echo "usage: bash $0 [NEW_RUN_ID]" >&2; exit 1; }
run_id="${1:-20260915_square_two_basin_fine_cuts_95_5_60}"
source "$(dirname "${BASH_SOURCE[0]}")/two_basin_submission_environment.sh"
[[ "$(cd "$new_project" && pwd -P)" != "$(cd "$original_project" && pwd -P)" ]] || {
  echo "error: use a separate checkout; the submitted cubic jobs must retain their original solver source" >&2
  exit 1
}
export PHASE1_GPU_TIME=12:00:00
export PHASE1_TWO_BASIN_FINE_CUTS_CONFIG="$new_project/configs/phase1_gpu_square_two_basin_fine_cuts_chi200_raw60.toml"
bash "$launcher" prepare-square-two-basin-fine-cuts "$new_project/data/two_basin_references.h5" "$run_id"
bash "$launcher" reconcile
exec bash "$launcher" submit "$run_id"
