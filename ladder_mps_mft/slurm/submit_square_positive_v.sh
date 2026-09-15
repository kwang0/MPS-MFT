#!/bin/bash
# User-run Perlmutter entry point: four square (t0,V)=(1.2,+0.2) seeds.
set -euo pipefail
(( $# <= 1 )) || { echo "usage: bash $0 [NEW_RUN_ID]" >&2; exit 1; }
run_id="${1:-20260915_square_t012_vp02_four_seeds_60}"
source "$(dirname "${BASH_SOURCE[0]}")/two_basin_submission_environment.sh"
[[ "$(cd "$new_project" && pwd -P)" != "$(cd "$original_project" && pwd -P)" ]] || {
  echo "error: use a separate checkout to preserve submitted campaigns" >&2
  exit 1
}
export PHASE1_GPU_TIME=12:00:00
export PHASE1_POSITIVE_V_CONFIG="$new_project/configs/phase1_gpu_square_positive_v_chi200_raw60.toml"
bash "$launcher" prepare-square-positive-v "$new_project/data/two_basin_references.h5" "$run_id"
bash "$launcher" reconcile
exec bash "$launcher" submit "$run_id"
