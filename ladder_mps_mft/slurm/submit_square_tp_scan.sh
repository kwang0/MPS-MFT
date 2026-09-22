#!/bin/bash
# User-run Perlmutter entry point: eight fresh square t_perp scan starts.
set -euo pipefail
(( $# <= 1 )) || { echo "usage: bash $0 [NEW_RUN_ID]" >&2; exit 1; }
run_id="${1:-20260922_square_t014_tp_scan_95_5_60}"
source "$(dirname "${BASH_SOURCE[0]}")/two_basin_submission_environment.sh"
[[ "$(cd "$new_project" && pwd -P)" != "$(cd "$original_project" && pwd -P)" ]] || {
  echo "error: use a separate checkout; ongoing trellis jobs must retain their original source" >&2
  exit 1
}
# Keep the shared accounting, but do not replace the trellis latest-run pointer.
export PHASE1_RUN_ROOT="$PHASE1_RUN_ROOT/square_tp_scan"
export PHASE1_GPU_TIME=16:00:00
export PHASE1_GPU_CPUS=32
export PHASE1_SQUARE_TP_SCAN_CONFIG="$new_project/configs/phase1_gpu_square_tp_scan_chi200_raw60.toml"
bash "$launcher" prepare-square-tp-scan "$new_project/data/two_basin_references.h5" "$run_id"
bash "$launcher" reconcile
exec bash "$launcher" submit "$run_id"
