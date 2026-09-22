#!/bin/bash
# User-run Perlmutter entry point: repeat the four trellis starts at V=-1.
set -euo pipefail
(( $# <= 1 )) || { echo "usage: bash $0 [NEW_RUN_ID]" >&2; exit 1; }
run_id="${1:-20260922_trellis_vm1_two_basin_comparison_60}"
source "$(dirname "${BASH_SOURCE[0]}")/two_basin_submission_environment.sh"
# Same 11.5-hour SCF deadline; allow 4.5 hours for terminal measurements.
export PHASE1_GPU_TIME=16:00:00
export PHASE1_TRELLIS_CONFIG="$new_project/configs/phase1_gpu_trellis_vm1_chi200_raw60.toml"
bash "$launcher" prepare-trellis-comparison "$new_project/data/two_basin_references.h5" "$run_id"
bash "$launcher" reconcile
exec bash "$launcher" submit "$run_id"
