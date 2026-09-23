#!/bin/bash
# User-run Perlmutter entry point: exactly four independent intertwined starts.
set -euo pipefail
(( $# <= 1 )) || { echo "usage: bash $0 [NEW_RUN_ID]" >&2; exit 1; }
run_id="${1:-20260923_trellis_two_ladder_intertwined_lambda16_60}"
source "$(dirname "${BASH_SOURCE[0]}")/two_basin_submission_environment.sh"
square_tp_project="${TWO_BASIN_SQUARE_TP_PROJECT:-$(dirname "$original_project")-square-tp-20260922/ladder_mps_mft}"
for protected_project in "$original_project" "$square_tp_project"; do
  if [[ -d "$protected_project" && "$(cd "$new_project" && pwd -P)" == "$(cd "$protected_project" && pwd -P)" ]]; then
    echo "error: use a separate checkout; ongoing trellis and square t_perp jobs must retain their source" >&2
    exit 1
  fi
done
# Preserve both existing latest-run pointers and keep the same locked budget.
export PHASE1_RUN_ROOT="$PHASE1_RUN_ROOT/trellis_intertwined"
export PHASE1_GPU_TIME=16:00:00
export PHASE1_GPU_CPUS=32
export PHASE1_TRELLIS_INTERTWINED_CONFIG="$new_project/configs/phase1_gpu_trellis_intertwined_chi200_raw60.toml"
bash "$launcher" prepare-trellis-intertwined "$new_project/data/positive_v_intertwined_recipe.toml" "$run_id"
bash "$launcher" reconcile
exec bash "$launcher" submit "$run_id"
