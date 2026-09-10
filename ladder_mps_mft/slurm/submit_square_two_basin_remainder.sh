#!/bin/bash
# User-run Perlmutter entry point: 14 remaining 95%/5%, chi=200, raw 40-step starts.
# Run from a separate checkout while the original anchors are still queued.
set -euo pipefail
(( $# <= 1 )) || { echo "usage: bash $0 [NEW_RUN_ID]" >&2; exit 1; }
run_id="${1:-20260910_square_two_basin_95_5_40_remainder}"
new_project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
original_project="${TWO_BASIN_ORIGINAL_PROJECT:-${CFS:?CFS must be set}/m4863/MPS-MFT/ladder_mps_mft}"
anchor_env="$original_project/output/phase1_gpu/20260908_square_two_basin_95_5_80_anchors/run.env"
[[ -f "$anchor_env" ]] || { echo "error: missing anchor environment: $anchor_env" >&2; exit 1; }
original_project="$(cd "$original_project" && pwd)"
[[ "$new_project" != "$original_project" ]] || {
  echo "error: use a separate checkout so pending anchors retain their original source" >&2; exit 1;
}

# Reuse the exact account, scratch, run, and shared budget settings of the
# anchors. Only the new campaign's source and base config are replaced.
set -a
# shellcheck disable=SC1090
source "$anchor_env"
set +a
export PHASE1_PROJECT_DIR="$new_project"
export PHASE1_REPO_ROOT="$(cd "$new_project/.." && pwd)"
export PHASE1_TWO_BASIN_CONFIG="$new_project/configs/phase1_gpu_square_two_basin_chi200_raw40.toml"
launcher="$new_project/slurm/phase1_gpu.sh"
bash "$launcher" prepare-square-two-basin-raw "$new_project/data/two_basin_references.h5" "$run_id" remainder
bash "$launcher" reconcile
exec bash "$launcher" submit "$run_id"
