#!/bin/bash
# User-run Perlmutter entry point for the 95%/5%, chi=200, 80-step campaign.
set -euo pipefail

(( $# <= 2 )) || { echo "usage: bash $0 [NEW_RUN_ID] [anchors|remainder|grid]" >&2; exit 1; }
stage="${2:-anchors}"
case "$stage" in
  anchors|remainder|grid) ;;
  *) echo "error: stage must be anchors, remainder, or grid" >&2; exit 1;;
esac
run_id="${1:-20260908_square_two_basin_95_5_80_${stage}}"

# Use this git checkout even if an earlier snapshot handoff exported another
# source path. Keep any explicit shared run, scratch, and accounting locations.
export PHASE1_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PHASE1_REPO_ROOT="$(cd "$PHASE1_PROJECT_DIR/.." && pwd)"
export PHASE1_TWO_BASIN_CONFIG="$PHASE1_PROJECT_DIR/configs/phase1_gpu_square_two_basin_chi200_raw.toml"
launcher="$PHASE1_PROJECT_DIR/slurm/phase1_gpu.sh"
references="$PHASE1_PROJECT_DIR/data/two_basin_references.h5"

# Preparation validates hashes and refuses an existing run ID. Reconciliation
# releases only sacct-confirmed unused reservations, including canceled jobs.
bash "$launcher" prepare-square-two-basin-raw "$references" "$run_id" "$stage"
ledger="${PHASE1_LEDGER_PATH:-${PHASE1_BUDGET_ROOT:-$PHASE1_PROJECT_DIR/output/project_budget}/additional_node_hours.tsv}"
if [[ -f "$ledger" ]] && awk 'NR > 1 && NF {found=1; exit} END {exit !found}' "$ledger"; then
  bash "$launcher" reconcile
fi
exec bash "$launcher" submit "$run_id"
