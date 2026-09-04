# Current project state

Last locally reviewed: 2026-09-04

This is the canonical mutable snapshot for resuming work. It is deliberately
short. Stable rules belong in `AGENTS.md` and the method documents; durable
history belongs in `docs/RUN_LOG.md`.

## Repository snapshot

- Branch: `codex/mps-mft-phase0-refactor`
- Baseline commit before this continuity update: `c1a853f` (`Add legacy stripe
  comparison workflow`)
- Before this continuity update, the working tree was clean inside
  `ladder_mps_mft/`; the current in-scope diff is this documentation patch. An
  untracked root `.claude/` directory exists outside this subproject and is not
  part of this work.
- Local `output/` and all HDF5 files are intentionally excluded from Git.

Recheck the branch, commit, and working tree at the start of each new task.

## Current scientific position

- Phase 0 is closed. Phase 1 uses the refactored Float64 dense-CUDA path.
- At square `(t0,V)=(1.4,0)`, the completed loose `chi=200` six-seed campaign
  produced six accepted period-one endpoints that are qualitatively close and
  pairing-bearing. Small remaining differences are not a thermodynamic phase
  claim and remain subject to tighter tolerance, bond-dimension, length, and
  error-budget checks.
- The one-shot legacy-field DMRG result is a selection-ineligible diagnostic,
  not an accepted SCF endpoint. It motivates a matched `chi=400` two-lineage
  comparison.
- The current surveys fill the square and cubic-unfrustrated `3 x 3`
  `(t0,V)` grids using one center-of-mass-uniform smooth-pairing access seed.
  They test coverage and basin access at loose `chi=200`; they do not prove
  basin uniqueness.
- A separate square `(1.4,-0.4)` comparison tests whether the inherited
  high-amplitude legacy stripe remains a self-consistent competing basin.

## Campaign inventory

These are the newest locally documented campaigns. Their live scheduler state
must not be inferred from the repository.

| Run ID | Local record | Question |
|---|---|---|
| `20260903_phase1_square_t014_v000_pairing_legacy_chi400_tight` | prepared workflow | Pairing versus legacy-like lineage at `chi=400` |
| `20260903_phase1_square_grid_smooth_pairing_chi200_loose` | prepared/submission handoff documented | Fill five missing square-grid points |
| `20260903_phase1_cubic_unfrustrated_grid_smooth_pairing_chi200_loose` | prepared/submission handoff documented | Fill eight missing cubic-unfrustrated points |
| `20260903_phase1_square_t014_vm04_legacy_stripe_compare_chi200_loose` | prepared workflow | Test inherited stripe stability and, conditionally, its energy against the paired control |

Campaign contracts and exact run commands are in the corresponding dated
documents linked from `docs/plans/ACTIVE.md`.

## Live Perlmutter and accounting boundary

- User-reported on 2026-09-04: the three latest jobs remain `PENDING` on
  Perlmutter.
- Exact job IDs and run membership were not provided with that report and are
  not verified by the local checkout. Do not guess them, resubmit them, or
  cancel them based on this file.
- The hard project control remains 400 additional node-hours. The last locally
  documented reconciled active total was `42.024097222` node-hours before the
  newest campaign reservations. It is historical, not a current allowance.
- The live append-only Perlmutter reservation and reconciliation ledgers plus
  `sacct` measurements are authoritative before any compute decision.
- Preserve substantial budget for later bond-dimension and length convergence;
  requested first-segment ceilings are not authorization for blanket
  continuations.

## Exact next action

Wait for the currently pending jobs. Do not submit duplicates or cancel them
from this repository state. After the user reports terminal status and
synchronizes the relevant compact results, logs, manifests, `jobs.tsv`, and
ledger snapshots:

1. verify the compact/stateless artifacts locally without requesting full
   scratch verification;
2. classify convergence using the raw-map recurrence, oscillation, slow-mode,
   density, and energy-stability gates;
3. rank only accepted same-geometry, same-Hamiltonian states with matching
   fingerprints through the canonical variational energy;
4. update this file and append the evidence and decision to `docs/RUN_LOG.md`;
5. choose any continuation or higher-`chi` calculation only after reconciling
   the authoritative ledger.

## Verification boundaries

- Local compact verification does not prove the recorded full scratch artifact
  is present or hash-valid.
- Scheduler and accounting claims require current user-provided Perlmutter
  output or a synchronized ledger snapshot.
- Accepted finite-`L`, finite-`chi` Phase 1 endpoints are variational states,
  not thermodynamic phase assignments.
- Never compare canonical energies across transverse geometries or different
  Hamiltonian points.
