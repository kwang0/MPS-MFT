# Documentation map

This page is the durable entry point for a new Codex task, workstation, or
collaborator. It points to the current project state without duplicating the
scientific and operational rules owned elsewhere.

## Start here

For substantive work inside `ladder_mps_mft/`, read in this order:

1. `AGENTS.md` for durable operating and scientific rules.
2. `docs/PROJECT_STATE.md` for the dated, mutable current snapshot.
3. `docs/ARCHITECTURE.md` for the code, data, and host-boundary map.
4. `docs/plans/ACTIVE.md` for the current completion plan.
5. The campaign-specific or method document relevant to the request.
6. Only the latest relevant section of append-only `docs/RUN_LOG.md`.

Then inspect the current Git branch, commit, and working tree. A live user
report or current Perlmutter output supersedes a dated scheduler or accounting
snapshot in this repository.

## Document roles and authority

| Document | Role | Update policy |
|---|---|---|
| `AGENTS.md` | Compact, durable rules that apply to every task | Change only when a lasting rule changes |
| `docs/PROJECT_STATE.md` | Canonical current snapshot and next action | Replace stale status after meaningful changes |
| `docs/ARCHITECTURE.md` | Stable code/data/workflow map | Update when structure or ownership changes |
| `docs/decisions/README.md` | Index of established decisions and their canonical sources | Add or redirect entries; do not duplicate the source rationale |
| `docs/plans/ACTIVE.md` | Current campaign completion plan | Keep short; move durable outcomes to the run log and campaign docs |
| `docs/RUN_LOG.md` | Append-only history of commands, evidence, failures, and decisions | Append only |
| Dated campaign documents | Scientific question, numerical contract, cost envelope, and interpretation boundary | Preserve as the campaign record |
| `docs/DEVICE_HANDOFF_2026-08-25.md` | Historical cross-device snapshot | Do not treat as current state |

Perlmutter measurements, scheduler state, full-artifact checks, and accounting
are authoritative over local mirrors. User-reported live state should be
recorded explicitly as user-reported until synchronized evidence is available.

## Stable scientific references

The latest synchronized results are in the
[September 18 combined campaign review](reports/campaign_review_20260918/README.md).
It brings together all 18 cubic starts, all 12 finer square starts, three
completed positive-V square seeds and the first completed trellis run,
with histories, spatial profiles, convergence diagnostics and cost evidence.
The finer-square report also plots full canonical variational energies against
both cut parameters, with common-background and interval-slope diagnostics.
All34 completed histories remain unaccepted. Cubic is stripe-like throughout;
two finer square points retain distinct seed-dependent textures; positive-V
square has not retained pairing in its three completed starts; the trellis
one-ladder stripe seed instead develops pairing. Partial logs remain distinct
from completed spatial outcomes.

Detailed results:

- [Complete cubic grid](reports/cubic_two_basin_grid_20260918/README.md).
- [Finer square transition cuts](reports/square_fine_cuts_20260918/README.md).
- [Square (1.2,+0.2)](reports/square_positive_v_20260918/README.md).
- [First trellis result and partial logs](reports/trellis_progress_20260918/README.md).
- [Coarse square grid with both continuations](reports/two_basin_grid_20260915/README.md):
  seven stripe and two paired coordinates,902 evaluations, zero accepted
  endpoints and 27.350764 actual node-hours across18 lineages/20 source jobs.
- [September 16 partial progress](reports/two_basin_progress_20260916/README.md),
  retained as a historical snapshot.

Preparation contracts remain in the
[trellis campaign](reports/trellis_comparison_20260916/README.md),
[positive-V seeds](reports/square_positive_v_seeds_20260915/README.md),
[finer cuts and bare-ladder interpolation check](reports/two_basin_fine_cuts_20260915/README.md),
and [cubic/continuation launchers](reports/two_basin_next_campaigns_20260915/README.md).
They are not instructions to resubmit the already recorded jobs. The
[trellis method contract](TRELLIS_MEAN_FIELD.md) specifies reciprocal maps,
energy normalization and the one-/two-ladder spatial ansatz distinction.

For manuscript background, see the
[introduction and results draft](manuscript/README.md), created September 15
and now updated in LaTeX and PDF with numerical evidence through September 18.
It includes the complete grids, full transition-cut energies, positive-V and
trellis outcomes, while retaining its 34 references and provisional status.
The [coupled-ladder literature review](literature/README.md) provides annotated
sources, LaTeX/PDF and 49 references. Its actual project-evidence section now
qualifies the legacy positive-V interpretation with the new square data,
without claiming a fresh literature search. The [living methods notes](METHODS_NOTES.tex)
([compiled PDF](METHODS_NOTES.pdf))
also describe the raw-update protocol and energy-shape interpretation.

The deferred [finite-size seed snapshot](reports/finite_size_seeds_20260906/README.md)
retains four chi=200 L=96/L=128 pairing/stripe seeds with fixed L=64 coupling.

For the preceding V=0 anchor result, see the
[September 12 V=0 anchor analysis](reports/two_basin_v000_20260912/README.md).
The stripe start reaches 80 evaluations with negligible pairing; the pairing
start reaches a deadline at 62 while spin grows and pairing collapses. Both
remain unaccepted for resolved field drift, not merely noise-floor checks.

For the paired control, see the
[September 10 V=-0.4 anchor analysis](reports/two_basin_vm04_20260910/README.md).
Both 80-step chi=200 lineages reach a common paired plateau; the report explains
the weak-channel acceptance failures and records 3.763125 actual node-hours.
The [September 10 remainder contract](reports/two_basin_remainder_20260910/README.md)
prepares the other fourteen starts with a 40-step cap, qualified channel noise
handling, and a separate-checkout submission that preserves pending anchors.

For the square coverage results, see the
[September 8 square chi=200 grid](reports/square_grid_20260908/README.md).
It provides a single-file 3 x 3 data bundle, legacy-style Fourier grids,
explicit selection provenance, and analysis of the flagged diverging point.
The [basin follow-up](reports/square_grid_20260908/BASIN_ASSESSMENT.md) records
SDW growth, the reproduced Anderson jump, and the proposed paired/striped
seed comparison; loose acceptance does not establish basin stability.
The [approved raw two-basin campaign](reports/two_basin_raw_20260908/README.md)
supplies all 18 square seeds, tighter chi=200 controls, energy -history tools,
and `slurm/submit_square_two_basin.sh` for four-anchor-first submission after
`git pull`, with versioned reference correlations and Anderson disabled.
For the preceding bond-dimension comparison, see the
[September 5 chi=400 comparison](reports/chi400_comparison_20260905/ANALYSIS.md).
It records a provisional striped-endpoint energy advantage and the unresolved
acceptance gates for both lineages.

For the latest cross-project assessment, see the
[September 4 systematic review](reports/systematic_review_20260904/REVIEW.md).
It records recommendations and local evidence, with reproducible supporting
files alongside it; the operating rules and live accounting authority remain
in the documents below.

- Phase plan: `docs/PHASES_0_TO_4.md`
- Convergence and recurrence: `docs/CONVERGENCE.md`
- Canonical energy: `docs/VARIATIONAL_FUNCTIONAL.md`
- Provenance and accepted-only selection: `docs/PROVENANCE_AND_SELECTION.md`
- Scratch-first storage and stateless mirrors: `docs/PERLMUTTER_STORAGE.md`
- Seed protocols: `docs/SEEDING.md`
- Numerical error budget: `docs/PHASE1_NUMERICAL_ERROR_BUDGET.md`
- Publication gates: `docs/LITERATURE_AND_PUBLICATION_GATES.md`
- Phase 1 operator workflow: `docs/PERLMUTTER_PHASE1_GPU.md`

## Cross-system handoff checklist

1. Synchronize code through Git without discarding uncommitted changes.
2. Synchronize only the intended compact results, logs, manifests, and ledger
   snapshots; full restartable MPS artifacts remain on Perlmutter scratch.
3. Update `docs/PROJECT_STATE.md` with the branch, commit, evidence boundary,
   user-reported or verified live status, and exact next action.
4. Append the durable event and validation boundary to `docs/RUN_LOG.md`.
5. Start the new task with `docs/NEW_DEVICE_CHAT_PROMPT.md`.
