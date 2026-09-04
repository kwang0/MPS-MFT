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

