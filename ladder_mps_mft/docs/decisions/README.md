# Established decision index

This directory indexes decisions already established elsewhere. The linked
document remains canonical; this page prevents a new task from reopening old
questions merely because their rationale is distributed across the project.

| Decision | Status | Canonical source |
|---|---|---|
| Codex never connects to Perlmutter or operates transfers/Slurm; the user is the operator | accepted | `AGENTS.md` and repository-root `AGENTS.md` |
| Full MPS artifacts are scratch-first; CFS/local copies are compact stateless mirrors | accepted | `docs/PERLMUTTER_STORAGE.md` |
| Phase 1 production uses Float64 dense CUDA; the Float32 v2 campaign is screening/warm-start evidence only | accepted | `docs/DEVICE_HANDOFF_2026-08-25.md`, `docs/PHASE1_V2_AUDIT.md`, and `docs/PHASE1_V3_AUDIT.md` |
| Preserve the complete MF history, including time-zero seed and separate physical orbit phases | accepted | `docs/PERLMUTTER_STORAGE.md`, `docs/CONVERGENCE.md`, and `docs/PROVENANCE_AND_SELECTION.md` |
| Probe raw-map recurrence before Anderson acceleration | accepted | `docs/CONVERGENCE.md` |
| Rank only accepted states with matching fingerprints through the canonical functional including double counting | accepted | `docs/VARIATIONAL_FUNCTIONAL.md` and `docs/PROVENANCE_AND_SELECTION.md` |
| Never rank energies across transverse geometries or different Hamiltonian points | accepted | `docs/VARIATIONAL_FUNCTIONAL.md` |
| Maintain an append-only 400-additional-node-hour project control and reconcile terminal jobs using Perlmutter accounting | accepted | `AGENTS.md`, `docs/PERLMUTTER_PHASE1_GPU.md`, and `docs/RUN_LOG.md` |
| Preserve legacy field inheritance as explicit lineage; never relabel it as an independent seed | accepted | `docs/SEEDING.md` and `docs/PROVENANCE_AND_SELECTION.md` |
| Keep future compute available for bond-dimension and length convergence | active planning constraint | `docs/PHASES_0_TO_4.md` and `docs/PROJECT_STATE.md` |

Create a separate decision record only when a genuinely new choice needs a
stable rationale, alternatives, consequences, and date. Routine run status
belongs in `docs/PROJECT_STATE.md` and `docs/RUN_LOG.md`, not here.

