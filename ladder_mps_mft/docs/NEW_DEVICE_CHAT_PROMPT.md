# First prompt for the other Codex desktop

Copy the text below into a new Codex task opened at the `MPS-MFT` repository
root after pulling `codex/mps-mft-phase0-refactor`.

---

Continue the ladder MPS+MF project from the cross-device handoff. Work only
inside `ladder_mps_mft/` unless I explicitly expand the scope.

Before taking action, read these files completely and in this order:

1. `ladder_mps_mft/AGENTS.md`
2. `ladder_mps_mft/docs/DEVICE_HANDOFF_2026-08-25.md`
3. `ladder_mps_mft/docs/PHASE1_V3_AUDIT.md`
4. the latest section of `ladder_mps_mft/docs/RUN_LOG.md`
5. `ladder_mps_mft/docs/PERLMUTTER_STORAGE.md`
6. `ladder_mps_mft/docs/CONVERGENCE.md`
7. `ladder_mps_mft/docs/VARIATIONAL_FUNCTIONAL.md`
8. `ladder_mps_mft/docs/PROVENANCE_AND_SELECTION.md`
9. `ladder_mps_mft/docs/PHASES_0_TO_4.md`

Then inspect the current branch, commit, and working tree. Check whether the v2
and v3 `stateless_results` directories described by the handoff exist locally.
If they do, run the compact-only stateless verifiers and reproduce the v3 audit
into a new output directory. Do not use `--full` on this Mac, because the
Perlmutter scratch paths are not mounted.

Give me a concise readiness report containing:

- the checked-out branch/commit and whether the working tree is clean;
- which lightweight data are present and what was actually verified;
- the v3 accepted/candidate count and the authorized same-geometry rankings;
- the unresolved period-two issue and the distinction between raw-map physics
  and Anderson fixed-point acceleration;
- the full-artifact, Perlmutter-accounting, and scientific-convergence
  verification boundaries; and
- your recommended exact next calculation and its plan-only cost estimate.

Do not submit or cancel jobs, change the budget ledger, migrate/prune/delete
data, overwrite immutable HDF5 files, or claim a thermodynamic phase in this
first turn. Perlmutter measurements and accounting are authoritative. Preserve
physical periodic orbit phases separately, rank only accepted states with
matching fingerprints through the canonical variational energy including
double-counting terms, and never compare energies across transverse
geometries. Do not re-litigate the established scratch-first/stateless-mirror,
Float64-CUDA, complete-history, legacy field-inheritance, or 400-additional-
node-hour decisions without new evidence.

---
