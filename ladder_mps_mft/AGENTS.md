# Operating rules for this subproject

1. Keep implementation and generated run controls inside `ladder_mps_mft/`. Do not modify legacy files as part of this migration unless the user explicitly expands scope.
2. Treat Perlmutter measurements and accounting as authoritative. Local smoke tests establish code-path validity only.
3. Never authenticate to, log in to, synchronize with, or operate the scheduler
   on NERSC/Perlmutter from Codex. The user always performs transfers,
   submission, status/accounting checks, continuation, and cancellation.
   Prepare and validate locally, then hand the user exact Perlmutter commands.
4. Run `slurm/phase0_calibrate_cpu.sh plan` before `submit`; do not bypass its node-hour cap. The QSL Project B CPU winner is prior information, not a ladder-DMRG result.
5. Never enable BLAS, Strided, and block-sparse threading together in a calibration candidate.
6. Preserve immutable `state.h5`, Phase 0 seed, cycle, and metric artifacts. A parent or resume path must be accompanied by its SHA-256.
7. Label independent seeds, continuations, scan direction, and recurrence status explicitly. Do not relabel a periodic orbit as a fixed point or average its members into one.
8. Rank branches only through the stored solution canonical energy after fixed-point or unmixed-orbit gates and fingerprint checks pass. Use the orbit phase average for periodic solutions. Never rank saved effective-Hamiltonian eigenvalues directly.
9. Keep `docs/RUN_LOG.md` append-only. Record commands, job IDs, hashes, validation boundaries, failures, and decisions before handing work to another chat or collaborator.
10. After source changes, use focused tests during development. Run the full
    suite at most once before handoff and only when the affected source and
    risk justify it. Do not rerun unrelated DMRG tests for documentation,
    report, configuration, launcher-resource, or similarly narrow changes.
    Record whether validation was unit, local DMRG smoke, Phase 0 timing, or
    scientific convergence.
11. Treat an already-implemented workflow as existing work to reuse. Do not
    broaden a report, analysis, or submission-script request into a source
    redesign or comprehensive audit without first explaining the added scope,
    expected benefit, and likely wall-clock cost and obtaining user approval,
    unless immediate correctness or safety requires the change.
12. Before starting any local check expected to exceed five minutes, tell the
    user what is being run and why. Avoid repeating unchanged expensive tests;
    rerun only the focused checks affected by a late change.
13. Before substantive work, read `docs/README.md`,
    `docs/PROJECT_STATE.md`, and the relevant active-plan and method documents;
    then inspect the current Git state and only the latest relevant section of
    `docs/RUN_LOG.md`.
14. Treat `docs/PROJECT_STATE.md` as a dated local snapshot, not live scheduler
    or accounting evidence. A current user report or current Perlmutter output
    supersedes it; record the source and verification boundary explicitly.
15. After a meaningful campaign, code, accounting, or scientific-status
    change, update `docs/PROJECT_STATE.md` and append the durable evidence and
    decision to `docs/RUN_LOG.md`. Do not copy long run histories into the
    current-state file.
