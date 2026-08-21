# Operating rules for this subproject

1. Keep implementation and generated run controls inside `ladder_mps_mft/`. Do not modify legacy files as part of this migration unless the user explicitly expands scope.
2. Treat Perlmutter measurements and accounting as authoritative. Local smoke tests establish code-path validity only.
3. Run `slurm/phase0_calibrate_cpu.sh plan` before `submit`; do not bypass its node-hour cap. The QSL Project B CPU winner is prior information, not a ladder-DMRG result.
4. Never enable BLAS, Strided, and block-sparse threading together in a calibration candidate.
5. Preserve immutable `state.h5`, Phase 0 seed, cycle, and metric artifacts. A parent or resume path must be accompanied by its SHA-256.
6. Label independent seeds, continuations, scan direction, and recurrence status explicitly. Do not relabel a periodic orbit as a fixed point or average its members into one.
7. Rank branches only through `canonical_variational_energy` after the fixed-point and fingerprint gates pass. Never rank saved effective-Hamiltonian eigenvalues directly.
8. Keep `docs/RUN_LOG.md` append-only. Record commands, job IDs, hashes, validation boundaries, failures, and decisions before handing work to another chat or collaborator.
9. Run the test suite after source changes. Record whether validation was unit, local DMRG smoke, Phase 0 timing, or scientific convergence.
