# Run log

Append new entries; do not rewrite prior entries. Distinguish code validation, timing calibration, and scientific convergence.

## 2026-08-21 — isolated refactor and local Phase 0 preparation

- Git branch: `codex/mps-mft-phase0-refactor`.
- Scope: all new implementation files are under `ladder_mps_mft/`; pre-existing `plot_ladder_mf_observables.jl` and `analysis/` changes were not modified.
- Julia environment resolved with Julia 1.12.7, ITensors 0.9.15, ITensorMPS 0.3.25, HDF5 0.17.2, and CSV 0.10.15; `Manifest.toml` was generated.
- Unit/integration suite: 57 tests passed. Coverage includes exact E_p selection, geometry kernels, deterministic seeds, on- and off-fixed-point variational constants, period-1 through period-4 handling, mixing, diagnostics primitives, immutable HDF5 restart fields, recursive accepted-only selection, and strict variational branch ranking.
- Local four-site end-to-end SCF smoke: solver, density search, MPO construction, checkpoint write/read, and final artifact path executed. It stopped at `maximum_iterations` by construction and is not scientific evidence.
- Local four-site pairing-field identity smoke: Hamiltonian identity error `-4.440892098500626e-16` and reconstructed/direct variational-energy difference `-4.440892098500626e-16`; the one-sweep effective eigenvalue differed from its final expectation by `-1.0199283164702422e-6`. This motivated using the direct bare-Hamiltonian energy for ranking and retaining the eigenvalue only as a consistency diagnostic.
- Phase 0 plan command passed. The configured worst-case reservation is `1.511718750` Perlmutter CPU node-hours under a `3.0` node-hour cap.
- Tiny Phase 0 script fixture: immutable seed creation, SHA verification, HDF5 MPS reload, two repeated metric solves, TOML output, MaxRSS parsing, numerical gates, and recommendation generation all executed. This L=2 fixture is an integration test only; its timing and energy are not benchmark evidence.
- Tiny diagnostic smoke: charge/spin grids, rung-cut entanglement, diagnostic HDF5, full sign-resolved rung/leg pair matrices, and one fixed-number sector DMRG all executed. The complete six-sector production gap bundle was not run locally.
- Perlmutter jobs submitted: none.
- Validation boundary: local tests validate APIs, algebraic bookkeeping, and serialization. They do not validate Perlmutter performance, production convergence, phase ordering, or physical conclusions.

Next action: sync the branch to Perlmutter, run `bash slurm/phase0_calibrate_cpu.sh plan`, review account/QOS/time/memory settings, then explicitly run `submit` if approved.
