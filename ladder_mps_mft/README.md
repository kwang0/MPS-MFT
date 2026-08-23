# Ladder MPS+MF refactor

This is an isolated, CPU-first replacement path for the ladder mean-field workflow. It lives entirely under `ladder_mps_mft/`; the legacy scripts and result files outside this directory remain untouched.

The implementation currently provides:

- a pinned Julia project and modular ITensorMPS solver;
- exact lookup of pair-binding data from the copied `data/E_p_values.csv` registry, with no interpolation and a default refusal to use an unbound-pair value;
- zero-temperature variational phase comparison with the mean-field double-counting constants and a direct bare-Hamiltonian cross-check;
- a mixer-independent raw-map probe that accepts physical period-two mean-field solutions, followed only when needed by adaptive linear or Anderson fixed-point acceleration;
- immutable final HDF5 states, hash-checked parent/restart lineage, model/numerics/implementation fingerprints, and accepted-only selection;
- charge and spin structure factors, `K_rho`, rung-cut entanglement/central-charge fits, sign-resolved pair correlations, and separate fixed-sector spin/charge/pair-gap calculations;
- a guarded Perlmutter Phase 0 CPU calibration with exclusive ITensor threading backends, a common immutable fixed-mu warm-start state, exact DMRG-only timing, numerical-equivalence gates, and shared-QOS budget accounting.

The first complete Perlmutter matrix (`20260821_phase0_cpu_v2`) is retained as
backend-equivalence and screening evidence. It shortlisted `serial-t1` and
`blocksparse-t4`, but its `chi=64` fixed-`mu=0` workload ran at `n=0.5614`
rather than `n=0.9375`, so it is not the final production-scale timing. The v3
density-search seed failed and is no longer blocking calibration. Phase 0
v1.3.1 directly compares the two finalists at fixed `mu=1.8`, `chi=200`, and
six sweeps. The legacy-GPU comparison remains an estimate until a matched GPU
timing and `sacct` record are available.

## Quick start

Instantiate and test from this directory:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

Inspect the Phase 0 plan without submitting anything:

```bash
bash slurm/phase0_calibrate_cpu.sh plan
```

Run a configured SCF branch:

```bash
julia --project=. scripts/run_scf.jl configs/example_scf.toml
```

Generate independent SC, SDW, and CDW configurations from a common base:

```bash
julia --project=. scripts/prepare_branch_scan.jl configs/example_scf.toml output/my_branch_scan
```

Accepted fixed points and validated periodic solutions with identical model, numerical, implementation, and E_p-registry fingerprints can be ranked:

```bash
julia --project=. scripts/compare_branches.jl /path/to/sc/state.h5 /path/to/sdw/state.h5 /path/to/cdw/state.h5
```

See `docs/PHASES_0_TO_4.md` for the staged plan, `docs/LITERATURE_AND_PUBLICATION_GATES.md` for numerical context, and `docs/RUN_LOG.md` for the append-only synchronization record.
The v2 data-quality decision and conditional timing observations are recorded
in `docs/PHASE0_V2_AUDIT.md`.

## Important interpretation rules

- `completed=true` means either an accepted period-one fixed point or an accepted unmixed periodic solution. Mixer-dependent recurrences remain incomplete candidates.
- A physical period-two solution is stored phase by phase and is never replaced by its field average. This follows the CDW construction of Bollmark, Kohler, and Kantian, Phys. Rev. B 111, 125141 (2025).
- A timing payload is not a converged scientific state.
- `canonical_variational_energy` is a zero-temperature energy, not a finite-temperature free energy.
- Cross-geometry energies describe different Hamiltonians and are not ranked as competing phases by the comparison tool.
- Fixed bond dimension, enhanced pair fields, long correlation lengths, or favorable finite-size pairing do not by themselves establish superconducting long-range order.
