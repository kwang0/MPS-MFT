# Ladder MPS+MF refactor

This is an isolated, CPU-first replacement path for the ladder mean-field workflow. It lives entirely under `ladder_mps_mft/`; the legacy scripts and result files outside this directory remain untouched.

The implementation currently provides:

- a pinned Julia project and modular ITensorMPS solver;
- exact lookup of pair-binding data from the copied `data/E_p_values.csv` registry, with no interpolation and a default refusal to use an unbound-pair value;
- zero-temperature variational phase comparison with the mean-field double-counting constants and a direct bare-Hamiltonian cross-check;
- hybrid absolute/relative SCF convergence, adaptive linear or Anderson mixing, and fundamental-period detection for periods 1 through 8;
- immutable final HDF5 states, hash-checked parent/restart lineage, model/numerics/implementation fingerprints, and accepted-only selection;
- charge and spin structure factors, `K_rho`, rung-cut entanglement/central-charge fits, sign-resolved pair correlations, and separate fixed-sector spin/charge/pair-gap calculations;
- a guarded Perlmutter Phase 0 CPU calibration with exclusive ITensor threading backends, a common immutable warm-start state, numerical-equivalence gates, shared-QOS budget accounting, and a separate chi=200 validation.

Phase 0 has been implemented and validated locally at smoke-test scale. No Perlmutter jobs have been submitted, so there is not yet evidence that CPU is cheaper or faster than the legacy GPU path.

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

Only accepted fixed points with identical model, numerical, implementation, and E_p-registry fingerprints can be ranked:

```bash
julia --project=. scripts/compare_branches.jl /path/to/sc/state.h5 /path/to/sdw/state.h5 /path/to/cdw/state.h5
```

See `docs/PHASES_0_TO_4.md` for the staged plan, `docs/LITERATURE_AND_PUBLICATION_GATES.md` for numerical context, and `docs/RUN_LOG.md` for the append-only synchronization record.

## Important interpretation rules

- `completed=true` means an accepted period-1 fixed point. A finished process that found a period-2 or longer cycle has `process_completed=true`, `accepted=false`, and `completed=false`.
- A timing payload is not a converged scientific state.
- `canonical_variational_energy` is a zero-temperature energy, not a finite-temperature free energy.
- Cross-geometry energies describe different Hamiltonians and are not ranked as competing phases by the comparison tool.
- Fixed bond dimension, enhanced pair fields, long correlation lengths, or favorable finite-size pairing do not by themselves establish superconducting long-range order.
