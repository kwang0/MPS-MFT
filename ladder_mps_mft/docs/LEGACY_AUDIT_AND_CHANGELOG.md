# Legacy audit and implementation changelog

The refactor was based on the existing ladder CPU/GPU main-loop scripts and `plot_ladder_mf_observables.jl`. No legacy file was edited for this work.

## Preserved physics

- two-leg open ladder site ordering and leg/rung hopping;
- onsite U, nearest-neighbor V, chemical potential, anomalous alpha, normal beta, and spin-resolved density fields;
- the three transverse geometries and their Appendix-E alpha/beta/density maps;
- number-parity and Sz conservation for the anomalous mean-field solver;
- warm-start DMRG sweeps and the E_p registry parameterization.

## Changed numerical behavior

- The monolithic global-state loop is a pinned Julia package with typed settings and explicit CPU-only runtime controls. CUDA was intentionally not copied into Phase 0.
- E_p is selected from the CSV by exact model key. The signed value and source hash are stored; `abs(E_p)` is used as a denominator only for a bound pair by default.
- The legacy elementwise relative mask is replaced by global absolute/relative residuals and all-phase recurrence tests. Following Bollmark et al. (2025), period two is now a first-class physical mean-field solution when it is reproduced by an unmixed raw-map probe; mixer-dependent recurrences remain nonaccepted candidates.
- Linear damping is supplemented by adaptive Anderson mixing. A detected cycle can stop or trigger a documented reset/reduction.
- Chemical-potential search is safeguarded, bounded, warm-started, and reports unbracketed/time-limit outcomes instead of silently returning success.
- Applied, measured, and next/restart fields are distinct in HDF5. A terminal process is not equivalent to an accepted physical solution. Schema v3 stores every field, correlator, energy, and MPS belonging to an accepted orbit.

## New variational comparison

The effective eigenvalue is retained but no longer used directly for phase ranking. The code adds chemical-potential and mean-field constants, evaluates the bare ladder energy independently, stores reconstruction errors, and ranks only direct canonical energies from matched accepted branches.

## New diagnostics and provenance

Charge/spin structure factors, K_rho, rung-cut entanglement fits, sign-resolved pairing, fixed-sector gaps, weak-coupling checks, immutable artifacts, exact hashes, seed/continuation lineage, and strict recursive selection are new.

## Known limitations

- Phase 0 has not run on Perlmutter and CPU/GPU crossover is unknown.
- Only real mean fields are implemented; flux or complex-order solutions require a complex field/correlator generalization and corresponding functional audit.
- The direct functional follows the current real Appendix-E self-consistency map. Any change to that map must update and retest the channel constants.
- Sector gaps are expensive independent DMRG calculations and do not yet share optimized MPS continuations across sectors.
- No L or chi extrapolation automation is implemented yet.
- The central-charge and K_rho estimators are finite-OBC diagnostics, not publication-ready extrapolations by themselves.
- Periodic SCF cycles are classified but their dynamical stability is not analyzed.
