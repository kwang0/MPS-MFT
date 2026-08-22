# Numerical-literature context and publication gates

This file translates the prior Fourier-grid audit into requirements for the refactored workflow. It does not add new scientific results.

The closest methodological benchmark is [Bollmark et al., Phys. Rev. X 13, 011039 (2023)](https://doi.org/10.1103/PhysRevX.13.011039), which studied the same repulsive-ladder baseline near U/t=8 and density 0.9375 and supplied the Appendix-E MPS+MF construction. Its production calculations used larger L and chi than the legacy L=64, chi=200 grid and emphasized bond-dimension sensitivity. The new common functional, fingerprint gates, and L/chi plan are intended to bring SDW/CDW/SC competition to that standard of control.

[Bollmark, Koehler, and Kantian, Phys. Rev. B 111, 125141 (2025)](https://doi.org/10.1103/PhysRevB.111.125141) demonstrates multichannel CDW/SC MPS+MF competition in coupled chains. Crucially, its CDW solution is a period-two orbit of the MF iteration: two distinct ground states and effective Hamiltonians alternate, and their density difference is the CDW order parameter. Independent phase seeds, mixer-independent orbit detection, phase-resolved storage, and a common partner-field functional are therefore central rather than optional bookkeeping.

For isolated repulsive two-leg ladders, [Dolfi et al., Phys. Rev. B 92, 195139 (2015)](https://doi.org/10.1103/PhysRevB.92.195139) and [Shen, Zhang, and Qin, Phys. Rev. B 108, 165113 (2023)](https://doi.org/10.1103/PhysRevB.108.165113) show why pair exponents, K_rho, boundaries, truncation, and long lengths matter in identifying Luther-Emery behavior. A finite-chi enhancement of an anomalous field or a long but exponential correlation length is not equivalent to superconducting long-range order.

[White, Affleck, and Scalapino, Phys. Rev. B 65, 165122 (2002)](https://doi.org/10.1103/PhysRevB.65.165122) provides the caution that open-boundary Friedel oscillations can imitate charge order. Charge peaks need length scaling, bulk-window checks, and consistency with real-space profiles. Broader cylinder calculations such as [Jiang and Devereaux, arXiv:1806.01465](https://arxiv.org/abs/1806.01465) similarly illustrate delicate competition among spin/charge texture and pairing.

## Minimum gates for a publishable phase comparison

1. Accepted period-one fixed points or unmixed, all-phase-validated periodic solutions from independent SC, SDW, and CDW seeds, plus forward/reverse continuations near transitions.
2. Common direct variational energy with Hamiltonian-identity and eigenvalue-consistency errors reported.
3. Matched model/numerics/code/E_p fingerprints and explicit metastable branches.
4. L and chi scaling of energy differences, structure-factor peaks, long-distance pair correlations, and entanglement diagnostics.
5. Sector-resolved spin/charge gaps and pair binding at each claimed weak-coupling point; t_perp must be compared with all available scales.
6. Boundary-safe real-space fits and uncertainty under fit-window movement.
7. Claims separated into robust phase identification, candidate transition, finite-size/finite-chi trend, and unresolved evidence.

An accepted period-two orbit is a numerical solution class, not automatically
a CDW label. A CDW claim also needs a finite bulk density contrast with the
expected profile and controlled `L`, `chi`, and boundary-window dependence.

The most promising near-term result remains a controlled geometry-dependent competition among magnetic, charge, and pairing solutions. The publishable advance is not a larger mean-field amplitude by itself; it is a variationally ranked, recurrence-resolved, provenance-complete phase diagram with controlled isolated-ladder and finite-entanglement diagnostics.
