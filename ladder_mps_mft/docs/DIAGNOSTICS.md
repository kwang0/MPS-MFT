# Ladder diagnostics

All Fourier axes use rung momentum `qx = 2 pi m/L` and transverse momentum `ky = 0,pi`. Charge and spin structure factors use

```text
S_O(qx,ky) = (1/(2L)) sum_ij exp[i q.(r_i-r_j)]
              ( <O_i O_j> - <O_i><O_j> ).
```

The saved peak includes `qx`, `ky`, and both values divided by pi. The reference `pi*density` is stored for tracking, but a two-leg ladder can have multiple bands, so mismatch to that reference is not by itself a phase diagnostic.

`K_rho` is estimated from the first up to three nonzero `ky=0` charge modes. Because the stored structure factor is normalized by `2L`, the file reports both conventions

```text
K_rho,site = pi * dS_charge(q,0)/dq
K_rho,rung = 2 pi * dS_charge(q,0)/dq.
```

The second corresponds to first converting the total `ky=0` density structure factor to a per-rung `1/L` normalization. Match the normalization used by a comparison paper explicitly. Both finite-OBC estimators require L and chi scaling before interpretation.

The entanglement profile stores von Neumann and second-Renyi entropies for every MPS bond. The central-charge fit uses only even MPS bonds, which are cuts between complete rungs, and fits the open-boundary Calabrese-Cardy form away from the edges. Report the fit window, R-squared, L and chi; parity oscillations and gapped crossovers can make a finite-size central charge unreliable.

Sign-resolved singlet-pair matrices are available for rung, leg-0, and leg-1 bonds using the stored unnormalized convention `Delta_ab = c_up,a c_dn,b - c_dn,a c_up,b`. They are optional because the full MPO contractions are expensive. Report the convention, signs, and spatial decay, not only a maximum Fourier component.

The separate fixed-number calculations produce:

```text
spin gap       = E(N,2Sz=2) - E(N,0)
charge gap     = [E(N+2,0)+E(N-2,0)-2E(N,0)]/2
hole binding   = E(N-2,0)+E(N,0)-2E(N-1,1)
particle bind. = E(N+2,0)+E(N,0)-2E(N+1,1).
```

Their HDF5 artifact is separate from the number-parity SCF state. Use the measured spin and charge gaps together with |E_p| when checking whether t_perp is perturbatively small. The registry lookup alone cannot certify weak coupling.

The current bundled formulas require an even target particle number and an Sz=0 reference sector. Odd-N reference sectors need an explicit spin-sector convention before extension.
