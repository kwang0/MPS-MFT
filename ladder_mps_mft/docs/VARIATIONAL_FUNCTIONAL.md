# Variational energy and mean-field constants

## What is compared

The code implements a zero-temperature energy functional. It does not implement a finite-temperature entropy or free energy.

For a trial MPS `psi`, the effective single-ladder Hamiltonian is written

```text
H_eff = H_ladder - mu N + H_lin[alpha,beta,mu_mf].
```

The DMRG eigenvalue by itself depends on the chemical potential and on how the decoupled transverse terms were split between linear fields and constants. It therefore cannot rank SDW, CDW, and SC fixed points.

Let `L = <H_lin>` use the fields actually applied in `H_eff`. Let `E_perp` be the quadratic transverse mean-field energy evaluated from the measured correlators and therefore from the newly measured self-consistent fields. The required double-counting constant is

```text
C_dc = E_perp - L.
```

The reconstructed canonical functional is

```text
E_var,reconstructed = E_eff + mu <N> + C_dc.
```

The implementation also evaluates the bare ladder MPO directly and uses

```text
E_var,direct = <H_ladder> + E_perp
```

as `canonical_variational_energy`. The two expressions must agree. Their difference is stored as `variational_consistency_error`.

## Channel-by-channel constants

All current fields and correlators are real. With `F_a` the anomalous pair correlator, `X_a` the normal exchange correlator, and `delta n = n-1/2`:

```text
L_pair     = -2 sum_a alpha_a F_a
E_pair     =  1/2 L_pair

L_exchange =     sum_a beta_a X_a
E_exchange = 1/2 L_exchange

mu_mf      = K delta n
L_density  = sum_a mu_mf,a n_a
E_density  = 1/2 sum_a mu_mf,a delta n_a.
```

Thus the pair and exchange constants remove one copy of a bilinear mean-field contraction, while the density constant also contains the reference-density shift. These are the “MF double-counting constants”: they restore the expectation value of the original quadratic mean-field functional after its derivative has been inserted as a linear one-body field. Away from a fixed point the applied field is used only in `L`, while the measured field is used in `E_perp`; at convergence they coincide.

## Internal identity check

Every SCF iteration checks

```text
<H_ladder> = <H_eff> + mu<N> - <H_lin>.
```

The error per physical site must pass `hamiltonian_identity_tol`. The final DMRG eigenvalue is separately compared with `<H_eff>` and must pass `effective_energy_consistency_tol`. These gates catch field-sign, missing-term, and insufficient-DMRG-convergence errors.

## When a comparison is permitted

`scripts/compare_branches.jl` requires every state to be an accepted period-1 fixed point and requires identical model, numerical, implementation, and E_p-registry fingerprints. It rejects cross-geometry ranking because the geometries define different Hamiltonians.

This functional is stationary only at a self-consistent solution. Energies from intermediate iterations or periodic SCF cycles are diagnostic and must not be published as phase competition.

The implementation restores field-dependent mean-field constants. A field-independent offset from the underlying perturbative derivation, if present, cancels among branches of the same model but would be needed for an absolute comparison between different transverse geometries. This is another reason cross-geometry phase ranking is disabled.
