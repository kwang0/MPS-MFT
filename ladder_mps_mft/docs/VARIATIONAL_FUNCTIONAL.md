# Variational energy and mean-field constants

## What is compared

The code implements a zero-temperature energy functional. It does not implement a finite-temperature entropy or free energy.

For a trial MPS `psi`, the effective single-ladder Hamiltonian is written

```text
H_eff = H_ladder - mu N + H_lin[alpha,beta,mu_mf].
```

The DMRG eigenvalue by itself depends on the chemical potential and on how the decoupled transverse terms were split between linear fields and constants. It therefore cannot rank SDW, CDW, and SC solutions.

Let `L = <H_lin>` use the partner fields actually applied in `H_eff`. The quadratic transverse mean-field energy `E_perp` is evaluated from those applied partner fields and the current MPS correlators. At a period-one solution, applied and newly measured fields coincide. At a physical period-`p` solution, the applied fields belong to the preceding orbit phase and must remain distinct. The required double-counting constant is

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

Density targeting is numerical rather than exact, so comparisons and SCF
energy-stability gates use the leading fixed-density interpolation

```text
E_target = E_var,direct + mu (N_target - <N>).
```

Both the correction and `E_target` are stored. This does not replace the
canonical functional or its double-counting terms; it puts nearby residual
density errors on the same target-N tangent plane. It is not a substitute for
tighter density convergence or a certified error bar.

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

Thus the pair and exchange constants remove one copy of a bilinear mean-field contraction, while the density constant also contains the reference-density shift. These are the “MF double-counting constants”: they restore the expectation value of the original quadratic mean-field functional after its derivative has been inserted as a linear one-body field.

For a physical two-cycle, phase A is solved in the fields produced by phase B and vice versa. Each phase energy therefore uses its applied partner field. The reported solution energy is the arithmetic mean of the two phase energies. Replacing the partner field by an Anderson average or by phase A's outgoing field would change the physical transverse contraction and can erase the CDW solution.

## Internal identity check

Every SCF iteration checks

```text
<H_ladder> = <H_eff> + mu<N> - <H_lin>.
```

The error per physical site must pass `hamiltonian_identity_tol`. The final DMRG eigenvalue is separately compared with `<H_eff>` and must pass `effective_energy_consistency_tol`. These gates catch field-sign, missing-term, and insufficient-DMRG-convergence errors.

## When a comparison is permitted

`scripts/compare_branches.jl` accepts either a gated period-one fixed point or
an unmixed, validated periodic solution. It requires identical model,
numerical, implementation, and E_p-registry fingerprints. For a periodic
solution it ranks the orbit-averaged target-density-corrected canonical energy,
never a single phase or an averaged field. It rejects cross-geometry ranking
because the geometries define different Hamiltonians.

Mixer-dependent recurrences and intermediate iterations remain diagnostic. A periodic branch may enter phase competition only when every phase, recurrence link, density, phase-energy recurrence, and Hamiltonian-consistency gate passes in the unmixed probe. Periods beyond those explicitly mapped to the transverse physics remain candidates rather than automatically accepted solutions.

The implementation restores field-dependent mean-field constants. A field-independent offset from the underlying perturbative derivation, if present, cancels among branches of the same model but would be needed for an absolute comparison between different transverse geometries. This is another reason cross-geometry phase ranking is disabled.
