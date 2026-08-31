# Phase 1 numerical-resolution and error-budget contract

## What an acceptance threshold is not

The fixed-point, density, and energy-change tolerances are stopping gates. They
are not error bars on a branch energy or order parameter. In particular, two
accepted branches can differ by much less than the uncertainty associated with
their residual density mismatch, finite bond dimension, or incomplete SCF
relaxation. Such branches are numerically ordered by the stored functional but
are scientifically unresolved.

All current production GPU states use Float64 tensors. Float32 roundoff is a
historical v2 limitation, not the precision model for the current square run.
The important unresolved scales are DMRG truncation/optimization, density
targeting, SCF closure, small-field resolution, finite bond dimension, and
finite length.

## Small-field policy

The physical raw map remains

```text
x_(k+1) = f(x_k)
```

with no hard field threshold. Hard-zeroing the map near a floor can make one
component alternate between zero and a value just above the floor, exactly the
threshold chatter that the legacy convergence check was designed to avoid. A
hard-zeroed map would also be a different numerical problem and require its own
fingerprint and calibration.

Small fields are instead handled in two separate layers:

1. The solver stores the unmodified applied and measured fields and evaluates
   the canonical variational functional from them.
2. Post-processing scans common absolute floors `0`, `1e-6`, `1e-5`, and
   `1e-4`. Plots may suppress entries below a declared floor. An energy
   sensitivity must threshold the applied interaction fields and recompute all
   pair, exchange, and density transverse terms, including double counting,
   against the stored correlations. Rounding the printed total energy is not a
   valid substitute.

The thresholded energy is a frozen-correlation sensitivity of a stored state,
not the energy of a newly self-consistent thresholded Hamiltonian. No single
nonzero floor is adopted as physical until the scan is compared with DMRG,
bond-dimension, and SCF noise.

A later resolution-aware convergence metric may ignore a component only when
both the applied and measured magnitudes are below a calibrated floor. It must
also report threshold crossings and retain a separate absolute closure gate so
that zero/nonzero chatter cannot masquerade as convergence. That metric is not
silently introduced into the five-update probe.

## Per-branch uncertainty components

Write energies per physical site, with `N_s=2L`. For a terminal record define:

```text
u_density = |mu| |n - n_target|
u_SCF     = max recent |E_k - E_(k-1)| / N_s
u_id      = |Hamiltonian identity error| / N_s
u_eig     = |effective eigenvalue - expectation| / N_s
u_floor   = max_tau |E_tau - E_0| / N_s
```

`u_density` is the leading fixed-density interpolation scale from
`d(E/N_s)/dn = mu`. `u_floor` is the common frozen-correlation floor scan above.
The signed `E_p` registry row also supplies a convergence diagnostic for the
coupling `t_perp^2/|E_p|`; its `rel_diff` can be propagated as a frozen-state
sensitivity, but it is not treated as a certified error bar without the source
calculation's convergence history.

The known numerical envelope is conservatively reported as a sum of component
scales, not a root-sum-square statistical uncertainty. The terms are correlated
and are mostly systematic. Pairwise resolution requires the energy separation
to exceed the two branch envelopes after accounting for common-mode
cancellation explicitly.

The following are separate verification boundaries and cannot be inferred from
the tolerances above:

- DMRG `energy_tol` and cutoff are inputs, not certified variational errors.
  Final analysis needs sweep-energy, variance, and discarded-weight evidence or
  matched higher-chi comparisons.
- Bond-dimension uncertainty requires matched `chi` continuation and preferably
  extrapolation; later larger-chi runs must remain in the node-hour reserve.
- Finite-size uncertainty requires matched `L` calculations and bulk-window
  stability.
- Order claims require correlation functions and scaling, not a nonzero
  mean-field amplitude alone.

An energy ranking is scientifically resolved only among accepted states with
matching model, numerical, implementation, and `E_p` fingerprints, inside one
transverse geometry, and only when the pairwise separation exceeds the complete
available envelope. Otherwise report a resolution class or tie, while retaining
the exact stored energies for provenance.

## Consequence for the six-seed square pilot

The full numerical spread of the six accepted square energies is only
`2.490e-5` per physical site. Their density errors are
`4.504e-4`--`4.959e-4`; with the terminal chemical potential near `0.55`, the
leading density-interpolation scale alone is about `2.5e-4` per site. It is
therefore an order of magnitude larger than the entire seed-to-seed energy
spread. The six-way fine ordering is not scientifically resolved.

The robust result at this stage is qualitative: all six deliberately different
starts reached very similar charge and pairing profiles with nonzero uniform
d-wave proxy, while the remaining SDW differences are weak and resolution
sensitive. This is evidence for basin collapse at finite `L=64, chi=200`, not a
thermodynamic phase identification.

## Five-update tightening probe

The next campaign continues all six accepted full scratch parents with one
common numerical fingerprint:

- `chi=200`, 16 sweeps, cutoff `1e-11`, DMRG energy tolerance `1e-9`;
- inner and outer density tolerances `1e-4`;
- fixed-point absolute/relative field gates `1e-7` / `1e-4`;
- variational-energy change gate `1e-7` per physical site;
- at most five new raw-map MF evaluations and no Anderson acceleration;
- map threshold zero and the declared post-processing floor scan.

Five records cannot validate a period-two orbit under the complete-history
contract, which needs eight contiguous raw records for three recurrence links.
This short calculation is therefore a fixed-point residual/noise diagnostic. A
period-two-looking trajectory remains unresolved and must be continued as
separate raw phases rather than averaged or certified from this run.
