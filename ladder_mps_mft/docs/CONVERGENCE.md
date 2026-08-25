# Fixed points and physical mean-field orbits

## Why period two is physical

Bollmark, Kohler, and Kantian, Phys. Rev. B 111, 125141 (2025), show that the
repulsive transverse density channel can have a physical period-two solution of
the mean-field iteration. At iteration `p` its energy contains the density from
the partner phase `p-1`, schematically

```text
V_perp proportional to (2 <n>_(p-1) - 1) <n>_p.
```

The two phases are distinct effective Hamiltonians and MPS ground states. Their
density difference is a CDW order parameter. Averaging the fields can erase this
solution, so a period-two orbit is not treated as failed fixed-point convergence.

Primary reference: https://doi.org/10.1103/PhysRevB.111.125141, especially Sec.
III, Eq. (9), Eq. (14), Fig. 2, and Appendix B.

## Mixer-independent probe

The production-shaped default begins with 20 complete raw mean-field updates:

```text
x_(k+1) = f(x_k)
```

No linear damping or Anderson extrapolation is applied during this probe. The
default `probe_max_period=2` and `accepted_periods=[1,2]` prioritize the physical
period-two construction supported by the paper. An early recurrence candidate
does not truncate those 20 updates: it must either pass every numerical gate or
remain under observation through the full probe budget. Anderson mixing starts
only after that exhaustive initial probe finds neither an accepted fixed point
nor an accepted orbit.

For every stored alpha, beta, and spin-resolved Hartree field, the code records

```text
r_abs = max |f(x)-x|
r_rel = ||f(x)-x||_2 / max(||f(x)||_2, ||x||_2, eps).
```

These residuals certify period one only. A physical period `p>1` is instead
validated by all of the following:

1. every one of the `p` phases recurs across `period_repeats` links;
2. every raw-map link closes, so the next applied field is the preceding measured field;
3. every phase passes the target-density and Hamiltonian-consistency gates;
4. the energy of each phase recurs on the next visit to that same phase;
5. `p` appears in the explicit `accepted_periods` list.

With `period_repeats=3`, period two needs four complete cycles, or eight
contiguous raw-map records. Searching upward reports the smallest passing
period. A recurrence in Anderson- or linearly mixed history is only a
`periodic_candidate` and cannot be accepted without a raw probe. The driver
archives that mixer-dependent candidate, clears mixer history, applies the last
measured field directly, and performs one fresh raw-map probe. Thus damping
cannot manufacture, suppress, or certify the physical orbit. If the controlled
probe remains unaccepted, the run stops for inspection instead of repeatedly
damping and re-probing the same recurrence.

## Outcomes

- `fixed_point`: accepted period-one solution.
- `periodic_solution`: accepted raw-map orbit, including the default physical period two.
- `periodic_candidate`: recurrence seen through a mixer, a failed raw-map gate, or a period not enabled in `accepted_periods`.
- `stagnated`, `diverging`, `nonfinite`, `time_limit`, and `maximum_iterations`: incomplete numerical outcomes.

An accepted orbit stores every phase's applied and measured fields,
correlations, energy decomposition, chemical potential, density, and MPS. Quick
diagnostics are produced separately for every phase. The HDF5 file also stores
the orbit-averaged canonical energy, phase-energy spread, and the Bollmark-style
density contrast over the central `orbit_bulk_fraction` of rungs (one half by
default) to reduce open-boundary contamination.

The code never averages orbit phases. `cycle_action` applies only after the
initial raw probe has exhausted its full budget with an unaccepted recurrence.
`stop` archives and terminates; `continue` archives it, clears Anderson history,
reduces damping, and begins accelerated fixed-point search. A recurrence first
seen through the mixer always triggers the separate raw probe described above.

For a phase-resolved follow-up, `parent_orbit_phase=NNN` selects the named
member of a hash-pinned full orbit artifact. The phase MPS is retained as the
DMRG warm start and that member's measured field becomes the next applied raw
field. This is a parent lineage, not a same-driver resume, and stateless copies
cannot supply the omitted phase MPS.

Acceptance certifies a numerically self-consistent orbit of an explicitly
enabled period; it does not by itself identify the broken symmetry. Calling a
period-two orbit a CDW additionally requires a nonzero bulk density contrast,
the expected spatial pattern, and stability under length, bond-dimension, and
bulk-window checks. An orbit confined to another field channel may represent a
different solution or a gauge/sign convention and must be interpreted on its
own observables.

## Higher periods

Recurrences can be inspected through `max_period=8`, but only periods in
`accepted_periods` can become physical solutions. To accept period `p>2`, raise
`probe_max_period`, include `p` in `accepted_periods`, and give
`probe_iterations` enough room for transients plus at least
`p*(period_repeats+1)` raw records. Such an extension also requires a physical
mapping between orbit phases and transverse sublattices; numerical recurrence
alone is not that justification.

## Mixing after the probe

Linear and Anderson mixing retain their adaptive damping. Damping is reduced
after residual growth and increased conservatively after improvement. Their
role is acceleration of a period-one search after the unbiased orbit probe, not
classification of physical recurrence. Density targeting remains a separate
safeguarded bracket/secant search with its own tolerance, iteration cap, and
deadline.
