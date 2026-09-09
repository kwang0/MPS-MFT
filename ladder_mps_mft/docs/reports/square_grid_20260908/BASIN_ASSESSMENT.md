# September 8 follow-up: SDW growth, Anderson cancellation, and competing seeds

The restored histories support revisiting the interpretation of the loose
square-grid endpoints. In particular, the six accepted `(t0,V)=(1.4,0)` runs
do **not** establish a unique stable paired basin. Three stop very early;
three show coherent SDW growth followed by suppression under Anderson mixing.
The selected run's plotted iteration 23-to-24 jump is quantitatively explained
by the first Anderson extrapolation. Original solver flags, source artifacts,
bundle selection, and energies are preserved.

The user proposes comparing representative uniform d-wave and stripe CDW/SDW
seeds across the grid. This is the recommended next scientific direction,
subject to qualifying convergence and stability checks before committing a
full campaign. This document records a proposal, not a solver change or a
submission. The cubic grid and separate `(1.4,-0.4)` strong-stripe/control
campaign remain pending according to the user's September 8 report.

## Evidence and definitions

The analysis reads the same 17 modern candidate states recorded in
[selection.json](selection.json), verifies their compact-file hashes, and uses
their full saved applied and measured fields. The two legacy coverage points
are not given a new stability certification. All analyzed states have L=64
and chi=200. Bulk RMS uses rungs 6–59, matching the Fourier boundary trim.

The plotted SDW diagnostic is the leg-odd spin Hartree profile, with the
existing `audit_spatial_phase_defects.field_profiles` normalization. It is an
MF proxy in energy units, not a physical spin structure factor. For a separate
response diagnostic, define the 128-site spin field
`s = (mu_cdw_up - mu_cdw_dn)/2` and compute
`g = dot(s_applied, s_measured) / dot(s_applied, s_applied)` and their cosine.
This projection gain describes a saved finite-amplitude, finite-chi map
evaluation. It is **not** a measured Jacobian eigenvalue, energy-Hessian test,
or proof of thermodynamic instability. Ratios at tiny spin amplitude can be
dominated by numerical error.

See [six-seed trajectories](six_seed_sdw_growth.png) and
[all-candidate summary](basin_seed_summary.csv):

- At `(1.4,0)`, `legacy_pairing_mixed`, `pairing_dwave_m000`, and `stripe_m004`
  stop after only six saved map evaluations. The first two have terminal spin
  RMS around `3e-7 t`, so their weak spin response needs a controlled
  perturbation above the numerical floor. The small `stripe_m004` run already
  has coherent spin growth: final projection gain `1.0694`, spin-channel
  relative residual `0.0804`, and global relative residual `0.00411`.
- The other three V=0 seeds (`stripe_m005`, `stripe_pairing_m004`, and
  `stripe_pairing_m005`) reach stored record 21 without mixing. Across their
  last five raw records, bulk spin RMS grows by factors `1.374–1.381` over
  four steps. Median full-spin projection gains are `1.0961–1.0968`, with
  input/output cosines above `0.9998`. These are coherent growing profiles,
  while the d-wave MF proxy remains approximately `0.00449 t`. All three
  subsequently reach their stored accepted endpoints using Anderson mixing.
- At `(1.4,-0.4)`, the four small stripe/coexistence seeds instead show
  coherent spin suppression: median projection gains `0.603–0.612`, cosines
  above `0.9998`, and bulk RMS falls to `0.134–0.143` of its value four raw
  steps earlier. This supports attraction toward the paired state against
  these tested weak stripe patterns. It does not decide whether a distinct
  strong-stripe solution exists or has lower energy. Pure `stripe_m004`
  remains excluded by the earlier global slow-mode selection screen.
- At `(1.2,0)` and `(1.2,-0.2)`, the short smooth-seed histories also show
  increasing spin signals, but final RMS is only `2.0e-6` and `1.2e-6 t`.
  These warrant controlled checks; they do not yet identify the final basin.
  The five-record `(1.2,-0.4)` history is likewise insufficient for a stable
  phase assignment.
- At `(1.0,-0.2)`, pairing decays to `3.05e-11 t` while a strong stripe
  profile settles, with terminal spin-channel relative residual `2.72e-4`.
  At `(1.0,-0.4)`, pairing also decays, but the subsequent accelerated
  trajectory stops as diverging; see [the divergence analysis](ANALYSIS.md).
  These observations support the user's delayed-stripe interpretation for
  those trajectories without establishing that every weak seed must do so.

## What caused the selected V=0 jump?

The history plot displays the initial seed at iteration 1. Thus plotted
23-to-24 means stored 22-to-23. Stored 22 is the first linearly mixed input;
stored 23 is the first Anderson input.

The timing is set by the archived config: `unmixed_cycle_probe=true`,
`probe_iterations=20`, and `mixing.method="anderson"`. Stored record 1 is the
initial evaluation; records 2–21 complete the 20 raw probe steps. With no
accepted fixed point or recurrence, `Solver.jl` ends the probe and clears
mixer history. Its first mixing call has only one input/output pair, so
`Mixing.jl` falls back to a linear half-step for stored record 22 (plotted 23).
The next call has two pairs and forms the first Anderson input for stored
record 23 (plotted 24). This transition tests the probe counter and solver
status, not whether a growing spin channel has settled.

The existing mixer builds `z_i = (1-damping)*x_i + damping*F(x_i)` and combines
them with coefficients summing to one. At this transition, damping is `0.5`
and the two coefficients for stored inputs 21 and 22 are
**`+22.6810987947` and `-21.6810987947`**. These large signed coefficients
cancel much of the growing spin field before the next DMRG solve. Replaying
the saved full-field algebra reproduces that next applied input to a maximum
absolute error of `1.57e-15`. All four saved mixed transitions reproduce to
within `1.58e-15`.

| Plotted iteration | Stored iteration | Applied input | Applied bulk spin RMS | Measured bulk spin RMS |
|---:|---:|---|---:|---:|
| 22 | 21 | Raw map | 0.00146884 | 0.00159570 |
| 23 | 22 | Linear mixing | 0.00153217 | 0.00166575 |
| 24 | 23 | First Anderson | 0.00039610 | 0.00038820 |
| 26 | 25 | Final Anderson | 0.00039114 | 0.00038457 |

Measured bulk spin RMS drops by **76.7%** from plotted 23 to 24. The full spin
profiles have cosine `-0.3394` across the jump, so their shape changes as well
as their magnitude. The middle-rung spin difference is nearly unchanged
(`0.00135564` to `0.00136663`); a center trace alone misses the spatial change.
The d-wave MF proxy changes by only about `0.26%`. The corrected canonical
energy actually rises by `2.72e-7 t/site` at this step, within the loose
energy tolerance.

This is a reproduced mixer effect. Anderson's least-squares objective combines
residuals; it does not enforce positive interpolation weights or canonical
energy descent. See the local [mixer](../../../src/Mixing.jl) and Walker's
[primary method description](https://users.wpi.edu/~walker/Papers/anderson_accn_algs_imps.pdf).
The selected terminal full-spin projection gain has fallen to `0.9918` after
the profile changes. Therefore, the earlier gain above one cannot simply be
assigned to that terminal state. Its transverse stability requires fresh
perturbed evaluations.

An illustrative scalar map makes the distinction explicit. Let `F_s(s)=1.1s`.
Raw iteration amplifies every nonzero `s`, but `s=0` still solves `F_s(s)=s`.
A linear half-step takes input `a` to `1.05a`; signed coefficients `21,-20`
then cancel both those inputs and their residuals exactly. Ideal unregularized
Anderson can consequently land on zero even though zero repels raw iteration.
This example explains the mechanism, not the actual coupled spin map or its
energy curvature. The actual run has multiple channels, regularization, and a
changing spatial spin profile.

SCF iteration is not physical time evolution, and its raw map is not generally
an energy-descent algorithm either. A mixed field defines the next effective
Hamiltonian; minimizing that Hamiltonian in the inner DMRG solve does not
establish a minimum of the full self-consistent canonical energy. Root finding
can in principle locate a stationary saddle as well as a minimum. Therefore
the observed cancellation has a valid numerical meaning, but it is not evidence
that the growing order has physically become unfavorable. Basin-access claims
must identify the update rule and use independent endpoint stability/energy
checks.

Evidence: [jump figure](sdw_jump.png), [every saved record](sdw_jump_history.csv),
and [coefficients, errors, source hash, and final DMRG diagnostics](anderson_replay.json).

## Why the acceptance gates can miss this

The period-one gate in [Convergence.jl](../../../src/Convergence.jl) combines
the global field residual, a global residual-direction slow-mode estimate,
density, energy stability, and identity/consistency controls. In
[Solver.jl](../../../src/Solver.jl), that gate can terminate a run before the
configured 20-step initial raw probe finishes. A fixed point accepted after
Anderson mixing does not receive a mandatory subsequent raw stability probe.

At the selected terminal V=0 point, the global relative residual is `0.00184`,
but the spin-only relative residual is `0.0375`. The global slow-mode
extrapolated residual is `0.00327`, passing `0.005`; its contraction estimate
comes from the mixed trajectory and is not the growth rate of an independent
SDW perturbation under the raw map.

For illustration, if a weak spin component obeys `F_s(s) ≈ lambda*s` with
`lambda > 1`, its absolute residual `(lambda-1)*s` can be arbitrarily small
while it grows. Normalizing the combined field residual by larger pairing,
charge, and exchange components further obscures that growth. Tighter global
tolerances alone do not establish local stability. Conversely, a large
relative residual in a spin component at the numerical floor or coherently
decaying toward zero is not by itself a reason to reject a paired state.

## Proposed next comparison

1. Use two **representative finite-amplitude seed families** at each target:
   uniform d-wave and stripe CDW/SDW. Leave every order channel free so that
   coexistence can emerge. Give each family a controlled weak competing-order
   perturbation; do not rely on numerical leakage out of a symmetry subspace.
   Record parent hashes and the seed transformation. Rebuild fields from
   template correlations using the target model's couplings rather than
   transplanting coupling-weighted fields unchanged across t0, V, or geometry.
2. Qualify the stopping procedure on `(1.4,0)` and `(1.4,-0.4)` first. Monitor
   full spatial/Fourier profiles and noise-aware channel residuals. Require a
   suitable initial raw observation window and raw checks after accelerated
   convergence. Apply controlled SDW/CDW perturbations to the paired endpoint
   and pairing perturbations to the stripe, preferably at two amplitudes above
   the map/DMRG error floor. Observe enough steps to resolve slow growth or
   decay. Keep numerical self-consistency, raw-map response, and energetic
   preference as separate recorded conclusions. A noncontracting map is not
   itself an energy-Hessian calculation.
3. Fill two branch slots per coordinate (18 for the square grid), reusing
   compatible verified results where possible. Tighten both inner DMRG and
   outer convergence until endpoint uncertainty is comfortably smaller than
   the competing energy gap. Compare target-density-corrected canonical
   solution energies only at matching Hamiltonian, density, length, chi,
   numerical, implementation, and E_p fingerprints. Retain unresolved cases
   instead of assigning winners from tiny differences.
4. Focus extra chi, length, and alternative stripe-wavelength checks near
   close crossings or failed convergence. Two seed families provide a useful
   finite-size variational phase comparison, not proof of the global minimum
   over all textures. Apply the qualified protocol to the cubic geometry
   after reviewing its pending coverage results. The prepared L=96/128 seeds
   remain available; the proposal does not cancel or submit any work.

## Reproduction and validation

Local PowerShell, from the repository root:

```powershell
python -B -X utf8 ladder_mps_mft/scripts/analyze_square_basin_stability.py
```

The script verifies all 17 source hashes, checks that each raw input equals
the preceding saved measured field, and replays four mixed transitions. It
writes the two CSVs, replay JSON, and two PNGs above; both figures were visually
inspected. This is stored-data analysis only: no new DMRG, finite-difference
Jacobian, energy curvature, Perlmutter access, or scheduler action was run.
