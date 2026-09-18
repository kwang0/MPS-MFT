# Active plan

Last reviewed: 2026-09-18

## Current outcome

The [combined campaign review](../reports/campaign_review_20260918/README.md)
completes the requested analysis of the cubic grid, finer square cuts,
positive-V comparison and first completed trellis run. It includes 34
complete 60-step chi=200 histories (2040 MF evaluations); none is formally
accepted. No simulation controls or acceptance flags changed.

- Cubic: all 18 starts, nine points, essentially unpaired stripes from both
  seeds. The square paired corner is absent in this tested cubic grid.
- Finer square: all 12 starts; paired outcomes at four points and distinct
  stripe/paired trajectories at (1.25,-0.4) and (1.4,-0.05). Signed endpoint
  energy gaps are diagnostics only. The weak spin of the paired V=-0.05
  trajectory is concentrated near the ends; central spin still decays.
  Full energy-versus-parameter plots now include both coarse endpoints per
  cut and the latest V=0 continuations. A small V-slope downturn is visible
  after common linear subtraction; t0 curvature is gradual. A first-order
  kink is not resolved by the current five-point cuts.
- Square (1.2,+0.2): three completed starts lose pairing and reach stripes.
  The period-eight intertwined seed has 46 complete log records, no state.
- Trellis: the one-ladder stripe seed becomes paired after 60 sweeps. Other
  logs contain 21 one-ladder pairing, 18 two-ladder stripe and 1 two-ladder
  pairing sweeps, without spatial states. A/B are spatial, not temporal.

The [coarse square grid](../reports/two_basin_grid_20260915/README.md) already
includes both V=0 continuations: 100/82 cumulative evaluations, seven stripe
and two paired coordinates, 902 total evaluations, 27.350764 actual node-hours.
Its underlying data and historical parent controls remain unchanged.
The combined report and LaTeX manuscript now compare square and cubic phase
diagrams, full energy grids and physical spin/pairing RMS grids side by side.
The 1982 plotted evaluations retain the square continuations. This is a
presentation update with matched RMS conventions, not a new phase selection.

## Next evidence and decisions

1. Await user-synchronized state/checkpoint artifacts for square job 58394105
   and trellis jobs 58468873/58468875/58468876. Do not infer phase from their
   logged energy alone or submit duplicates based on this snapshot.
2. Compare terminal physical profiles and complete raw histories. Trellis
   analysis must use stored correlations because normal bonds contribute
   to its Hartree fields. Distinguish skew one-ladder repetition from the
   rectangular A/B cell and its separate mean-density constraints.
3. Decide whether selected square boundary lineages need further convergence
   or stability probes. No new continuation or tolerance change is prepared.
   Distinct unfinished endpoints do not yet prove a first-order transition,
   and end-weighted spin does not certify bulk coexistence.
4. Rank only accepted, fingerprint-compatible canonical solutions. Keep
   diagnostic endpoint differences separate from certified energy selection.
5. Reconcile actual allocation costs before choosing additional compute.
   The current completed subset records 31.931350 solver-only node-hours;
   only 12 cubic jobs have actual reconciliations, totaling 9.441597.
   These are different coverage sets, not interchangeable totals.

Higher chi, length, stripe wavelength and precise E_p sensitivity checks remain
deferred at the user's request. The four L=96/L=128 chi=200 seeds with fixed
L=64 E_p remain available in the [finite-size preparation](../reports/finite_size_seeds_20260906/README.md).
The 400-additional-node-hour project control and user-only Perlmutter boundary
remain in force. First-segment ceilings do not authorize blanket extensions.

## Continuity

Current findings and costs live in `PROJECT_STATE.md` and the linked detailed
reports; `RUN_LOG.md` has the append-only provenance, commands and validation
record. The September 16 partial snapshot and September 15 preparation reports
are retained as history and now link to the September 18 results. At the user's
request the actual manuscript, methods notes and literature-review LaTeX/PDF
now incorporate the new results. The manuscript explicitly uses numerical
evidence through September 18, including full energy cuts and slope diagnostics;
the bibliography and literature-search cutoff are unchanged.
