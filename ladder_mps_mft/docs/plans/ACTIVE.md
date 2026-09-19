# Active plan

Last reviewed: **2026-09-19**

## Completed analysis

The final sync completes the [combined review](../reports/campaign_review_20260918/README.md):
38 runs, 2280 cell updates, 2400 ladder solves, all unaccepted after 60 steps.
No simulation controls, original artifacts or acceptance flags changed.

- Cubic: both seeds stripe at all nine coordinates.
- Fine square: paired outcomes at four new points, distinct textures at
  (1.25,−0.4) and (1.4,−0.05). Full energy and matched physical spin/pairing
  cuts are in the report and manuscript; no first-order kink is resolved.
- Positive-V square: all four seeds lose pairing. The period-eight start
  retains an irregular six-node magnetic texture and a higher diagnostic
  endpoint energy, not sustained intertwined order.
- Trellis: both one-ladder seeds paired; both two-ladder seeds striped.
  The latter still has large alternating outer-MF relaxation. A/B are
  spatial; sweep parity is numerical, not physical dynamics.
- Coarse square remains the seven-stripe/two-paired reference, including
  both V=0 continuations (902 evaluations, 27.350764 actual node-hours).

The manuscript now separates Sections 3.11/3.12 into comprehensive positive-V
and trellis analyses. Section 3.10 has physical spin/pairing cut figures.
Existing square/cubic comparisons and the material introduction remain.
Actual methods/literature-review LaTeX and PDF notes are updated too;
the Overleaf bundle includes the new figures.

## Next evidence and decisions

1. Full terminal correlations are enabled for future accepted and
   maximum-iteration states. The retrospective workflow covers 56 latest
   branches / 58 spatial MPSs. All terminal state branches are now synced,
   but full scratch-MPS preflight remains user-run. New pair–pair evidence
   awaits measurement and sync; it is not inferred from anomalous order.
2. The [prepared CPU handoff](../DIAGNOSTICS.md) is unchanged. GPU speed
   remains unbenchmarked; CPU is the existing implementation, not a proven
   performance winner. No new submission or GPU port was requested here.
3. Consider selective convergence/stability work on the two square boundary
   points and the alternating two-ladder trellis trajectories. Linear damping
   is a possible targeted trellis test, not a prepared control change.
4. Rank only accepted, fingerprint-compatible canonical solutions. Retain
   open-end/cell qualifications on trellis endpoint energy comparisons and
   end-localized spin qualifications near the square boundary.
5. All 38 reviewed jobs are now accounted: 40.782361 actual node-hours,
   versus 40.068427 solver-only. Reconcile future compute through the existing
   400-additional-node-hour control; do not count reservations as actual cost.

Higher chi, length, wavelength and precise E_p sensitivity studies remain
deferred. Only the user operates or transfers to/from Perlmutter.
PROJECT_STATE.md carries the current snapshot, RUN_LOG.md the append-only
evidence, and dated preparation/earlier partial reports remain historical.
