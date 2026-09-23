# Active plan

Last reviewed: **2026-09-23**

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

The user reports ongoing square t_perp and trellis runs; no live job IDs or
scheduler/accounting verification are available locally. Keep both source
checkouts fixed.

The [four trellis intertwined starts](../reports/trellis_intertwined_20260923/README.md)
are prepared at t0=1.2/1.4 and V=0/+0.2, only two-ladder cells, one period-16
intertwined seed each. They retain tau0=tau1=0.1, L64/chi200/60 raw sweeps,
exact E_p and full terminal correlations. Use a third detached worktree and
`submit_trellis_intertwined.sh`, with a separate control root and the existing
shared budget ledger. Four one-segment jobs reserve at most 16 node-hours.
No new submission, scheduler call or budget entry was made locally.

Earlier preparation of the user's [eight square t_perp starts](../reports/square_tp_scan_20260922/README.md):
t0=1.4, V=0 with tp=0.06/0.08 and V=-0.2 with tp=0.12/0.14, two seed families
each. Preparation and local checks are complete. Use a detached worktree and
`submit_square_tp_scan.sh`; retain the original trellis checkout and shared
budget ledger. The eight one-segment jobs reserve at most 32 node-hours.
The runs are now user-reported ongoing; synchronized job IDs remain unavailable.

The user now reports ongoing trellis jobs. Their current job IDs/status have
not been verified locally; the prepared-only account below is historical.

The user-requested [V=-1 trellis repetition](../reports/trellis_vm1_20260922/README.md)
is prepared for user-run submission: the same four one-/two-ladder and
stripe/pairing starts, exact target E_p, L64/chi200/60 raw sweeps, full terminal
measurements. `slurm/submit_trellis_vm1_comparison.sh` requests four 16-hour
one-GPU jobs under the shared cap (16 node-hours ceiling), with one segment
and an 11.5-hour solver deadline. No jobs submitted locally. Analyze after
the user syncs results; no damping, larger-cell or stability campaign is added.

1. The [September 22 correlation report](../reports/pair_correlations_20260922/README.md)
   verifies all 56 retrospective branches / 58 MPSs plus eight new square
   A/B diagnostics. No backfill entries remain missing. Local pairing survives
   near hole-rich stripe walls while longer-distance correlations decrease;
   full/connected and channel-sign comparisons, boundary/ref-window sensitivity
   and source validation are complete. All retrospective states remain unaccepted.
2. All four square A/B runs are accepted paired fixed points. The manuscript
   now includes the complete correlation analysis and square results with five
   new figures. Future controls are stationary stripes and matched L/chi
   comparisons; measurement GPU performance remains unbenchmarked.

3. The targeted square A/B test at (1.4,-0.2) and
   (1.4,-0.4) is complete. The [four-start contract](../reports/square_two_ladder_20260920/README.md) used
   95%/5% references, B stripe displacement eight rungs, chi=200, 60 raw
   cell sweeps, full terminal measurements, one segment per job. The actual
   square bonds and A=B fields/energy reduction are checked. Both seeds
   reproduce paired states at both attractions with accepted fixed-point gates.
   The original reservation ceiling was 16 node-hours. No further submission
   or continuation is requested by this analysis.
   Leg-odd charge and further registrations remain possible future controls.
   Manuscript Section 3.18 now explicitly qualifies the equal-average-density
   sector: a common external chemical potential and independent ladder
   densities would test an additional charge-transfer mode. Longer transverse
   charge/spin periods could be necessary; no such campaign is prepared.
   Cubic cell tests remain lower priority and are not prepared.
   Selective trellis convergence work, potentially modest linear damping,
   remains a separate future decision; its geometric frustration persists.
4. Rank stationary branches only after acceptance and fingerprint checks.
   The September 20 common-cell evaluation already establishes lower
   striped trial energies; residuals qualify optimality rather than invalidate
   that comparison. Larger transverse periods and continuing diagonal-stripe
   shifts remain possible. Choose cells that close both charge and spin
   textures for a specified tilt. No such campaign is prepared. Retain
   end-localized spin qualifications near the square boundary.
5. All 38 reviewed jobs are now accounted: 40.782361 actual node-hours,
   versus 40.068427 solver-only. Reconcile future compute through the existing
   400-additional-node-hour control; do not count reservations as actual cost.

Higher chi, length, wavelength and precise E_p sensitivity studies remain
deferred. Only the user operates or transfers to/from Perlmutter.
PROJECT_STATE.md carries the current snapshot, RUN_LOG.md the append-only
evidence, and dated preparation/earlier partial reports remain historical.
