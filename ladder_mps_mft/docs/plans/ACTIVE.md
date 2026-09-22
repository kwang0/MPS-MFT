# Active plan

Last reviewed: **2026-09-21**

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

September 21 user-provided retry status supersedes the preparation-only
status below: `20260921_latest_correlations_retry1` reports **49/56 branches
measured (51/58 MPSs)**. Rows 6, 11, 31, 34, 46, 51, 53 lack verified receipts.
The user then supplied accounting showing all seven timed out at their
two-hour limit; the other 49 jobs completed. No retry results/logs are synced
locally. The [selective retry](../reports/correlation_retry_20260921/TIMEOUT_RETRY.md)
is prepared: `slurm/complete_missing_correlations.sh` requests four hours
for each of the seven missing rows (1.96875 CPU node-hours), preserves the
49 successful entries, and reports combined completion across retry1/retry2.
User-run submission follows `git pull --ff-only`; nothing is submitted here.
The full-startup recovery below is historical and must not be rerun on retry1.

1. Full terminal correlations are enabled for future accepted and
   maximum-iteration states. The retrospective workflow covers 56 latest
   branches / 58 spatial MPSs. The submitted `20260918_latest_correlations`
   campaign has 56 recorded jobs; all 56 synced logs fail at shell startup
   because `phase1_gpu.sh` is sought beside Slurm's temporary script. There
   are zero local backfill diagnostic files or receipts. Recovery needs a
   user-run retry after synchronizing the corrected launcher. The authorized
   [recovery](../reports/correlation_retry_20260921/README.md) uses a separate
   `20260921_latest_correlations_retry1` run and requires an exact match to
   the original manifest. It reconciles only the failed measurement jobs,
   requires all parent jobs to have terminal failure accounting, and retains
   shared budget limits. No local submission; full scratch-MPS preflight
   remains user-run. Both complete sidecars for the new square A/B pairing state are
   available and verified, separately from that backfill campaign.
2. The [prepared CPU retry handoff](../reports/correlation_retry_20260921/README.md)
   retains the 8.15625-node-hour ceiling. GPU speed
   remains unbenchmarked; CPU is the existing implementation, not a proven
   performance winner. Retry preparation is authorized; no GPU port is included.
3. The user authorized the targeted square A/B test at (1.4,-0.2) and
   (1.4,-0.4). [Four starts are prepared](../reports/square_two_ladder_20260920/README.md):
   95%/5% references, B stripe displacement eight rungs, chi=200, 60 raw
   cell sweeps, full terminal measurements, one segment per job. The actual
   square bonds and A=B fields/energy reduction are checked. The campaign
   is submitted. Its [first result](../reports/square_two_ladder_20260920/FIRST_RESULT_20260921.md),
   pairing seed at (1.4,-0.4), is accepted at 40 sweeps and reproduces the
   one-ladder paired state. Stripe-seed stdout also reaches fixed_point at
   40; wait for its terminal state and the V=-0.2 results to compare profiles.
   Reservation ceiling: 16 node-hours, preserving the existing project cap.
   The latest user update reports two new A/B runs still ongoing; identities
   and live status remain unverified locally. The measurement retry excludes
   this campaign and preserves its reservations.
   Leg-odd charge and further registrations remain possible future controls.
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
