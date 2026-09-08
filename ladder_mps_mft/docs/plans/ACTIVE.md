# Active plan

Last reviewed: 2026-09-06

Current state: the chi=400 comparison is locally analyzed. The user judges its
energetic result sufficient to proceed with a small finite-size study, despite
the retained unaccepted solver flags. Four L=96/128 chi=200 seeds and review
configs are ready. The user reports the square-grid, cubic-grid and
V=-0.4 stripe campaigns remain pending. Their submission IDs are now synced;
live scheduler state is not independently verified. This document is not
submission or cancellation authority.

## Active scientific questions

1. **Square finite-size comparison.** Review the prepared pairing/stripe seeds
   at L=96 and 128, chi=200. The L=64 energetic advantage of the stripe is
   `2.43361262e-4 t/site` at chi=400, and the user regards its bond-dimension
   robustness as sufficient motivation to study length next. Hold the L=64
   E_p fixed, retain the original edge halves, insert flat paired bulk or
   complete stripe cells, and compare energy differences within each length.
   See `docs/reports/finite_size_seeds_20260906/README.md`. No new campaign is
   submitted; source isolation and accounting precede the eventual handoff.
2. **Square loose grid completion.** Fill the five missing cells of the square
   `3 x 3` `(t0,V)` grid with one smooth pairing-access seed.
   See `docs/SQUARE_SMOOTH_PAIRING_GRID_2026-09-03.md`.
3. **Cubic-unfrustrated loose grid completion.** Fill the eight cells not
   represented by the legacy `(1.0,0)` point using the matched seed protocol.
   See `docs/CUBIC_UNFRUSTRATED_SMOOTH_PAIRING_GRID_2026-09-03.md`.
4. **Square stripe-basin test at `(1.4,-0.4)`.** Determine whether the inherited
   high-amplitude legacy stripe is stable and, only if both endpoints are
   accepted and fingerprint-compatible, compare it with the paired control.
   See `docs/SQUARE_T014_VM04_LEGACY_STRIPE_COMPARISON_2026-09-03.md`.

## Completion sequence

1. Show the finite-size seed snapshot for user review. Prepare a submission
   handoff afterward, preserving the code used by current jobs. Wait for the
   pending grids and V=-0.4 stripe comparison; do not duplicate them.
2. After terminal status, reconcile requested ceilings against authoritative
   `sacct` elapsed time before choosing more compute.
3. Synchronize only the relevant compact results, logs, manifests, `jobs.tsv`,
   and ledger snapshots. Leave full MPS artifacts on scratch.
4. Run compact-only local verification and reproduce the campaign audit in a
   new output directory.
5. Report accepted/candidate/excluded counts, convergence evidence, spatial
   profiles, discarded weight/maxlinkdim, and only authorized energy rankings.
6. Update `docs/PROJECT_STATE.md`; append evidence, commands, failures, and the
   decision to `docs/RUN_LOG.md`.

## Compute horizon

The loose grids are coverage data. The next expensive choices should be made
only after their outcomes and the `chi=400` comparison are known. Preserve the
majority of the 400-additional-node-hour cap for selected higher-bond-dimension
and length checks, discarded-weight extrapolation, and any scientifically
necessary basin controls. A first-segment reservation does not authorize all
possible continuation segments.

## Exit condition for this plan

Replace this plan when the synchronized evidence has been audited and the next
calculation is chosen with a live Perlmutter cost envelope. Preserve the
completed campaign records and append the transition to `docs/RUN_LOG.md`.
