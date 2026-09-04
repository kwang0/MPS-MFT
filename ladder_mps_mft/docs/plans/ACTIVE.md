# Active plan

Last reviewed: 2026-09-04

Current state: waiting on Perlmutter. The user reports that the three latest
jobs are pending; their exact IDs and campaign membership have not been
verified locally. This document is not submission or cancellation authority.

## Active scientific questions

1. **Square `chi=400` lineage comparison.** Determine whether pairing and
   legacy-like parents remain distinct under common tighter controls, and how
   the observables move from `chi=200` to `chi=400`.
   See `docs/SQUARE_V0_T014_CHI400_COMPARISON_2026-09-03.md`.
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

1. Wait for current jobs; do not create duplicate submissions or blanket
   continuations.
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

