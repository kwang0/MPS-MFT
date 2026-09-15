# Active plan

Last reviewed: 2026-09-15

September 15 update: the full square two-basin grid is synchronized and
analyzed in `docs/reports/two_basin_grid_20260915/README.md`. Eighteen runs
provide 862 MF evaluations and use 26.798333 actual allocation node-hours.
The preliminary diagram assigns seven stripe points (including the still
converting t0=1.4,V=0 pairing start) and two paired points at t0=1.4,V=-0.4/-0.2.
All endpoints remain formally unaccepted. There is no accepted energetic
selection. Following review, the user authorized the same grid for
cubic_unfrustrated and a short square (1.4,0) continuation. Both are now
prepared: see `docs/reports/two_basin_next_campaigns_20260915/README.md`.
The two user-run launchers submit 18 cubic starts (max 60, minimum 40)
and two square continuations (20 additional, ten fresh stable records),
all chi=200 with raw updates. Energy tolerances relax modestly; resolved
field drift remains disqualifying. No new jobs submitted locally.

Current state: the chi=400 comparison is locally analyzed. Four L=96/128
chi=200 seeds remain available for the now-deferred finite-size study.
The square fill now has five synced terminal states (four
accepted, one diverging), and the full chi=200 square grid is compiled with
explicit legacy/divergence flags. The user reports the cubic-grid and V=-0.4
stripe/control campaigns are still running. Their submission IDs are synced;
live scheduler state is not independently verified. This document is not
submission or cancellation authority.

September 8 follow-up: full histories expose coherent SDW growth and Anderson
cancellation in the loose V=0 six-seed results. The user approved representative
d-wave/stripe seeds with reciprocal weak perturbations and rejected Anderson
for new comparisons. All 18 square starts are locally prepared with 95%/5%
correlation mixtures, chi=200, up to 80 raw evaluations, a minimum of 50,
and tighter channel/energy/inner-DMRG gates. The V=-0.4 pair now has two synced
80-evaluation histories: both reach the same paired basin, with acceptance
blocked chiefly by weak-channel fluctuations and a scalar-extrapolation issue.
Their user-supplied allocation charge is 3.763125 node-hours. The V=0 pair
is now synced and analyzed: stripe ends at 80 with negligible pairing;
the pairing start ends at the deadline at 62 with spin growing and pairing
collapsing. Both remain unaccepted for resolved field motion. September 15
accounting establishes 4.444722 actual node-hours for this V=0 pair.
The fourteen remaining starts have completed their 40-step cap with revised
noise handling, using 18.590486 allocation node-hours. No automatic
continuation is configured.

## Active scientific questions

1. **Next two-basin campaigns (prepared, awaiting user submission).** Run
   `slurm/submit_cubic_unfrustrated_two_basin.sh` and
   `slurm/submit_square_two_basin_finish.sh` from the normal Perlmutter
   checkout after pulling. The first spans all nine coordinates with two
   95%/5% seeds. The second restores both V=0 MPS lineages and retains full
   plotting ancestry. Combined reservation ceiling: 58 node-hours;
   shared accounting, one segment each, no automatic further runs.
2. **Competing basins with raw updates (both anchor points analyzed).** Review
   `docs/reports/two_basin_v000_20260912/README.md`: the pairing-dominated
   V=0 start evolves toward a similar stripe texture, but its residual pairing
   is still changing rapidly at the deadline. The stripe start has nearly
   flat energy but persistent coherent spatial drift. Neither is fixed-point
   accepted under the original or revised necessary global-field gate.
   The V=-0.4 paired control remains in
   `docs/reports/two_basin_vm04_20260910/README.md`. Both V=-0.4 lineages reach
   a paired plateau by roughly 20–30 evaluations; late unaccepted flags do not
   indicate continuing stripe growth. The revised remainder controls require
   30 evaluations and ten stable records, with a 5e-7 channel noise floor,
   full-window channel-span guard, and 2e-8 t/site energy range. Global and
   inner-DMRG gates remain unchanged. Saved-history gates first pass at
   35/30 for the paired anchors, while old growing V=0 raw histories fail.
   See `docs/reports/two_basin_remainder_20260910/README.md` for qualification,
   limitations, and separate-checkout handoff preserving pending anchors.
   Keep all channels free and compare accepted matching corrected canonical
   energies, retaining unresolved gaps. State ancestry does not determine
   final order. Use the current checkout, versioned reference input, and
   existing shared budget.
3. **Square finite-size comparison (deferred).** Retain the prepared pairing/stripe seeds
   at L=96 and 128, chi=200. The L=64 energetic advantage of the stripe is
   `2.43361262e-4 t/site` at chi=400, and the user regards its bond-dimension
   robustness as sufficient motivation to study length next. Hold the L=64
   E_p fixed, retain the original edge halves, insert flat paired bulk or
   complete stripe cells, and compare energy differences within each length.
   See `docs/reports/finite_size_seeds_20260906/README.md`. No new campaign is
   submitted; source isolation and accounting precede the eventual handoff.
4. **Square loose grid completion.** Fill the five missing cells of the square
   `3 x 3` `(t0,V)` grid with one smooth pairing-access seed.
   Locally compiled September 8: review `docs/reports/square_grid_20260908/`.
   The `(1.0,-0.4)` terminal diagnostic remains unaccepted; no continuation
   is prepared. Campaign contract: `docs/SQUARE_SMOOTH_PAIRING_GRID_2026-09-03.md`.
5. **Cubic-unfrustrated loose grid completion.** Fill the eight cells not
   represented by the legacy `(1.0,0)` point using the matched seed protocol.
   See `docs/CUBIC_UNFRUSTRATED_SMOOTH_PAIRING_GRID_2026-09-03.md`.
6. **Square stripe-basin test at `(1.4,-0.4)`.** Determine whether the inherited
   high-amplitude legacy stripe is stable and, only if both endpoints are
   accepted and fingerprint-compatible, compare it with the paired control.
   See `docs/SQUARE_T014_VM04_LEGACY_STRIPE_COMPARISON_2026-09-03.md`.

## Completion sequence

1. The qualified 40-step implementation is published as `7443d9d`. All
   fourteen starts and four original anchors are now analyzed and accounted
   for locally. The September 15 follow-up now prepares cubic starts and
   targeted continuation of both (1.4,0) lineages. The pairing lineage is
   still evolving at 62; the late (1.2,-0.4) conversion and drifting
   (1.0,-0.4) texture informed the cubic minimum and cap.
   Do not loosen the thresholds again to accept these resolved drifts.
   No unchanged V=-0.4 extension is recommended solely to obtain acceptance.
   Retain the finite-size seeds for future work.
   Update older campaign status when job-specific results/accounting are synced.
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

The loose grids are coverage data. First-segment ceilings are 12 fractional
node-hours for four anchors and 42 for fourteen later starts. The stages are
not blanket continuation authority. Preserve the
majority of the 400-additional-node-hour cap for selected higher-bond-dimension
and length checks, discarded-weight extrapolation, and any scientifically
necessary basin controls. A first-segment reservation does not authorize all
possible continuation segments.

## Exit condition for this plan

Replace this plan when the synchronized evidence has been audited and the next
calculation is chosen with a live Perlmutter cost envelope. Preserve the
completed campaign records and append the transition to `docs/RUN_LOG.md`.
