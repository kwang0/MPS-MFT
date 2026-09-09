# Active plan

Last reviewed: 2026-09-08

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
and tighter channel/energy/inner-DMRG gates. No jobs or reservations exist.

## Active scientific questions

1. **Competing basins with raw updates (approved/prepared).** Review the
   concrete seeds and handoff in `docs/reports/two_basin_raw_20260908/README.md`.
   First run both families at `(1.4,0)` and `(1.4,-0.4)` under the new raw
   contract, then the fourteen remaining starts after reviewing the anchors.
   Keep all channels free and compare accepted matching corrected canonical
   energies, retaining unresolved gaps. State ancestry does not determine
   final order. Use the isolated source snapshot and existing shared budget.
2. **Square finite-size comparison (deferred).** Retain the prepared pairing/stripe seeds
   at L=96 and 128, chi=200. The L=64 energetic advantage of the stripe is
   `2.43361262e-4 t/site` at chi=400, and the user regards its bond-dimension
   robustness as sufficient motivation to study length next. Hold the L=64
   E_p fixed, retain the original edge halves, insert flat paired bulk or
   complete stripe cells, and compare energy differences within each length.
   See `docs/reports/finite_size_seeds_20260906/README.md`. No new campaign is
   submitted; source isolation and accounting precede the eventual handoff.
3. **Square loose grid completion.** Fill the five missing cells of the square
   `3 x 3` `(t0,V)` grid with one smooth pairing-access seed.
   Locally compiled September 8: review `docs/reports/square_grid_20260908/`.
   The `(1.0,-0.4)` terminal diagnostic remains unaccepted; no continuation
   is prepared. Campaign contract: `docs/SQUARE_SMOOTH_PAIRING_GRID_2026-09-03.md`.
4. **Cubic-unfrustrated loose grid completion.** Fill the eight cells not
   represented by the legacy `(1.0,0)` point using the matched seed protocol.
   See `docs/CUBIC_UNFRUSTRATED_SMOOTH_PAIRING_GRID_2026-09-03.md`.
5. **Square stripe-basin test at `(1.4,-0.4)`.** Determine whether the inherited
   high-amplitude legacy stripe is stable and, only if both endpoints are
   accepted and fingerprint-compatible, compare it with the paired control.
   See `docs/SQUARE_T014_VM04_LEGACY_STRIPE_COMPARISON_2026-09-03.md`.

## Completion sequence

1. Review the 18 prepared seeds and raw-map contract. The user transfers the
   source bundle, prepares the four anchors, and submits through the guarded
   launcher after checking live accounting. Retain the finite-size seeds for
   future work. Await the cubic grid and V=-0.4 legacy stripe comparison
   without disturbing their source checkout.
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
