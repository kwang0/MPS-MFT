# Current project state

Last locally reviewed: **2026-09-20 — transverse order and spatial-cell interpretation**

This is a local, mutable snapshot. Stable rules live in AGENTS.md and the
method documents; durable history is append-only in RUN_LOG.md. Local
artifacts establish solver outcomes, not live scheduler state.

## Repository and workflow

- Branch: `codex/mps-mft-phase0-refactor`; baseline for this update: `13b289d`.
- Root `.claude/` is unrelated and remains untouched.
- Output/state files are excluded from Git and immutable. Reports, scripts,
  LaTeX/PDF notes and the Overleaf bundle are maintained together.
- Only the user transfers data or operates Perlmutter. Its ladder checkout is
  `$CFS/m4863/MPS-MFT/ladder_mps_mft`. No local authentication, transfers,
  submissions, scheduler calls or budget-ledger edits accompany this analysis.
- The 400-additional-node-hour project control remains in force. Historical
  reservations do not authorize new continuations. Preserve submitted source
  trees and each campaign's original controls/fingerprints.

## Current scientific evidence

The [combined campaign review](reports/campaign_review_20260918/README.md),
updated September 19, covers **38 completed runs, 2280 cell updates and
2400 individual ladder solves**. Every run reaches 60 steps and remains
unaccepted. The earlier coarse square grid is a separate reference.

- **Square coarse grid:** both seeds agree on seven stripe and two paired
  coordinates. The paired points are (1.4,−0.4) and (1.4,−0.2). Both (1.4,0)
  continuations are included, with 100/82 cumulative evaluations. There are
  902 evaluations across 18 lineages/20 source jobs, zero accepted endpoints.
  [Full square report](reports/two_basin_grid_20260915/README.md).
- **Cubic unfrustrated:** all 18 starts reach essentially unpaired stripes,
  including the square paired corner. Spin RMS 0.280–0.356; maximum leg-pair
  RMS 2.21e−11. Energy windows pass but field/profile/slow-mode gates fail.
  [Cubic report](reports/cubic_two_basin_grid_20260918/README.md).
- **Finer square cuts:** all twelve starts complete. Four coordinates are
  paired from both seeds; (1.25,−0.4) and (1.4,−0.05) retain distinct textures.
  Pairing-minus-stripe endpoint gaps −3.19e−5/−5.95e−5 t/site are diagnostics.
  At paired (1.4,−0.05), 96.7% of spin-squared weight lies in the outer 14
  rungs at each end and central spin still decays. Full energy curves show
  a small V-slope downturn (0.38%) and gradual t0 curvature, not a resolved
  first-order kink. New physical spin/pairing cuts use the same twenty
  endpoint sources and matched RMS definitions. [Fine-cut report](reports/square_fine_cuts_20260918/README.md).
- **Square (1.2,+0.2): all four final states are now synced.** All lose pairing.
  Stripe, pairing and period-16 starts reach spin RMS about 0.233 and nominal
  charge/spin-envelope periods 16/32. The period-eight seed reaches spin
  0.207 and pair RMS 7.01e−9, with six unevenly spaced spin nodes and dominant
  charge m=4. Its excess 9.3242e−4 t/site over the stripe-seeded endpoint is
  diagnostic; spatial drift prevents acceptance. This is consistent with a
  defect/coarsening remnant, not sustained intertwining. [Positive-V report](reports/square_positive_v_20260918/README.md).
- **Trellis: all four final states are now synced.** Both skew one-ladder
  seeds approach the same paired texture (pair RMS 0.017969, spin about
  1.3e−5, opposite leg/rung signs). Both rectangular A/B starts instead
  develop essentially unpaired stripes (spin 0.2282–0.2288). Two-ladder
  global residuals 0.1845–0.2085 remain large despite passing final inner-DMRG
  windows. Successive field increments nearly reverse (cosines about −0.99995)
  with norm ratio about 0.975: slowly damped alternating relaxation, not an
  accepted orbit or physical dynamics. A/B are spatial ladders. Diagnostic
  two-ladder energies are lower, but differing cell constraints/open ends and
  large residuals prevent a certified ranking. [Trellis report](reports/trellis_progress_20260918/README.md).

The [September 20 transverse-sector audit](reports/trellis_progress_20260918/TRANSVERSE_INTERPRETATION_20260920.md)
finds 79-83% A/B-odd weight in the dominant endpoint charge harmonic after
correcting the half-rung registration. Central charge modulation is about
0.047, versus only 5-7e-6 leg-odd charge RMS: the stripe charges both legs
together. This is distinct from the old leg-parity k_y=pi label. Both trellis
cells remain geometrically frustrated; skew and rectangular repetition impose
different stripe registrations. A same-map bipartite two-cycle can encode
static A/B order, but that equivalence does not identify these two trellis
ansatzes. Square runs already used raw updates and allowed periods 1 and 2;
their paired outcomes are evidence, not a complete transverse stability test.

The 95%/5% reference mixtures use the legacy stripe at (1,0) and paired
state at (1.4,−0.4), rebuilt with target couplings. Raw updates and no
Anderson are retained. Flat energy or RMS cannot certify a stationary
profile; old loose Anderson acceptance is not stability evidence.
The user's September 6 chi=400 working energetic interpretation remains
in its [dated report](reports/chi400_comparison_20260905/ANALYSIS.md), with
original unaccepted flags. Larger chi/length/wavelength and precise E_p
sensitivity work remains deferred; prepared L96/L128 seeds are unchanged.

## Reports and actual LaTeX notes

The [manuscript](manuscript/README.md) now has comprehensive separate
Sections 3.11 (positive-V square) and 3.12 (trellis). Section 3.10 includes
physical spin/pairing cuts, alongside its full canonical energy plots.
Square/cubic phase diagrams, full energy grids and physical spin/pairing
RMS grids remain side by side, preserving all 1982 plotted evaluations
and square continuation markers. Energy panels keep individual y scales.

Numerical evidence now extends through September 19. The material/trellis
introduction and 48 cited references from September 18 are preserved;
the literature-search cutoff is unchanged. Living methods notes and the
literature review's project-evidence section incorporate the final results.
The Overleaf bundle includes every manuscript figure.

The September 20 living-methods update distinguishes R composed with itself
(lambda squared) from the simultaneous spatial-cell Jacobian (even/odd
eigenvalues plus/minus lambda), and explains leg parity and trellis shear.
The manuscript and its Overleaf bundle retain the completed September 19
results; the new detailed interpretation is linked in the trellis report.

## Correlation measurements: prepared, not new evidence

Full equal-time raw/connected pair–pair, charge/spin, density-spin,
single-particle/anomalous, double-occupancy and entanglement measurements
are enabled for future accepted **and maximum-iteration** states.
Archived status and convergence flags remain unchanged. Retrospective
measurement covers 56 latest branches / 58 spatial MPSs, with the two
old square V=0 endpoints superseded by their continuations.

All previously missing terminal states are present in this sync. This is
not a fresh full-MPS/backfill preflight: full scratch availability and hashes
must be checked by the user-run workflow. No new campaign pair–pair
measurement was analyzed here. Anomalous pairing disappearing does not
determine the surviving connected pair correlations.

The [existing CPU backfill handoff](DIAGNOSTICS.md#user-run-perlmutter-backfill)
has an 8.15625-node-hour requested ceiling. The user asked whether GPU would
be faster: CPU was chosen because it is the existing measurement path,
not from a CPU/GPU benchmark. No GPU port or new submission is implied.

## Accounting from synchronized evidence

All 38 completed jobs in this review now have validated reconciliations:

| Campaign | Solver-only node-hours | Actual node-hours |
|---|---:|---:|
| Cubic, 18 jobs | 13.886995 | 14.196667 |
| Fine square, 12 jobs | 13.030485 | 13.274722 |
| Positive V, 4 jobs | 4.523293 | 4.602431 |
| Trellis, 4 jobs | 8.627653 | 8.708542 |
| **Total** | **40.068427** | **40.782361** |

[Job-level audit and ledger hash](reports/campaign_review_20260918/completion_accounting.json).
Coarse square separately costs 27.350764 actual node-hours including
continuations. Do not add historical reservations or superseded estimates
to these totals. Accounting/solver time are distinct; no ledger was edited.

## Next action and boundaries

Review the completed reports and LaTeX/PDF. New connected-correlation
interpretation awaits user-run measurement and sync; do not resubmit the
completed SCF campaigns. The split square points and alternating two-ladder
trellis are candidates for a future selective convergence/stability decision.
Modest linear damping could test the negative trellis iteration mode, but
no controls or follow-up jobs have been prepared in this analysis.

Prioritize a targeted square A/B comparison at (1.4,-0.2) and (1.4,-0.4),
using paired and translated-stripe starts with explicit transverse
perturbations. The square spatial map must reproduce the existing map and
per-site energy when A=B. The current rectangular trellis tau1=0 switch is
not a substitute for that bond-level check. Cubic transverse-registration
tests are useful but lower priority for the paired/stripe boundary. These
are recommendations, not new run preparations or submissions.

All analysis is local and read-only with respect to simulations. Validation
checks hashes, physical profiles, full histories, canonical normalization,
gate windows and accounting, plus figure/PDF rendering and Overleaf paths.
No new DMRG calculation or expensive full test suite is needed for this update.
