# Current project state

Last locally reviewed: **2026-09-22 — four V=-1 trellis runs prepared**

This is a local, mutable snapshot. Stable rules live in AGENTS.md and the
method documents; durable history is append-only in RUN_LOG.md. Local
artifacts establish solver outcomes, not live scheduler state.

## Repository and workflow

- Branch: `codex/mps-mft-phase0-refactor`; baseline for this update: `68af0ed`.
- Root `.claude/` is unrelated and remains untouched.
- Output/state files are excluded from Git and immutable. Reports, scripts,
  LaTeX/PDF notes and the Overleaf bundle are maintained together.
- Only the user transfers data or operates Perlmutter. Its ladder checkout is
  `$CFS/m4863/MPS-MFT/ladder_mps_mft`. No local authentication, transfers,
  submissions, scheduler calls or budget-ledger edits accompany this preparation.
- The 400-additional-node-hour project control remains in force. Historical
  reservations do not authorize new continuations. Preserve submitted source
  trees and each campaign's original controls/fingerprints.

## Current scientific evidence

**Complete pair correlations and square A/B results:** the
[September 22 report](reports/pair_correlations_20260922/README.md) verifies
all 56 retrospective branches / 58 MPSs and all eight new square A/B sidecars.
Retrospective states remain unaccepted snapshots; the four square A/B runs
are accepted fixed points. Both seeds reach paired states at (1.4,-0.4)
in 40 sweeps and at (1.4,-0.2) in 55/44 sweeps (stripe/pairing). Their
seed energies agree within 5.3e-11 t/site. A/B pairing profiles and connected
pair correlations reproduce the earlier one-ladder paired endpoints.

Stripes retain short-distance pair correlations, strongest near hole-rich
magnetic walls, while long-distance correlations are strongly suppressed.
At matched intraladder parameters, cubic stripes have roughly 2200–2300
times weaker connected rung correlations at separations 16–24 than square
paired endpoints; striped two-ladder trellis is about 70 times weaker than
paired one-ladder trellis. Rung–leg signs predominantly survive. Finite-L,
finite-chi, acceptance and reference/window qualifications remain explicit.
The isolated comparison uses chi=1200 versus 200 in coupled states and lacks
cross-channel data. No phase-stiffness or fluctuating-superconductivity claim.

The manuscript now includes these results and five new figures. The report
has seven figures plus complete source/coverage, correlation, fit-window,
bulk-cut and square fixed-point audits. No new simulations were run.

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
  accepted orbit or physical dynamics. A/B are spatial ladders. A September
  20 evaluation of paired copies in the same rectangular cell confirms
  that striped trials are lower by 0.002238–0.002308 t/site; the paired
  embedding shift is only 2.464e−6. This is a direct trial-energy comparison,
  not a certified stationary/global phase minimum. [Trellis report](reports/trellis_progress_20260918/README.md).

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

The physical distinction is relative stripe registration, which changes
interladder energy, versus translating all ladders together. Uniform bulk
pairing is compatible with both repetitions more readily than modulated
stripe order. Independent A/B profiles are therefore a necessary phase
competition check; the one-ladder paired result is a restricted-cell outcome.
The focused trellis test is to start the rectangular cell from the actual
paired trellis endpoint and compare its response to weak relative-stripe
perturbations with a stationary stripe branch in the same cell. No such
test has been prepared or performed, and existing cells are not strictly
nested for arbitrary finite OBC profiles.

The same-cell energy evaluation above requires no new optimization. It
removes the embedding ambiguity for these specified trials, while the
proposed relaxation/stability test remains future work. A diagonal stripe
with a continuing phase advance is another plausible competitor: A/B
repetition at the observed 126–131-degree charge offset instead alternates
the phase advance. A larger rectangular cell must close both charge and
spin periods under its chosen tilt; its ladder count need not equal the
longitudinal charge wavelength. No larger trellis cells have been prepared.

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

Numerical evidence includes the September 20 same-cell energy audit and
September 22 complete correlations and square A/B results in Sections
3.13/3.14. The manuscript has 43 pages and 18 figures. The material/trellis
introduction and 48 cited references from September 18 are preserved;
the literature-search cutoff is unchanged. Living methods notes and the
literature review's project-evidence section incorporate the final results.
The Overleaf bundle includes every manuscript figure.

The September 20 living-methods update distinguishes R composed with itself
(lambda squared) from the simultaneous spatial-cell Jacobian (even/odd
eigenvalues plus/minus lambda), and explains leg parity and trellis shear.
The manuscript and its Overleaf bundle also include the September 20 trial
energy comparison and the diagonal-stripe commensurability distinction.

## Correlation measurements: complete synchronized coverage

Retry1 has 49 receipts / 51 spatial MPS diagnostics; retry2 supplies the
remaining seven receipts / MPSs. The September 22 analysis checks all
manifest/receipt hashes, exact retry subset, full-source identifiers,
compact-state/config hashes, Hermiticity, Gram positivity, connected
subtraction and density/spin agreement. Frozen artifacts remain unchanged.
New square A/B adds eight verified intraladder diagnostic files. The
[report and reproducible evidence](reports/pair_correlations_20260922/README.md)
supersede all earlier incomplete-measurement status. Startup/timeout retry
history remains in RUN_LOG and the original recovery handoffs.

Terminal measurements still run on CPU inside GPU allocations. The first
square A/B result required 99.2/99.4 minutes for its sequential passes;
no optimized GPU measurement benchmark or performance implementation is
part of this analysis. No new submission is needed for this completed set.

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

User requested the same four trellis starts at V=-1.0. The
[September 22 preparation](reports/trellis_vm1_20260922/README.md) provides
one-/two-ladder cells with stripe/pairing 95%/5% seeds, unchanged chi=200,
L64 and 60 raw sweeps, exact E_p=-0.2713195876256691, and full terminal
correlations. Run `bash slurm/submit_trellis_vm1_comparison.sh` on Perlmutter
after `git pull --ff-only` from the checkout above. Four one-GPU jobs request
16 hours each (16 node-hours total ceiling), one segment each, retaining the
11.5-hour solver deadline and shared budget gates. Prepared locally only;
no submission or job IDs have been reported. Keep the checkout fixed while
the jobs run. Previous V=0 campaigns and their artifacts are unchanged.

Review the completed correlation report and manuscript Sections 3.13/3.14.
All requested measurements and all four square A/B endpoints are analyzed;
no further backfill or square A/B submission is needed. Updated reports and
sources use the standing commit/push and `git pull --ff-only` workflow on
`codex/mps-mft-phase0-refactor`, with no individual file-transfer handoff.

The next scientific controls are stationary stripe branches, matched L/chi
comparisons, and selective cell/stability tests. The split square points
and alternating two-ladder trellis need a separate convergence decision.
Modest linear damping could test the negative trellis iteration mode, but
no damping/stability follow-up is prepared. The V=-1 repetition above keeps
the original raw-map controls. Cubic cells and larger transverse periods
remain recommendations. Measurement performance profiling is also separate.

Archived scientific states and ledgers remain read-only. This update uses
focused real-data analysis assertions, LaTeX compilation, rendered-page QA
and an extracted Overleaf-root build. No DMRG or measurement contractions,
GPU timing, transfer or scheduler action were performed locally. Earlier
implementation/launcher tests remain recorded in the append-only run log.
