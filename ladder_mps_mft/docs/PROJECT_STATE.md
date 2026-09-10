# Current project state

Last locally reviewed: 2026-09-10 (40-step remainder prepared and locally qualified)

This is the canonical mutable snapshot for resuming work. It is deliberately
short. Stable rules belong in `AGENTS.md` and the method documents; durable
history belongs in `docs/RUN_LOG.md`.

## Repository snapshot

- Branch: `codex/mps-mft-phase0-refactor`
- Baseline commit for the September 4 systematic review: `a744d29` (project
  continuity update).
- The September 4 review's uncommitted report/evidence directory and
  documentation updates were preserved during the September 5 chi=400
  analysis. The latter adds its own evidence directory and status updates. An
  untracked root `.claude/` directory exists outside this subproject and is not
  part of this work.
- Local `output/` and simulation HDF5 files are excluded from Git. The small
  `data/two_basin_references.h5` is a versioned seed-input exception.
- Current local work adds explicit reference-length E_p support and four
  L=96/128 seed-review configs. This changes the implementation fingerprint;
  preserve the old Perlmutter source checkout for pending/running campaigns.

Recheck the branch, commit, and working tree at the start of each new task.

## Current scientific position

- Manuscript reference: the 2023 repulsive-ladder benchmark's mean-field
  equations match our `cubic_frustrated` geometry. The equation-level evidence
  and wording are in the [literature source notes](literature/SOURCE_NOTES.md#manuscript-note-geometry-of-the-2023-ladder-benchmark).
- Phase 0 is closed. Phase 1 uses the refactored Float64 dense-CUDA path.
- The user approved the legacy stripe at `(1.0,0.0)` and explicit uniform
  d-wave lineage at `(1.4,-0.4)` as two reference profiles, each with a weak
  contribution from the other. All 18 square starts are locally prepared as
  95%/5% correlation mixtures rebuilt with target couplings. New comparisons
  use chi=200 and **no Anderson**. The four original anchors allow 80 raw
  evaluations and require at least 50 before acceptance. See the
  [prepared campaign and handoff](reports/two_basin_raw_20260908/README.md).
  `slurm/submit_square_two_basin.sh` prepares and submits the four anchors from
  the normal checkout after `git pull`, using the versioned reference input
  and existing shared budget gates. The V=-0.4 pair is now locally analyzed:
  both reached 80 evaluations and the same d-wave-like pairing basin, but
  remain unaccepted because of weak-channel noise-floor checks, a scalar
  charge extrapolation over-sensitivity, and one marginal energy-window miss.
  User-supplied sacct records give 3.763125 node-hours for this pair. The V=0
  anchors remain PENDING in the September 10 job-specific output. The user
  now authorizes the remaining fourteen with a 40-evaluation cap, minimum 30,
  ten stable records, a 5e-7 channel noise floor plus full-window drift gate,
  and a 2e-8 t/site energy span. Global/inner-DMRG tolerances are unchanged.
  Saved-history gates first pass at 35/30 for the stripe/pairing anchors;
  older growing V=0 raw histories still fail. Missing per-iteration identity
  errors prevent retrospective acceptance certification. The new scope is
  locally prepared, not submitted. Its separate-checkout launcher shares
  existing accounting and leaves pending source paths unchanged. See the
  [remainder contract](reports/two_basin_remainder_20260910/README.md).
  See the [V=-0.4 analysis](reports/two_basin_vm04_20260910/README.md).
  Higher chi, length, and stripe-wavelength work remains deferred.
- At square `(t0,V)=(1.4,0)`, the completed loose `chi=200` six-seed campaign
  produced six software-accepted, pairing-bearing period-one endpoints.
  September 8 full-history review revises the basin interpretation: three
  runs stop after six map evaluations, while three show coherent raw SDW
  growth followed by suppression under Anderson mixing. They do not establish
  a unique stable paired basin. The selected run's plotted 23-to-24 drop is
  reproduced by first-Anderson coefficients `+22.6811,-21.6811` to `1.57e-15`.
  The switch follows the configured 20 raw steps and one linear startup step;
  it does not require spin growth to have settled.
  Its spin profile changes, so the earlier growth does not by itself prove
  terminal instability. Controlled competing-order perturbations are needed.
- At `(1.4,-0.4)`, tested small stripe/coexistence seeds show coherent spin
  decay (raw projection gains about `0.60–0.61`), supporting attraction toward
  pairing for those perturbations. The September 10 raw 95%/5% comparison now
  shows the strong-stripe seed also relaxing into the paired basin at chi=200:
  final leg-odd spin Hartree RMS is about 2.5e-8–3.9e-8 and both pairing/charge
  profiles coincide. Terminal corrected energies differ by only 7.49e-9 t/site;
  this is numerical agreement, not an accepted-state ranking. See the
  [September 8 basin assessment](reports/square_grid_20260908/BASIN_ASSESSMENT.md).
- The matched square `(1.4,0)` `chi=400` jobs have terminated, but neither
  endpoint is accepted: pairing is `stagnated` after 32 records and the
  legacy-like stripe is `time_limit` after 40. Their fingerprints match and
  the two spatial textures remain distinct. The stripe's terminal corrected
  energy is lower by `2.43361262e-4 t/site` (`0.03115024 t` total), a
  stored terminal difference. On September 6 the user judged the paired
  identity error negligible for this energy comparison, treated it as
  converged, and judged the stripe plateau/downward residual sufficient to
  proceed. The working conclusion is that the stripe's energy advantage is
  reasonably robust to the tested chi=200 to 400 increase. Original solver
  status/acceptance flags remain unchanged.
- Pairing passes the final field/slow-mode/density/energy-stability gates but
  fails the `1e-10 t/site` Hamiltonian-identity gate (`2.68869e-10`). The stripe
  misses the `1e-4` field gate (`2.49005e-4`) and ends during DMRG. Neither
  final solve meets its last-sweep energy tolerance. See the
  [September 5 chi=400 analysis](reports/chi400_comparison_20260905/ANALYSIS.md).
- The coverage surveys use one center-of-mass-uniform smooth-pairing access
  seed for the square and cubic-unfrustrated `3 x 3` `(t0,V)` grids. The square
  fill is now terminal and compiled; the cubic campaign is still running per
  the September 8 user report.
  They test coverage and basin access at loose `chi=200`; they do not prove
  basin uniqueness.
- A separate square `(1.4,-0.4)` comparison tests whether the inherited
  high-amplitude legacy stripe remains a self-consistent competing basin.
- Four new chi=200 length-study seeds are locally ready for user review:
  pairing and stripe at L=96 and 128, using the latest L=64 chi=400 measured
  fields as templates. Each original half is preserved; pairing receives a
  uniform middle and stripes receive one/two 32-rung bulk cells. The user
  explicitly selected fixed L=64 E_p. See the
  [seed snapshot and method](reports/finite_size_seeds_20260906/README.md).

The [September 4 systematic review](reports/systematic_review_20260904/REVIEW.md)
audits local evidence and recommends focused implementation/science follow-up;
it does not change solver behavior or certify a phase. Of 28 historical stored
accepted flags, 17 survive the limited existing history screen; the six newest
square V=0 states all survive that global screen and match their compact
manifests; the September 8 channel analysis shows why this is not a stability
certification. Their corrected
energy spread is only `6.0274e-7 t/site`, with fine ordering unresolved. Priority
code recommendations concern implementation hashing, invalid selection inputs,
and propagation of inner-DMRG convergence. The finite-size even-particle charge
gap is not the denominator of the pair-breaking projection; the report corrects
that interpretation while retaining weak-hopping and convergence controls.

## Campaign inventory

The [September 8 square-grid snapshot](reports/square_grid_20260908/README.md)
compiles all nine chi=200 coordinates into one HDF5: six software-accepted small-seed
fixed points, two user-approved legacy coverage points, and the flagged
`(1.0,-0.4)` diverging endpoint. The five-point square fill is locally terminal
(four accepted, one diverging). Fourier plots and a focused divergence analysis
are available. The bundle now also retains every selected run's complete saved
histories; grid clicks open the MF profiles/middle histories with the iteration
slider. The divergence analysis shows a slow stripe-like trajectory and an
accelerated residual spike, not an accepted solution. Original statuses remain
unchanged. A separate saved-history analysis explains the V=0 SDW jump and
records the motivation for the now-prepared two-family raw comparison.
Two of its 18 prepared starts now have synchronized 80-iteration histories;
their September 10 analysis is linked above.

These are the newest locally documented campaigns. Their live scheduler state
must not be inferred from the repository.

| Run ID | Local record | Question |
|---|---|---|
| `20260910_square_two_basin_95_5_40_remainder` | 14 seeds/configs locally prepared; 75 Julia assertions and mock launcher test pass; not submitted | Proceed with user-authorized remainder using qualified noise handling; separate checkout and shared ledger |
| `20260908_square_two_basin_95_5_80_anchors` | V=-0.4: both 80-step states synced/analyzed, unaccepted paired plateaus, 3.763125 node-hours; V=0 PENDING per user | Let original pending anchors finish; review synced outcomes |
| `20260903_phase1_square_t014_v000_pairing_legacy_chi400_tight` | synced; 0/2 software-accepted; user accepts energetic comparison as adequate | Retain as the L=64 energy/basin reference; higher-chi checks remain future work |
| `20260903_phase1_square_grid_smooth_pairing_chi200_loose` | five terminal states synced; 4 accepted, 1 diverging; compiled September 8 | Review Fourier grid and divergence at `(1.0,-0.4)` |
| `20260903_phase1_cubic_unfrustrated_grid_smooth_pairing_chi200_loose` | user reports still running September 8; two local terminal files seen, not analyzed here | Finish eight-point cubic-unfrustrated fill |
| `20260903_phase1_square_t014_vm04_legacy_stripe_compare_chi200_loose` | user reports still running September 8; no local terminal states | Test inherited stripe stability and, conditionally, its energy against the paired control |
| L=96/128 square chi=200 length comparison | four local field seeds and review configs; no scheduler campaign | Review the enlarged seeds before submission preparation |

Campaign contracts and exact run commands are in the corresponding dated
documents linked from `docs/plans/ACTIVE.md`.

## Live Perlmutter and accounting boundary

- Latest supplied evidence, 2026-09-10: jobs 58093799 (stripe) and 58093800
  (pairing) at V=-0.4 are COMPLETED in pasted sacct output, with elapsed
  seconds 30799 and 23390. Their combined project charge is 3.763125 node-hours.
  Their local histories confirm maximum_iterations after 80 evaluations each.
  Jobs 58093802/58093803 at V=0 are PENDING in the user's status output.
  The latest prose refers to pending `(1.0,0.0)` whereas those job labels are
  `(1.4,0.0)`; clarification was requested because a separate `(1.0,0.0)`
  submission would overlap the prepared remainder.
  No automatic remainder or continuation submission is configured. The local
  reconciliation ledger has no entries yet for these four jobs; analysis
  does not modify accounting, and 2.236875 node-hours of the finished pair's
  6-hour reserved ceiling are eligible for release through normal reconciliation.
- Earlier user report, 2026-09-08: queued runs were canceled (job IDs not
  provided). The direct submission wrapper reconciles finalized reservations
  through the existing `sacct`-based accounting before submitting. Cancellation
  and released compute are not independently verified in this local snapshot.
- User-reported on 2026-09-08: the square fill finished; cubic-unfrustrated and
  `(1.4,-0.4)` stripe/control runs are still running. Local square artifacts
  confirm five terminal solver outcomes (4 accepted, 1 diverging). This does
  not provide live scheduler verification or reconciled accounting. Synced
  submission files record 5, 8, and 2 jobs; IDs are in the September 5 analysis.
- Chi=400 jobs `57905744` and `57905745` produced terminal solver artifacts;
  this does not supply reconciled `sacct` accounting. Do not resubmit or cancel
  work based only on this snapshot.
- The hard project control remains 400 additional node-hours. The last locally
  documented reconciled active total was `42.024097222` node-hours before the
  newest campaign reservations. It is historical, not a current allowance.
- The live append-only Perlmutter reservation and reconciliation ledgers plus
  `sacct` measurements are authoritative before any compute decision.
- Preserve substantial budget for later bond-dimension and length convergence;
  requested first-segment ceilings are not authorization for blanket
  continuations.

## Exact next action

The user authorizes the fourteen remaining starts now. Publish the locally
validated revision, then use the separate-checkout user-run commands in
`docs/reports/two_basin_remainder_20260910/README.md`. The source checkout used
by pending anchors must remain at its submitted revision. Confirm the pending
coordinate if it is a separate `(1.0,0.0)` submission; the prepared remainder
excludes only the known `(1.4,-0.4)` and `(1.4,0.0)` anchor pairs.
No new jobs, live checks, or reservations have been performed locally. The
14 starts allow at most 560 evaluations and reserve at most 42 fractional
node-hours under the unchanged 12-hour segment ceiling and live 400-hour cap.
Actual runtime can be lower. Retain incomplete outcomes at the 40-step limit
for analysis; no automatic extensions. Let the existing anchors finish and
analyze their synchronized results under their original provenance.

The four finite-size seeds remain ready for review; their configs keep tight
comparison controls at chi=200, use an identity gate of 1e-8 t/site, and hold
E_p at the measured L=64 value with `pair_binding.reference_L=64`. Their prior
four Python tests and 99 Julia assertions passed without DMRG. Keep this
preparation available for the deferred length study. No new jobs or
reservations exist; reconcile actual accounting before compute.

Review the compiled square grid and its flagged divergence analysis. Wait for
the cubic and `(1.4,-0.4)` comparison campaigns. Do not submit duplicates or cancel them
from this repository state. After the user reports terminal status and
synchronizes the relevant compact results, logs, manifests, `jobs.tsv`, and
ledger snapshots:

1. verify the compact/stateless artifacts locally without requesting full
   scratch verification;
2. classify convergence using the raw-map recurrence, oscillation, slow-mode,
   density, and energy-stability gates;
3. rank only accepted same-geometry, same-Hamiltonian states with matching
   fingerprints through the canonical variational energy;
4. update this file and append the evidence and decision to `docs/RUN_LOG.md`;
5. choose any continuation or higher-`chi` calculation only after reconciling
   the authoritative ledger.

## Verification boundaries

- Local compact verification does not prove the recorded full scratch artifact
  is present or hash-valid.
- Scheduler and accounting claims require current user-provided Perlmutter
  output or a synchronized ledger snapshot.
- Accepted finite-`L`, finite-`chi` Phase 1 endpoints are variational states,
  not thermodynamic phase assignments.
- Never compare canonical energies across transverse geometries or different
  Hamiltonian points.
