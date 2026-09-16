# Current project state

Last locally reviewed: 2026-09-16 (square continuations and eight cubic results synced)

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
- The finer-cut campaign added explicit V-axis E_p interpolation and changed
  the solver implementation fingerprint. The new positive-V comparison adds
  preparation scripts and four seeds without further solver changes. Preserve
  submitted source checkouts and use each campaign's separate-worktree handoff.

Recheck the branch, commit, and working tree at the start of each new task.

## Current scientific position

- The [full square-grid report](reports/two_basin_grid_20260915/README.md)
  is now updated in place with both September 15 continuations. It joins
  complete V=0 histories to 100/82 evaluations and uses the new endpoint
  profiles/gates. All nine points now have agreeing phase families across
  seeds (seven stripe, two paired), with zero accepted endpoints. Totals:
  902 evaluations, 20 source jobs in 18 independent lineages, 27.350764
  actual node-hours. The JSON/source inventory retains both original V=0
  parent endpoints and their distinct controls. All five figures and CSVs
  were regenerated; the sixteen unaffected endpoint/history/profile data
  are unchanged. The September 16 progress report remains a separate snapshot.

- The [September 16 progress report](reports/two_basin_progress_20260916/README.md)
  analyzes 520 new evaluations: both square (1.4,0) continuations finish
  20 more (100/82 cumulative), and eight cubic starts finish 60 each at
  (1.0,-0.4/-0.2/0) and (1.2,-0.4). All are maximum_iterations, unaccepted.
  The square pairing lineage now has leg-pair RMS 4.42e-8 and a stripe texture;
  its remaining drift concerns stripe profiles/positions. All four available
  cubic points reach essentially unpaired stripes from both seeds, with
  stronger physical spin/charge order than on the square. The cubic
  (1.0,-0.2) stripe start fails only the charge span, 1.31% above its relative
  threshold; other endpoints have varying residual positional drift and
  extrapolation failures. No thresholds or acceptance flags changed.
  Square continuation cost is 0.552431 actual node-hours; cubic saved solver
  time totals 6.160414 node-hours excluding overhead, with no cubic allocation
  reconciliation yet. The synced jobs.tsv files confirm submissions for all
  four September 15 campaigns; no states/stdout are available for the remaining
  ten cubic, twelve fine-cut or four positive-V starts. Maintenance is
  user-reported; live scheduler state was not queried.

- Four chi=200 starts are now prepared at square (1.2,+0.2): the existing
  reciprocal 95%/5% reference mixtures plus two regular intertwined textures
  with charge/pair periods 8/16 and spin-envelope periods 16/32 rungs.
  A compact recipe extracts the legacy cubic-frustrated run's bulk amplitudes
  and relative pairing signs, while discarding irregular peak positions and
  restoring exact target density. The legacy source remains incomplete after
  60 records; it provides shape guidance, not an energy competitor. Pairing
  peaks align with holes and spin antiphase nodes. All fields are rebuilt
  using the square kernel and evolve freely. Max 60/minimum 40, ten stable
  records and the recent square gates; exact E_p, no new bare jobs, no solver
  source changes. The [seed report and handoff](reports/square_positive_v_seeds_20260915/README.md)
  provides previews and a four-job launcher (12-node-hour ceiling), with
  82 local assertions and six mock/launcher checks passed. No submission
  performed locally. September 16 synced submission records supersede the
  original preparation-only status; the four positive-V starts have no results yet.
- Six finer square coordinates are prepared with both 95%/5% reference
  mixtures: t0=1.4 at V=-0.05/-0.10/-0.15, and V=-0.4 at t0=1.25/1.30/1.35.
  Controls use chi=200, max 60/minimum 40 raw evaluations and ten stable
  records, with square field tolerances and the revised 1e-7 energy gates.
  User-requested linear signed E_p interpolation is explicit and recorded.
  The [bare-ladder cuts and submission report](reports/two_basin_fine_cuts_20260915/README.md)
  shows smooth sampled total energies and a broad binding maximum around
  t0=1.2–1.4. A degree-three shape check changes the inferred coupling by
  at most 0.62% along V and about 5% along t0; these are sensitivity checks,
  not uncertainty bounds. No bare measurements exist at the six new points.
  All twelve starts are locally validated, with a 36-node-hour reservation
  ceiling and no pair-binding jobs or automatic extensions. Cubic submission
  was user-reported at preparation; September 16 submission records now list
  all twelve fine-cut jobs, with no results yet. Square continuations are analyzed above.
- Following full-grid review, the user authorized the same two-basin cubic
  grid and a short square (1.4,0) extension. Prepared 18 cubic-unfrustrated
  chi=200 starts: reciprocal 95%/5% templates, max 60 raw evaluations,
  minimum 40 and ten stable records. Prepared both square V=0 lineages for
  up to 20 additional raw evaluations from their pinned full MPS sources;
  minimum ten fresh records. Energy-window and inner-DMRG tolerances are
  1e-7 t/site and 1e-7 t total. Relative field/slow-mode checks remain strict;
  cubic absolute field tolerances scale with its threefold density kernel.
  User-run launchers share existing accounting and cap reservations at
  54+4 node-hours, with no automatic extensions. No jobs submitted locally.
  See the [settings, qualification and two submission commands](reports/two_basin_next_campaigns_20260915/README.md).
- The full square two-basin grid is locally complete: 18 histories, 862 MF
  evaluations, and 26.798333 allocation node-hours from synced sacct
  reconciliations. The [preliminary phase diagram](reports/two_basin_grid_20260915/README.md)
  assigns stripes to all six t0=1.0/1.2 points and to t0=1.4,V=0 with an
  ongoing-conversion flag; t0=1.4,V=-0.4/-0.2 are paired from both seeds.
  No endpoint is formally accepted. At (1.2,-0.4), the pairing start has a
  long paired transient before collapsing late into stripes. At (1.4,-0.2),
  both profiles/energies agree closely while small spin order decays; a
  ten-record span still prevents acceptance. Keep basin assignment separate
  from converged energetic selection. All pointwise comparison fingerprints
  match; the original anchors and remainder retain different stopping
  controls. Original artifacts and their acceptance flags are unchanged.
- Manuscript artifact, September 15: the new
  [introduction/background and results draft](manuscript/README.md) collects
  the existing literature and analyses through September 13. The 16-page
  PDF has 34 references, including five additions on stripes and coherence.
  This is a synthesis artifact; it does not change numerical acceptance,
  campaign status, or the working phase qualifications below.
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
  anchors are now synced and analyzed: stripe reaches 80 evaluations with
  negligible pairing; the pairing lineage stops at the deadline after 62
  while spin grows and pairing falls to 15.75% of its first measured value.
  Both have strong, similar CDW/SDW textures, but neither is self-consistent.
  September 15 synced accounting gives 4.444722 actual allocation node-hours
  for V=0, replacing the earlier 4.411889 solver-time estimate. See the
  [V=0 analysis](reports/two_basin_v000_20260912/README.md).
  The remaining fourteen completed with a 40-evaluation cap, minimum 30,
  ten stable records, a 5e-7 channel noise floor plus full-window drift gate,
  and a 2e-8 t/site energy span. Global/inner-DMRG tolerances are unchanged.
  Saved-history gates first pass at 35/30 for the stripe/pairing anchors;
  older growing V=0 raw histories still fail. Missing per-iteration identity
  errors prevent retrospective acceptance certification. The new scope is
  complete in the synchronized evidence; all fourteen stopped at 40 and
  remain formally unaccepted. Their allocation cost is 18.590486 node-hours.
  The separate-checkout launcher shared existing accounting. See the
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
All four original anchors now have synchronized terminal histories:
80/80 evaluations at V=-0.4, and 80/62 at V=0. Their dated analyses are linked above.

These are the newest locally documented campaigns. Their live scheduler state
must not be inferred from the repository.

| Run ID | Local record | Question |
|---|---|---|
| `20260915_cubic_unfrustrated_two_basin_95_5_60` | 18 submissions; 8 terminal histories at 4 points analyzed September 16, all stripe-like and unaccepted | Await remaining coordinates, especially t0=1.4 |
| `20260915_square_t014_v000_two_basin_finish20` | 2 terminal continuations analyzed; 100/82 cumulative, pairing collapses, neither accepted; actual 0.552431 node-hours | Qualitative basin resolved; positional self-consistency still unfinished |
| `20260915_square_two_basin_fine_cuts_95_5_60` | 12 synced submission records, no result/stdout yet | Resolve the square boundary cuts |
| `20260915_square_t012_vp02_four_seeds_60` | 4 synced submission records, no result/stdout yet | Test positive-V intertwined order |
| `20260910_square_two_basin_95_5_40_remainder` | Published, 14 configs prepared; 75 Julia assertions and mock launcher test pass; live status not reviewed here | Retain unresolved 40-step outcomes; review before continuation |
| `20260908_square_two_basin_95_5_80_anchors` | All four synced/analyzed, none accepted; V=-0.4 paired plateaus (80/80); V=0 stripe retained / pairing collapsing (80/62) | Exact V=0 allocation cost pending; targeted continuation may be needed for self-consistency |
| `20260903_phase1_square_t014_v000_pairing_legacy_chi400_tight` | synced; 0/2 software-accepted; user accepts energetic comparison as adequate | Retain as the L=64 energy/basin reference; higher-chi checks remain future work |
| `20260903_phase1_square_grid_smooth_pairing_chi200_loose` | five terminal states synced; 4 accepted, 1 diverging; compiled September 8 | Review Fourier grid and divergence at `(1.0,-0.4)` |
| `20260903_phase1_cubic_unfrustrated_grid_smooth_pairing_chi200_loose` | user reports still running September 8; two local terminal files seen, not analyzed here | Finish eight-point cubic-unfrustrated fill |
| `20260903_phase1_square_t014_vm04_legacy_stripe_compare_chi200_loose` | user reports still running September 8; no local terminal states | Test inherited stripe stability and, conditionally, its energy against the paired control |
| L=96/128 square chi=200 length comparison | four local field seeds and review configs; no scheduler campaign | Review the enlarged seeds before submission preparation |

Campaign contracts and exact run commands are in the corresponding dated
documents linked from `docs/plans/ACTIVE.md`.

## Live Perlmutter and accounting boundary

- September 16 sync: square continuation jobs 58383124/58383125 are COMPLETED
  in the reconciliation ledger (3930/4025 elapsed seconds, quarter-node
  charges totaling 0.552431 node-hours). Eight cubic jobs have terminal
  maximum_iterations artifacts and complete logs but no synced allocation
  reconciliation. Their 6.160414 solver node-hours exclude overhead. The
  user reports monthly maintenance; no live scheduler status was checked.
  Older dated accounting statements below describe their original snapshots.

- Allocation evidence supplied on 2026-09-10: jobs 58093799 (stripe) and 58093800
  (pairing) at V=-0.4 are COMPLETED in pasted sacct output, with elapsed
  seconds 30799 and 23390. Their combined project charge is 3.763125 node-hours.
  Their local histories confirm maximum_iterations after 80 evaluations each.
  Jobs 58093802/58093803 were PENDING in that output; the September 12
  synced terminal artifacts supersede that status, as recorded below.
  No automatic remainder or continuation submission is configured. The local
  reconciliation ledger has no entries yet for these four jobs; analysis
  does not modify accounting, and 2.236875 node-hours of the finished pair's
  6-hour reserved ceiling are eligible for release through normal reconciliation.
- September 12: the user identifies `(1.4,0.0)` as completed. Synced states
  and logs confirm 58093802 ended `maximum_iterations` at 80 and 58093803
  ended `time_limit` at 62, both accepted=false and with matching original
  numerical/implementation fingerprints. Recorded MF times are 22189.829
  and 41341.378 seconds, or 4.411889 combined fractional node-hours excluding
  allocation overhead. The synced ledger has only their reservations; exact
  sacct accounting was requested. No live scheduler check was performed.
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

Await user-synchronized results from the remaining cubic coordinates, the
finer square cuts and positive-V comparison. All have synced submission IDs;
do not submit duplicates based on this local snapshot. Analyze full raw
histories and matched physical profiles, retaining incomplete status and
reserving energetic selection for accepted compatible solutions. The square
(1.4,0) extension has resolved the observed pairing collapse; further stripe
position relaxation is a separate decision. No extra runs or threshold
changes are prepared by the September 16 analysis. Reconcile cubic actual
allocation costs when the user supplies the updated ledger after maintenance.

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
