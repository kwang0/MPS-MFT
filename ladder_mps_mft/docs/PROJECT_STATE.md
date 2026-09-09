# Current project state

Last locally reviewed: 2026-09-08 (two-reference campaign revised to 95%/5%, 80 evaluations)

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
- Local `output/` and all HDF5 files are intentionally excluded from Git.
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
  use chi=200 and **no Anderson**: up to 80 raw evaluations, at least 50 before
  acceptance, and tighter channel/energy/inner-DMRG gates. Four anchors precede
  the remaining fourteen. See the
  [prepared campaign and handoff](reports/two_basin_raw_20260908/README.md).
  No jobs are submitted. Higher chi, length, and stripe-wavelength work is
  deferred by the user's latest instruction.
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
  pairing for those perturbations. The pending strong-stripe comparison is
  still needed to test a distinct competing solution and its energy. See the
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
Its 18 seeds/configs are local preparation, without new simulation results.

These are the newest locally documented campaigns. Their live scheduler state
must not be inferred from the repository.

| Run ID | Local record | Question |
|---|---|---|
| `20260908_square_two_basin_raw_eps005_iter80_anchors` (planned ID) | 18 square starts locally previewed; four-anchor handoff ready; no jobs/reservations | Qualify the raw chi=200 comparison before the remaining fourteen starts |
| `20260903_phase1_square_t014_v000_pairing_legacy_chi400_tight` | synced; 0/2 software-accepted; user accepts energetic comparison as adequate | Retain as the L=64 energy/basin reference; higher-chi checks remain future work |
| `20260903_phase1_square_grid_smooth_pairing_chi200_loose` | five terminal states synced; 4 accepted, 1 diverging; compiled September 8 | Review Fourier grid and divergence at `(1.0,-0.4)` |
| `20260903_phase1_cubic_unfrustrated_grid_smooth_pairing_chi200_loose` | user reports still running September 8; two local terminal files seen, not analyzed here | Finish eight-point cubic-unfrustrated fill |
| `20260903_phase1_square_t014_vm04_legacy_stripe_compare_chi200_loose` | user reports still running September 8; no local terminal states | Test inherited stripe stability and, conditionally, its energy against the paired control |
| L=96/128 square chi=200 length comparison | four local field seeds and review configs; no scheduler campaign | Review the enlarged seeds before submission preparation |

Campaign contracts and exact run commands are in the corresponding dated
documents linked from `docs/plans/ACTIVE.md`.

## Live Perlmutter and accounting boundary

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

Review the concrete seeds and numerical contract in
`docs/reports/two_basin_raw_20260908/README.md`. The source bundle keeps this
implementation separate from pending campaign code. The user transfers it,
prepares four anchors at `(1.4,0)` and `(1.4,-0.4)`, and uses the existing
guarded launcher and shared live ledgers. First-segment ceilings are 12
fractional node-hours for those four and 42 for the later fourteen, subject
to the live 400-hour cap. Do not infer current remaining budget locally.

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
