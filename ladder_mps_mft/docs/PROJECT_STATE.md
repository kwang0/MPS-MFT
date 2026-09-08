# Current project state

Last locally reviewed: 2026-09-06 (finite-size seeds prepared for review)

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

- Phase 0 is closed. Phase 1 uses the refactored Float64 dense-CUDA path.
- At square `(t0,V)=(1.4,0)`, the completed loose `chi=200` six-seed campaign
  produced six accepted period-one endpoints that are qualitatively close and
  pairing-bearing. Small remaining differences are not a thermodynamic phase
  claim and remain subject to tighter tolerance, bond-dimension, length, and
  error-budget checks.
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
- The current surveys fill the square and cubic-unfrustrated `3 x 3`
  `(t0,V)` grids using one center-of-mass-uniform smooth-pairing access seed.
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
square V=0 states all survive and match their compact manifests. Their corrected
energy spread is only `6.0274e-7 t/site`, with fine ordering unresolved. Priority
code recommendations concern implementation hashing, invalid selection inputs,
and propagation of inner-DMRG convergence. The finite-size even-particle charge
gap is not the denominator of the pair-breaking projection; the report corrects
that interpretation while retaining weak-hopping and convergence controls.

## Campaign inventory

These are the newest locally documented campaigns. Their live scheduler state
must not be inferred from the repository.

| Run ID | Local record | Question |
|---|---|---|
| `20260903_phase1_square_t014_v000_pairing_legacy_chi400_tight` | synced; 0/2 software-accepted; user accepts energetic comparison as adequate | Retain as the L=64 energy/basin reference; higher-chi checks remain future work |
| `20260903_phase1_square_grid_smooth_pairing_chi200_loose` | submission records synced; user reports pending; no local terminal states | Fill five missing square-grid points |
| `20260903_phase1_cubic_unfrustrated_grid_smooth_pairing_chi200_loose` | submission records synced; user reports pending; no local terminal states | Fill eight missing cubic-unfrustrated points |
| `20260903_phase1_square_t014_vm04_legacy_stripe_compare_chi200_loose` | submission records synced; user reports pending; no local terminal states | Test inherited stripe stability and, conditionally, its energy against the paired control |
| L=96/128 square chi=200 length comparison | four local field seeds and review configs; no scheduler campaign | Review the enlarged seeds before submission preparation |

Campaign contracts and exact run commands are in the corresponding dated
documents linked from `docs/plans/ACTIVE.md`.

## Live Perlmutter and accounting boundary

- User-reported on 2026-09-05: the square grid, cubic-unfrustrated grid, and
  `(1.4,-0.4)` stripe comparison remain pending. Their synced submission files
  record 5, 8, and 2 branch jobs respectively; live scheduler state is not
  independently verified. IDs are recorded in the September 5 analysis.
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

Review the four finite-size seeds. Their configs keep tight comparison controls
at chi=200, relax the identity gate to 1e-8 t/site, and hold E_p at the measured
L=64 value using explicit `pair_binding.reference_L=64`. The local preparation
passed four Python tests and 99 Julia config/seed assertions, without DMRG.
After seed review, assign scratch paths and prepare the four-run launcher
handoff in a source checkout that does not disturb the three pending campaigns.
No new jobs or reservations exist; reconcile actual accounting before compute.

Wait for the three pending campaigns. Do not submit duplicates or cancel them
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
