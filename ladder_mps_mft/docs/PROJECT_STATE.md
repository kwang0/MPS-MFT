# Current project state

Last locally reviewed: 2026-09-18 (cubic grid, finer square cuts, positive V and first trellis result)

This is the canonical mutable snapshot for resuming work. It is deliberately
short. Stable rules belong in `AGENTS.md` and the method documents; durable
history belongs in `docs/RUN_LOG.md`.

## Repository snapshot

- Branch: `codex/mps-mft-phase0-refactor`
- Analysis baseline: `558c2b2` (Expand MPS-MFT workflow and analysis support).
- Existing uncommitted trellis implementation/preparation changes are
  preserved alongside the September 18 analysis additions. An untracked
  root `.claude/` directory remains outside this subproject's analysis scope.
- Local `output/` and simulation HDF5 files are excluded from Git. The small
  `data/two_basin_references.h5` is a versioned seed-input exception.
- The finer-cut campaign added explicit V-axis E_p interpolation and changed
  the solver implementation fingerprint. The new positive-V comparison adds
  preparation scripts and four seeds without further solver changes. Preserve
  submitted source checkouts and use each campaign's separate-worktree handoff.

Recheck the branch, commit, and working tree at the start of each new task.

## Current scientific position

The [September 18 combined review](reports/campaign_review_20260918/README.md)
is the current results entry point. It covers 34 complete chi=200 histories,
2040 MF evaluations, plus partial square/trellis logs. All 34 completed
states reach maximum_iterations at 60 and remain unaccepted. The user
reports that one square run and some trellis runs are still unfinished;
local artifacts/logs are the evidence below, not live scheduler queries.

- **Cubic unfrustrated:** all 18 starts at all nine coordinates are synced.
  Both seeds reach essentially unpaired stripes everywhere, including
  (1.4,-0.4/-0.2), which are paired on square. Physical spin RMS is
  0.280–0.356; maximum leg-pair RMS is 2.21e-11. All final energy -window,
  density, inner-DMRG and identity checks pass; field/profile/slow-mode
  checks still prevent acceptance. See the [complete cubic report](reports/cubic_two_basin_grid_20260918/README.md).
- **Finer square cuts:** all twelve starts complete 60 evaluations. Both
  seeds are paired at (1.4,-0.15/-0.10) and (1.30/1.35,-0.4). Distinct
  stripe/paired trajectories remain at (1.25,-0.4) and (1.4,-0.05), with
  signed E(pair seed)-E(stripe seed) diagnostics -3.19e-5/-5.95e-5 t/site.
  These are not accepted energy winners or proof of first-order behavior.
  At (1.4,-0.05), 96.7% of the paired trajectory's spin-squared weight is
  in the outer 14 rungs at each end; central spin still decays despite a
  slight increase in the usual bulk RMS. Do not label this established
  bulk coexistence. All six E_p interpolations match the requested linear
  construction. See the [finer-cut report](reports/square_fine_cuts_20260918/README.md).
- **Square (1.2,+0.2):** three complete starts (stripe, pairing, intertwined
  period 16) lose pairing and reach nominal charge/spin-envelope periods
  16/32. Spin RMS is about 0.233; leg-pair RMS is below 8.3e-10. The
  period-eight intertwined job has 46 complete MF stdout records but no
  state/checkpoint; no phase is inferred for it. Positive V has not yet
  reproduced legacy cubic -frustrated intertwining on square. See the
  [positive-V report](reports/square_positive_v_20260918/README.md).
- **Trellis:** the one-ladder stripe start (job 58468871) completes 60 sweeps
  and becomes paired: physical spin RMS 0.0627 to 1.21e-5, leg-pair RMS
  0.00965 to 0.01797, opposite leg/rung signs. This differs from square and
  cubic at (1,0). Global residual checks pass late, but energy, inner-DMRG
  and channel/profile gates fail. Other logs contain 21 one-ladder pairing,
  18 two-ladder stripe and one two-ladder pairing cell sweeps, without
  spatial artifacts. See the [trellis progress report](reports/trellis_progress_20260918/README.md).
  The [method contract](TRELLIS_MEAN_FIELD.md) distinguishes skew one-ladder
  repetition from a rectangular A/B spatial cell. Physical profiles must
  use stored raw correlations, not square/cubic Hartree inversion.
- **Coarse square reference:** the [full square-grid report](reports/two_basin_grid_20260915/README.md)
  includes both (1.4,0) continuations, giving 100/82 cumulative evaluations
  there. Both seeds now reach the same family at all nine points: seven
  stripe, two paired, zero accepted endpoints. It covers 902 evaluations,
  20 source jobs in 18 independent lineages and 27.350764 actual node-hours.
  Its sixteen unaffected endpoints and all historical acceptance flags are
  unchanged. The [September 16 partial report](reports/two_basin_progress_20260916/README.md)
  is retained as a historical snapshot, superseded for cubic coverage/cost.

The approved comparison uses reciprocal 95%/5% correlation mixtures of the
legacy stripe at (1,0) and paired reference at (1.4,-0.4), rebuilt with target
couplings, chi=200, raw updates and no Anderson. Old loose accepted V=0
states showed SDW growth canceled by Anderson; their flags do not establish
stability. The [basin assessment](reports/square_grid_20260908/BASIN_ASSESSMENT.md)
and raw anchor reports retain that evidence. MF iterations are not physical
time, and flat energy is not a certificate of a stationary spatial profile.

The user's September 6 working interpretation of the chi=400 square (1.4,0)
comparison remains recorded: stripe's endpoint advantage 2.43361262e-4 t/site
was judged adequate evidence of robustness to the tested chi increase,
although both archived endpoints remain formally unaccepted. See the
[chi=400 analysis](reports/chi400_comparison_20260905/ANALYSIS.md).
Four [L=96/L=128 seeds](reports/finite_size_seeds_20260906/README.md) remain
prepared with fixed L=64 E_p. Higher chi, length and wavelength studies are
deferred; this review authorizes no new runs or threshold changes.

The [manuscript draft](manuscript/README.md) still has a September 13 numerical
evidence cutoff. Its source notes now point to the new results; the LaTeX/PDF
has not silently been rewritten. The [literature source notes](literature/SOURCE_NOTES.md)
retain the equation-level identification of Bollmark2023 with this project's
cubic_frustrated kernel and now qualify the positive-V interpretation using
the new square evidence. No new literature search is claimed.

## Campaign inventory

| Run ID | Current local evidence | Remaining question |
|---|---|---|
| `20260915_cubic_unfrustrated_two_basin_95_5_60` | 18/18 terminal, 60 each, all stripe-like/unaccepted | Qualify spatial convergence if needed; no paired basin observed |
| `20260915_square_t014_v000_two_basin_finish20` | Both continuations analyzed, 100/82 cumulative, both stripe-like | Remaining stripe-position stationarity |
| `20260915_square_two_basin_fine_cuts_95_5_60` | 12/12 terminal, 60 each, two seed-dependent points | Branch stability and transition order |
| `20260915_square_t012_vp02_four_seeds_60` | 3/4 terminal, all stripe; period-eight stdout through 46 | Missing wavelength test's spatial outcome |
| `20260916_trellis_two_basin_comparison_60` | 1/4 terminal, one-ladder stripe seed becomes paired; other logs partial | Other seed and rectangular A/B cell outcomes |
| `20260910_square_two_basin_95_5_40_remainder` | All 14 terminal, analyzed/accounted in square grid | Retain unaccepted status |
| `20260908_square_two_basin_95_5_80_anchors` | All 4 terminal; V=0 continuations supersede parent endpoints | Parent provenance/controls retained |
| L=96/L=128 square chi=200 | Four prepared field seeds, no new scheduler campaign | Deferred finite-size study |

Older loose coverage campaigns, legacy stripes, chi=400 comparison and
Anderson-era acceptance retain their dated reports and run-log entries.
The [September 8 square bundle](reports/square_grid_20260908/README.md)
contains all nine selected small-seed/approved-legacy states with complete
histories, Fourier plots and the flagged divergence. It is historical
coverage, not the newer raw two-family phase comparison. Do not infer the
live state of unreviewed older campaigns from their dated records.

## Live Perlmutter and accounting boundary

- The September 18 review covers 31.931350 solver-only fractional node-hours:
  cubic 13.886995, finer cuts 13.030485, three positive-V starts3.027065,
  completed trellis 1.986804. One GPU is one quarter node. These exclude
  allocation overhead and all partial jobs; no exact combined allocation
  cost can be reported from this sync.
- Reconciled cubic actual cost is 9.441597 node-hours for 12/18 jobs
  (t0=1.0/1.2 only). The six t0=1.4 jobs have 4.642934 solver-only node-hours.
  No synced actual reconciliation exists for finer-cut, positive-V or trellis
  jobs. Analysis did not write either budget ledger.
- Coarse square actual cost, including continuations, remains 27.350764
  node-hours. The two continuation jobs 58383124/58383125 contribute 0.552431.
  The four original anchors and fourteen remainder jobs are accounted in
  that square total; do not add their older estimates again.
- Local terminal artifacts establish solver outcomes, not live Slurm state.
  Only the user performs transfers, submissions, scheduler queries and
  reconciliation. The Perlmutter checkout remains
  `$CFS/m4863/MPS-MFT/ladder_mps_mft`; submitted source directories are preserved.
- The 400-additional-node-hour project control remains. Requested ceilings
  are not actual usage or blanket continuation authority. The live ledgers
  and current user-provided accounting are authoritative for new compute.

## Exact next action

Review the [combined report](reports/campaign_review_20260918/README.md).
Await user-synchronized spatial artifacts for square period-eight job 58394105
and the other trellis jobs 58468873/58468875/58468876. Do not submit duplicates
or infer missing phases from their scalar stdout energies. Once available,
compare full histories, physical profiles and same-Hamiltonian accepted
canonical solutions, preserving the different trellis cell interpretations.

The split square points (1.25,-0.4) and (1.4,-0.05) are candidates for a later
selective stability/convergence decision, not prepared follow-up jobs. Resolve
end-dominated versus bulk spin before claiming coexistence or a first-order
transition. Reconcile actual allocation costs before additional compute.
Higher chi, length and precise E_p interpolation work remains deferred.

## Verification boundaries

- Local compact verification does not prove the recorded full scratch artifact
  is present or hash-valid.
- Scheduler and accounting claims require current user-provided Perlmutter
  output or a synchronized ledger snapshot.
- Accepted finite-`L`, finite-`chi` Phase 1 endpoints are variational states,
  not thermodynamic phase assignments.
- Never compare canonical energies across transverse geometries or different
  Hamiltonian points.
