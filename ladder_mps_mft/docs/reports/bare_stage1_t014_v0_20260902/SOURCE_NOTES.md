# Bare Stage 1 report source notes

## Reporting question

What did the completed `V=0`, `t0=1.4` bare-ladder backbone and Stage 1
covariance screen establish, how efficiently did the implementation run, and
what is the smallest defensible Stage 2 finite-field pilot?

The intended audience is the technical research team. The evidence cutoff is
2026-09-02, after the user synchronized run
`20260901_bare_t014_v0_stage1` locally and before any Stage 2 submission.

## Local evidence used

- `output/bare_stage1/20260901_bare_t014_v0_stage1/stateless_results/backbone.h5`
  supplies model parameters, sector energies, convergence flags, gap and
  binding estimates versus bond dimension, validity ratios, and provenance.
- `output/bare_stage1/20260901_bare_t014_v0_stage1/stateless_results/stage1.h5`
  supplies diagnostics, covariance matrices and modes, decay fits, and the
  zero-field map norm.
- `output/bare_stage1/20260901_bare_t014_v0_stage1/stateless_results/stateless_manifest.tsv`
  supplies the 33 full/compact sizes and hashes used for the storage analysis.
- The six `logs/sector-*.out` files supply per-sweep elapsed times, discarded
  weights, sweep counts, and the sector-array critical path.
- `configs/bare_stage2_t014_v0.toml`, `src/BareStage2.jl`, and
  `slurm/bare_stage2_cpu.sh` define the recommended Stage 2 protocol and its
  conservative reservation bounds.

The builder opens the synchronized sources read-only and writes only within
this report directory. `artifact.json` records SHA-256 hashes for the four
compact source artifacts. The CSV files in `data/` are deterministic extracts.

## Definitions

- `DMRG hours` is the sum of ITensor's printed per-sweep `time=` values. It is
  neither scheduler elapsed time nor allocation charge.
- `sector-array critical path` is the largest such sum over the six independent
  sector logs. It is an idealized lower bound on the sector phase when all
  array tasks can run concurrently.
- `participation rank` is `(sum lambda)^2 / sum(lambda^2)` after clipping
  covariance eigenvalues at zero within floating-point precision.
- `k90` is the number of descending nonnegative covariance eigenvalues needed
  to accumulate 90 percent of their sum.
- Pair-binding signs follow the repository convention: negative means bound.
- The two `K_rho` values are both reported because the saved rung and site
  normalizations differ by a factor of two.

## Interpretation exclusions

- Equal-time covariance eigenvalues are not called susceptibilities and are
  not compared across operator classes with different normalizations.
- The finite `L=64`, `chi=1200` gap and decay results are not promoted to a
  thermodynamic phase claim.
- The central-charge point estimate is not interpreted because its fit has
  `R^2=0.155`.
- Logged DMRG times are not used to claim CPU efficiency, peak memory, or
  charged node-hours because no Stage 1 `sacct` or `/usr/bin/time -v` output
  was synchronized.
- The Stage 2 node-hour values are conservative reservation ceilings computed
  from requested walltimes and memory, not measured costs.
- The weak-coupling MPS+MF approximation is not described as controlled at
  this point because `tp/Delta_c` is about 9.21 and the other two ratios are
  about 0.65--0.68.

## Stage 2 decision logic

The response bank preserves 14 meaningful names for provenance but contains
12 independent vectors after metric orthogonalization: nine number-conserving
normal fields and three parity-conserving pairing fields. `extended_s` and
`d_wave` are exact combinations of the q=0 onsite/rung/leg field span, so
removing their duplicate columns does not prevent Stage 2 eigenvectors from
choosing those mixtures.

Discovery uses one amplitude per independent direction. It is allowed to
produce a validation plan only if all solves converge, within-block response
reciprocity is within 5 percent, and the normal/pair cross block is within 5
percent. The optional second submission checks three selected eigenvectors at
`h` and `h/2`, requires 5 percent linearity, and forms the Richardson estimate.

No random residual Stage 3 probe is included yet. That decision should use the
Stage 2 response spectrum and leakage rather than the broad Stage 1 covariance
rank alone.
