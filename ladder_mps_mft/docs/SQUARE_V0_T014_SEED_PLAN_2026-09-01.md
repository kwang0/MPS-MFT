# Square V=0, t0=1.4 six-seed Phase 1 plan

Status: implemented and tested locally; not prepared on Perlmutter, submitted,
or charged to the project ledger.

## Scientific question

At the square-geometry point `L=64`, `U=8`, `V=0`, `t0=1.4`,
`t_perp=0.1`, and density `0.9375`, do six deliberately different but
matched-scale initial conditions enter one common self-consistent basin or
distinct accepted basins? In particular, the bank tests whether the normal
stripe and d-wave-pairing sectors exclude one another, coexist, or lose their
initial-seed memory. It is a finite-ladder basin/convergence test, not a
thermodynamic phase determination.

The exact pair-binding registry row is `E_p=-0.14653773091916378`; interpolation
is disabled. All branches use Float64 CUDA tensors, `chi=200`, 12 DMRG sweeps,
cutoff `1e-10`, DMRG energy tolerance `1e-6`, target-density tolerances `1e-3`,
and field gates `1e-6` absolute or `5e-3` relative. These loose settings are
exploratory. Accepted outputs may be ranked only against accepted siblings with
the same model, numerical, solver-implementation, and E_p fingerprints.

## Predeclared seed bank

| branch | source |
|---|---|
| pairing control | uniform d-wave, envelope mode 0 |
| legacy-like pairing control | random relative-bond alpha copied along the ladder; beta and Hartree fields zero |
| stripe control m=4 | AF spin mode 59 plus charge harmonic 8 |
| stripe control m=5 | AF spin mode 58 plus charge harmonic 10 |
| stripe+d-wave m=4 | the m=4 stripe and uniform d-wave together |
| stripe+d-wave m=5 | the m=5 stripe and uniform d-wave together |

Every branch uses total matched field norm `1e-3`, phase zero, and product-state
random seed `1404`. No branch inherits or resumes an MPS or field state.

## Numerical gates in this campaign

- The first 20 MF updates are the raw map. Anderson acceleration is available
  only afterward and cannot certify a physical orbit.
- A period-two classification additionally requires recent step vectors to
  reverse (`cosine <= -0.5`) and the two-step/one-step norm ratio to be at most
  `0.5`. Slow monotone drift can no longer masquerade as a two-cycle.
- When successive fixed-point residuals align with cosine at least `0.9`,
  acceptance uses the extrapolated residual `r/(1-lambda)`. `lambda >= 1`
  fails the gate.
- The density search carries a positive compressibility estimate between SCF
  updates and uses `1e-8,1e-9,0` noise for warm-started chemical-potential
  re-solves. Per-sweep energy, discarded weight, and maximum link dimension are
  stored for later convergence analysis.
- Energy stabilization and branch ranking use
  `E + mu*(N_target-N)`. The uncorrected canonical energy and complete
  double-counting decomposition remain stored as provenance.

## Offline pre-fix re-audit

Before changing the classifier, the read-only audit inspected all 45 local
Phase 1 `state.h5` paths. Thirty-six had complete applied/measured histories;
nine v2 paths could not be reclassified because their compact artifacts lack
the applied-field history. Fourteen auditable paths change classification.

For v3 and Stage A specifically, five stored v3 fixed points fail the
slow-mode-extrapolated residual gate. The v3 unfrustrated-pairing period-two
candidate and both Stage A phase-parent candidates have step cosine about
`+0.9999` and two-step/one-step ratio about `2`, so they are monotone drift,
not oscillatory period two. The audit is external evidence only; no immutable
HDF5 status or acceptance field was edited. See
`analysis/numerics_reaudit_pre_fix_20260901/`.

## Ledger and staged Perlmutter commands

Each 12-hour shared-GPU branch reserves `3.000` GPU node-hours, so launcher
v1.13.1's direct first-segment envelope is `18.000` node-hours. Perlmutter
measurement and the live ledger remain authoritative. The four-segment
emergency ceiling is `72.000` and is not pre-authorized. For this already-
prepared v1.12.0 campaign, any smoke reservation recorded before the launcher
update remains in the append-only ledger until the terminal job is reconciled;
it is not repeated by direct submission.

Run on Perlmutter after syncing the code:

```bash
cd "$CFS/m4863/MPS-MFT/ladder_mps_mft"
bash slurm/phase1_gpu.sh plan-square-v0-seed-pilot

SQUARE_V0_RUN=20260901_phase1_square_t014_v000_seed_chi200_loose
bash slurm/phase1_gpu.sh prepare-square-v0-seed-pilot "$SQUARE_V0_RUN"
bash slurm/phase1_gpu.sh submit "$SQUARE_V0_RUN"
bash slurm/phase1_gpu.sh status "$SQUARE_V0_RUN"
```

`submit` sends all still-pending scientific branches directly. Each branch
performs the Float64 artifact-runtime and linear-algebra preflight before its
SCF work. `submit-matrix` remains only as a backward-compatible alias for the
same direct action. Preparation alone does not submit a job or reserve ledger
capacity.
