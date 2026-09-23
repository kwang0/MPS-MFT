# Four two-ladder trellis intertwined starts

Prepared September 23, 2026 at the user's request. Exactly one intertwined
seed at each of four coordinates; all use the rectangular two-ladder cell.
The user reports that the square t_perp scan and earlier trellis runs are
ongoing. Their source checkouts must remain fixed. No Perlmutter connection,
transfer, submission, scheduler query or accounting mutation was performed locally.

| t0 | V | tau0 | tau1 | Exact signed E_p |
|---:|---:|---:|---:|---:|
| 1.2 | 0.0 | 0.1 | 0.1 | -0.17989619749147323 |
| 1.2 | +0.2 | 0.1 | 0.1 | -0.15307266912955697 |
| 1.4 | 0.0 | 0.1 | 0.1 | -0.14653773091916378 |
| 1.4 | +0.2 | 0.1 | 0.1 | -0.11678278200975001 |

All pair-binding values are exact-coordinate chi=1000 registry entries.
There is no interpolation or new pair-binding computation. Fix t=1, U=8,
L=64 rungs per ladder, n=15/16 on each ladder, chi=200 and r_range=4.

## One seed family, four independent starts

Reuse `intertwined_lambda16` from the existing
[positive-V seed construction](../square_positive_v_seeds_20260915/README.md).
Its charge and pairing-amplitude period is 16 rungs; the staggered-spin
envelope has period 32. This fits the observed period-16 charge stripes and
provides four charge cycles at L64. Pairing is strongest at hole-rich spin
walls, has opposite rung/leg signs, and keeps the same sign between walls.
It tests legacy-like intertwined order, without imposing an antiphase PDW.
No period-eight, uniform-only or stripe-only branch is added.

The frozen `data/positive_v_intertwined_recipe.toml` supplies the same
amplitudes and relative-bond coefficients for all four targets. Its SHA-256 is
`786fa2e8846820f42aabbabb625c3a645558d25e5388afa4058abf63cf834d35`.
The original legacy source was incomplete; it supplies shape information
only, not a converged state or an energy reference. No legacy MPS is inherited.

A and B start from identical templates in their local rung coordinates,
following the existing trellis initialization convention. B retains the
physical -1/2-rung offset. Thus identical local arrays are not a claim of
identical modulation phase at a common physical x. Each ladder's fields
are independently rebuilt with the target rectangular trellis map and E_p.
A/B have the same global pairing sign initially, separate fresh MPSs with
RNG seed 1404, and subsequently independent profiles. No spatial pinning
or enforced equality of their fields is introduced.

Each configuration, seed hash, target model fingerprint, numerical fingerprint
and registry/implementation hash is recorded in the generated manifest.
The [local preview receipt](prepared_branches.csv) records the Windows preview;
Perlmutter preparation regenerates its own paths and hashes.

## Controls and resources

Keep the previous trellis controls: 60 maximum simultaneous raw cell sweeps,
40 minimum, ten stable records, no damping or Anderson, and stationary
period-one acceptance only. The four runs allow at most 240 cell sweeps /
480 individual ladder solves. Existing field, energy, density and inner-DMRG
thresholds are preserved. Full terminal correlations are enabled on both
ladders, including finite maximum-iteration endpoints without accepting them.

The wrapper requests four one-GPU jobs, each with 32 logical CPU cores and
16 hours in shared QOS, one segment per job. The solver deadline remains
11.5 hours, with 4.5 hours available for terminal diagnostics. The reservation
ceiling is **16 fractional node-hours** (4 x 16 x 0.25), not a measured cost.
The existing shared 400-additional-node-hour control and locking remain in
force; the user-run launcher reconciles and checks available budget before
submission. No new budget ledger or independent spending allowance is created.

## Isolation and Perlmutter handoff — user-run only

Use a third checkout, separate from both ongoing campaigns. Do not pull into
either running checkout. Fetch updates Git objects/references without changing
their working files or HEAD; the new detached worktree holds the new launcher.

```bash
cd "$CFS/m4863/MPS-MFT/ladder_mps_mft"
git -C .. fetch origin
git -C .. worktree add --detach "$CFS/m4863/MPS-MFT-trellis-intertwined-20260923" origin/codex/mps-mft-phase0-refactor
cd "$CFS/m4863/MPS-MFT-trellis-intertwined-20260923/ladder_mps_mft"
bash slurm/submit_trellis_intertwined.sh
```

The wrapper refuses the original trellis checkout and the standard square
scan checkout `MPS-MFT-square-tp-20260922/ladder_mps_mft`. If the square scan
uses another path, set `TWO_BASIN_SQUARE_TP_PROJECT` to that project directory
so the same guard covers it. Keep the new checkout fixed while its jobs run.

The wrapper reads the original anchor run.env to retain the existing account,
scratch root, budget ledger and reconciliation ledger. It then uses the
isolated control root `PHASE1_RUN_ROOT/trellis_intertwined` and unique run ID
`20260923_trellis_two_ladder_intertwined_lambda16_60`. The original and square
scan latest-run pointers remain untouched. Full scratch results are separated
by the unique run ID; existing directories cannot be overwritten. Solver
sources and previous configs, preparers and submission wrappers are unchanged.

To inspect this campaign after submission, from its new checkout:

```bash
bash slurm/phase1_gpu.sh status "$CFS/m4863/MPS-MFT/ladder_mps_mft/output/phase1_gpu/trellis_intertwined/20260923_trellis_two_ladder_intertwined_lambda16_60"
```

The displayed path assumes the standard inherited root; preparation prints
the authoritative path if the original campaign used an override.

## Local validation

`test/test_trellis_intertwined.jl` passed **157 assertions** (32.4 seconds of
test execution, excluding Julia package startup). The nine focused Python
launcher tests passed in 9.3 seconds. `git diff --check` passed.

Focused checks cover exact four-point coverage and E_p values, period-16
charge/pair shapes and period-32 spin envelope, pairing at hole-rich walls,
nonzero pairing/spin fields on both ladders, target-map reconstruction,
hash/readback consistency, production seed loading into fresh CPU product
MPSs, and rejection of wrong cells, damping and overwrite attempts.
Launcher checks use local fake commands and extracted validation functions,
never Slurm. They verify both checkout guards, shared accounting, isolated
control root, one segment, exact branch count, seed-family/E_p restrictions,
preparation-failure stop and compatibility with existing campaign versions.
No DMRG solve, GPU benchmark or full test suite is needed for this preparation.
