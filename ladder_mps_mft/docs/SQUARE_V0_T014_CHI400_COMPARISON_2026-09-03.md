# Square V=0, t0=1.4 chi=400 two-lineage comparison

## Question

This campaign asks whether the pairing-dominated and legacy-like high-amplitude
basins remain distinct when both are evolved with the same tighter numerical
controls at twice the previous bond dimension. It is a bond-dimension and basin
comparison at one finite system size, not evidence for a thermodynamic phase.

The two `chi=200` parents are:

1. `square__pairing_dwave_m000_chi200_loose`, the clean pure-d-wave seed branch
   with the smallest final residual and no deliberately retained stripe source;
2. `square__legacy_terminal_fields_frozen_dmrg_chi200`, the current Float64 DMRG
   evaluated at the terminal legacy fields.

Both new branches load the exact full parent MPS from scratch, verify its
SHA-256, and start a fresh `chi=400` SCF history. The legacy-like branch uses
the frozen calculation's **measured restart fields**, not its applied legacy
field tensor. This preserves the physical outgoing map while removing the
unused onsite-beta entries that dominated the diagnostic's stored all-entry
residual. Preparation refuses the lineage unless those inactive restart entries
are zero to `1e-12`.

## Numerical contract

- Model: square, `L=64`, `U=8`, `V=0`, `t0=1.4`, `t_perp=0.1`, density
  `0.9375`.
- Pair binding: exact registry value `E_p=-0.14653773091916378`.
- DMRG: `chi=400`, 16 sweeps, cutoff `1e-11`, energy tolerance `1e-9`.
- Density search and outer acceptance: `1e-4`.
- Field acceptance: absolute residual `1e-7` or relative residual `1e-4`.
- Target-density-corrected variational-energy stability: `1e-7` per physical
  site, for two stable records.
- After the initial raw evaluation, up to 20 explicitly labeled unmixed updates
  form the raw-map probe. A physical recurrence is stored phase by phase.
  Anderson acceleration is used only after a recurrence-free or
  archived-unaccepted raw probe.
- Up to 80 MF updates may fit in the 11.5-hour solver deadline. A scheduler
  request is 12 hours so compaction has headroom.
- Full MPS artifacts remain on scratch; the CFS mirror is stateless.

The two new branches share one model, numerical, implementation, pair-binding,
geometry, and scalar-type fingerprint. They may be ranked against each other
only if accepted. Comparisons with the `chi=200` parents diagnose bond-dimension
dependence; they are not a same-numerical-fingerprint variational ranking.

## Perlmutter handoff

These commands are for the user-managed Perlmutter checkout. Preparation does
not submit anything or modify the budget ledger.

```bash
cd "$CFS/m4863/MPS-MFT/ladder_mps_mft"

PAIRING_RUN=20260902_phase1_square_t014_v000_seed_chi200_loose_cuda130
FROZEN_RUN=20260903_phase1_square_t014_v000_legacy_frozen_dmrg_chi200
COMPARE_RUN=20260903_phase1_square_t014_v000_pairing_legacy_chi400_tight

bash slurm/phase1_gpu.sh plan-square-v0-chi400-compare
bash slurm/phase1_gpu.sh prepare-square-v0-chi400-compare \
  "$PAIRING_RUN" "$FROZEN_RUN" "$COMPARE_RUN"

column -ts $'\t' "output/phase1_gpu/$COMPARE_RUN/manifest.tsv" | less -S
bash slurm/phase1_gpu.sh budget
bash slurm/phase1_gpu.sh submit "$COMPARE_RUN"
bash slurm/phase1_gpu.sh status "$COMPARE_RUN"
```

There is no smoke job. Submission creates two independent scientific jobs,
each requesting one of four GPUs for 12 hours. The conservative first-segment
reservation is therefore `2 * 12 / 4 = 6.000` node-hours. The last synchronized
ledger contains `42.024097222` active node-hours after reconciliation, so this
would project to `48.024097222` active and `351.975902778` unreserved. The
authoritative Perlmutter `budget` output must be checked immediately before
submission.

Four segments for both branches would have a `24.000` node-hour ceiling, but
that is not pre-authorized and is not reserved by the first submission. If a
branch reaches a time limit, inspect its compact history before submitting an
individual continuation. Reconcile terminal jobs against `sacct` before making
the next compute decision.

## Completion gate

The scientific comparison should report, for each lineage:

- accepted classification and whether it is a fixed point or a preserved raw
  periodic orbit;
- final density error, field residual, slow-mode extrapolated residual, and
  target-corrected energy stability;
- realized maximum link dimension and discarded weight;
- canonical energy with double-counting terms and its target-density-corrected
  counterpart;
- changes from `chi=200` to `chi=400` in pairing, charge, spin, and spatial
  profiles.

If either branch is unaccepted, its energy is diagnostic and must not enter the
authorized same-fingerprint ranking. No cross-geometry comparison is allowed.
