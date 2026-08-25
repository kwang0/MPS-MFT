# Phase 1 GPU v3 Float64-history audit

## Decision

Run `20260824_phase1_gpu_v3_float64_history` repairs the two numerical defects
that made v2 screening-only: the production tensor scalar type is Float64 and
schema-v5 states retain the complete applied and measured mean fields at every
MF iteration. A local audit of the stateless mirror on 2026-08-25 found eight
accepted fixed points and one unresolved raw-map period-two candidate.

This run is a controlled representative-point comparison, not a phase diagram.
It uses

```text
L=64, t=1, U=8, V=-0.2, t0=1.1, t_perp=0.1,
density=0.9375, chi=200, r_range=4
```

for `cubic_frustrated`, `cubic_unfrustrated`, and `square`, with pairing, SDW,
and CDW seed lineages inherited from the independently seeded v2 parents. The
pair binding is the bracketed linear interpolation

```text
E_p(t0=1.0) = -0.1545120066237189
E_p(t0=1.2) = -0.21453418655934797
E_p(t0=1.1) = -0.18452309659153346
t_perp^2 / |E_p| = 0.05419375777188662
```

with no extrapolation.

## Reproducible local audit

From `ladder_mps_mft/`, run:

```bash
RUN=output/phase1_gpu/20260824_phase1_gpu_v3_float64_history

julia --project=. scripts/verify_stateless_results.jl \
  "$RUN/stateless_results"

julia --project=. scripts/audit_phase1_campaign.jl \
  "$RUN" "$RUN/audit-NEW"
```

Choose a new audit output directory because the audit refuses to overwrite an
existing one. The 2026-08-25 verification covered 50 compact artifacts. Their
manifest represents 7,385,759,171 full bytes and 666,136,203 compact bytes;
the local compact tree occupies about 635 MiB. It verified compact hashes,
sizes, stateless markers, full-artifact hash links, and absence of MPS objects.
It did **not** verify the full scratch files because those paths are not mounted
on the Mac. On Perlmutter, add `--full` to verify both sides.

The audit generated at `2026-08-25T20:04:31.004` UTC reported:

- accepted states: `8 / 9`;
- raw-map periodic candidates: `1`;
- mixer-dependent periodic candidates: `0`;
- Hamiltonian-identity gate failures: `0 / 9`;
- effective-energy gate failures: `0 / 9`;
- recorded tensor scalar type: `float64`; and
- stored MF-iteration wall time: `17.518` GPU-hours, or `4.379`
  one-of-four-GPU node-hours before scheduler, compilation, and other overhead.

## State-level result

| Branch | Geometry | Seed | Class | Iterations | Solution canonical energy | Relative residual | dE/site | max abs pairing field |
|---|---|---|---|---:|---:|---:|---:|---:|
| `frustrated__pairing_s1` | `cubic_frustrated` | pairing | accepted fixed point | 33 | -101.446541298263 | 1.429e-5 | 6.535e-8 | 1.098e-2 |
| `frustrated__sdw_s1` | `cubic_frustrated` | SDW | accepted fixed point | 32 | -101.447620447185 | 7.285e-6 | 4.869e-8 | 1.098e-2 |
| `frustrated__cdw_s1` | `cubic_frustrated` | CDW | accepted fixed point | 34 | -101.454988710643 | 2.824e-5 | 8.422e-9 | 1.099e-2 |
| `unfrustrated__pairing_s1` | `cubic_unfrustrated` | pairing | raw-map period-2 candidate | 49 | -103.854169068279 | 6.141e-4 | 2.087e-5 | 1.925e-2 |
| `unfrustrated__sdw_s1` | `cubic_unfrustrated` | SDW | accepted fixed point | 4 | -104.414717975962 | 8.233e-4 | 1.279e-8 | 3.540e-9 |
| `unfrustrated__cdw_s1` | `cubic_unfrustrated` | CDW | accepted fixed point | 4 | -104.415030580718 | 1.087e-3 | 3.929e-8 | 2.318e-8 |
| `square__pairing_s1` | `square` | pairing | accepted fixed point | 6 | -103.405021512408 | 5.074e-4 | 3.937e-9 | 2.152e-8 |
| `square__sdw_s1` | `square` | SDW | accepted fixed point | 22 | -103.403976196299 | 3.610e-3 | 8.829e-8 | 1.080e-10 |
| `square__cdw_s1` | `square` | CDW | accepted fixed point | 6 | -103.346128978834 | 6.625e-4 | 6.139e-8 | 7.382e-9 |

The unfrustrated pairing state has orbit-energy spread
`0.00134058148835`, relative cycle residual `0.00122545994554`, and fails the
configured variational-energy gate. It is therefore a physically interesting
raw-map period-two **candidate**, but is neither accepted nor eligible for an
energy ranking. Its two orbit phases must remain separate; Anderson or linear
mixing must not be used to average it into a fixed point.

## Authorized within-geometry comparisons

`scripts/compare_branches.jl` checked the acceptance gates and required
fingerprints. It authorizes these limited rankings:

- `cubic_frustrated`: CDW-seeded is lowest; SDW-seeded is higher by
  `0.007368263458`, and pairing-seeded by `0.008447412380` in total energy.
- `cubic_unfrustrated`: among the two accepted branches, CDW-seeded is lower
  than SDW-seeded by only `0.000312604756` in total energy. The unresolved
  pairing candidate is excluded.
- `square`: pairing-seeded is lowest; SDW-seeded is higher by
  `0.001045316109`, and CDW-seeded by `0.058892533574` in total energy.

These are seed-to-seed finite-system comparisons at one parameter point. They
do not establish thermodynamic phases, superconducting long-range order, or a
competition between different transverse geometries. Cross-geometry energies
belong to different Hamiltonians and must never be ranked against one another.

## Interpretation and next gates

1. Resolve the unfrustrated pairing recurrence without suppressing it. Inspect
   both stored phases and complete raw history, continue from each phase in a
   recurrence-focused run, and test stability with larger `chi`, tighter DMRG
   convergence, and a second independent pairing seed.
2. Run a second independent seed for each currently lowest accepted branch
   before treating the seed ranking as robust: frustrated CDW, unfrustrated
   CDW/SDW because their splitting is tiny, and square pairing.
3. Use accepted survivors for controlled `L` and `chi` scaling, discarded-weight
   extrapolation, connected spin/charge structure factors, pair correlations,
   gaps, and entanglement diagnostics. One-point MF profiles are diagnostic,
   not sufficient evidence of long-range order.
4. Then scan the frustrated transition over `t0=1.00:0.025:1.20` for
   `V=-0.4,-0.2,0`, with independent seeds and continuation in both directions.
   Use exact `E_p` where available and bracketed interpolation otherwise; add
   selected exact `E_p` calculations to validate the interpolation.
5. Keep all energy comparisons variational: use the stored common canonical
   functional including MF double-counting terms, never a saved effective-MPO
   eigenvalue by itself.

Use `plot_phase1_mf_observables.jl` to render the full schema-v5 histories and
exact seeds. The v2 files have only sparse saved snapshots and cannot be used
to reconstruct a history that was never stored.
