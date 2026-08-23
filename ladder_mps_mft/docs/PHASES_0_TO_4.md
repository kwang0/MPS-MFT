# Phases 0 to 4

Status date: 2026-08-22.

## Phase 0 — CPU resource calibration

Status: v2 completed and is sufficient to shortlist `serial-t1` and
`blocksparse-t4`. Its fixed-`mu=0`, `chi=64`, two-sweep calculation was at the
wrong physical density, so it is backend screening rather than a final
production measurement. The v3 seed's density search failed after 16
evaluations and produced no benchmark data.

Script v1.3.1 removes density targeting from the timed workload and performs
the missing production-scale comparison directly: fixed `mu=1.8`, `L=64`,
`chi=200`, six sweeps, two repetitions, and only the two shortlisted backends.
Each timer encloses exactly `run_dmrg_ground`. The fixed-mu warm seed, MPO
construction, compilation, MPS copying, GC, density measurement, and chemical-
potential search are excluded.

The guarded worst-case reservation is `0.570312500` Perlmutter CPU node-hours,
below the three-node-hour cap. Candidates must pass provenance, topology,
numerical-equivalence, MaxRSS, and timing-stability gates. This matrix is itself
the production-scale confirmation; no separate validation job follows it.

The pre-run legacy-GPU comparison is an extrapolation, not matched evidence:
approximately 35--60 s and `0.00243--0.00417` GPU node-hours for a six-sweep
`chi=200` fixed-mu solve, versus projected CPU times of 44--78 min for
`blocksparse-t4` and 62--109 min for `serial-t1`. CPU and GPU node-hours belong
to separate pools, and the CPU uses `S_z` plus fermion-parity conservation while
the legacy GPU path uses no QNs.

Acceptance evidence:

- seed, candidate, and report job IDs with terminal Slurm states;
- immutable seed and metric SHA-256 values;
- `summary.csv`, `recommendation.md`, MaxRSS, and actual `sacct` charge;
- exact fixed-mu timing-region and thread-topology metadata;
- numerical equivalence to serial and stable repeated timing; and
- an entry in `RUN_LOG.md` that distinguishes timing calibration from
  scientific convergence.

## Phase 1 — controlled fixed-point and periodic branches

Run independent pairing, SDW, and CDW seeds at one representative model point
for each transverse geometry. Begin with the unmixed period-two probe. Require
either gated period-one convergence or a gated all-phase periodic solution,
then repeat the lowest-energy branch from a second seed. Archive every orbit
phase without averaging it; periods beyond two remain candidates until their
transverse-sublattice meaning is established.

Phase 1 is complete only when accepted immutable fixed points or unmixed
validated periodic solutions can be ranked by the common zero-temperature
functional at matched model and numerical fingerprints. Periodic branches use
phase-resolved artifacts and orbit-averaged energies; fields are never averaged.

## Phase 2 — transition and hysteresis scans

For the frustrated geometry, resolve the candidate transition near t0=1.0--1.2
with spacing 0.025 for V=-0.4,-0.2,0. Use independent seeds plus forward and
reverse continuations. Preserve every branch and its parent SHA; do not replace
a metastable continuation merely because another state has lower energy.

Apply a targeted square-geometry scan in t0 and t_perp only after Phase 1
establishes reliable branch controls. Record recurrence and convergence status
at every point.

## Phase 3 — finite-size, bond-dimension, and isolated-ladder controls

Use at least L=32,48,64,96,128 where E_p and model coverage permit, and
chi=200,400,800,1200+ as resources allow. Extrapolate energy and long-distance
observables against discarded-weight or controlled chi proxies. Recompute or
extend the E_p registry rather than interpolating it silently.

Run the fixed-sector diagnostics separately to obtain spin and charge gaps and
particle/hole pair binding. Check t_perp against all known isolated-ladder
scales, not only |E_p|.

## Phase 4 — publication analysis

Report common-functional branch energies, finite-size/chi uncertainty,
structure-factor peak positions and scaling, `K_rho`, entanglement fits, and
sign-resolved pair correlations. Separate robust findings, candidate phase
boundaries, finite-chi enhancement, and unresolved cases. Archive the exact
accepted states, configs, hashes, code commit, implementation fingerprint, and
selection table used for every figure.
