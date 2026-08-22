# Phases 0 to 4

Status date: 2026-08-22.

## Phase 0 — CPU resource calibration

Status: v2 matrix completed on Perlmutter but is rejected for wrong density;
the corrected v1.2.0 workflow is implemented and awaits a new v3 submission.

The matrix uses one immutable, density-targeted L=64, chi=64 warm-start state
and benchmarks 11 exclusive configurations: serial, block-sparse at 2/4/8/16
threads, Strided at 2/4/8 threads, and BLAS at 2/4/8 threads. Every candidate
runs three copies of the same density search, with two DMRG sweeps per search
evaluation. A candidate is eligible only if every repetition reaches the
configured target density, follows the serial search path, reproduces
`serial-t1` energy, density, and chemical potential within the report
tolerances, has exact provenance and exclusive thread topology, and records
MaxRSS.

The current guarded worst-case reservation is 1.511718750 Perlmutter CPU
node-hours, below the 3-node-hour Phase 0 cap. The completed v2 matrix used only
about 0.1017513 node-hours, but its `n=0.5614` workload missed the `n=0.9375`
target and cannot select a backend. After a fresh schema-v3 matrix passes,
submit exactly one chi=200, six-sweep-per-evaluation validation using the
recommended backend and right-sized memory. Do not use the winner for
production before that validation.

Acceptance evidence:

- all matrix and validation job IDs and terminal Slurm states;
- immutable seed and metric SHA-256 values;
- `summary.csv`, `recommendation.md`, MaxRSS, and projected shared-QOS charge;
- numerical equivalence to serial and successful chi=200 validation;
- an entry in `RUN_LOG.md` that distinguishes timing validation from scientific convergence.

## Phase 1 — controlled fixed-point and periodic branches

Run independent pairing, SDW, and CDW seeds at one representative model point for each transverse geometry. Begin with the unmixed period-two probe. Require either gated period-one convergence or a gated all-phase periodic solution, then repeat the lowest-energy branch from a second seed. Archive every orbit phase without averaging it; periods beyond two remain candidates until their transverse-sublattice meaning is established.

Phase 1 is complete only when accepted immutable fixed points or unmixed validated periodic solutions can be ranked by the common zero-temperature functional at matched model and numerical fingerprints. Periodic branches use phase-resolved artifacts and orbit-averaged energies; fields are never averaged.

## Phase 2 — transition and hysteresis scans

For the frustrated geometry, resolve the candidate transition near t0=1.0–1.2 with spacing 0.025 for V=-0.4,-0.2,0. Use independent seeds plus forward and reverse continuations. Preserve every branch and its parent SHA; do not replace a metastable continuation merely because another state has lower energy.

Apply a targeted square-geometry scan in t0 and t_perp only after Phase 1 establishes reliable branch controls. Record recurrence and convergence status at every point.

## Phase 3 — finite-size, bond-dimension, and isolated-ladder controls

Use at least L=32,48,64,96,128 where E_p and model coverage permit, and chi=200,400,800,1200+ as resources allow. Extrapolate energy and long-distance observables against discarded-weight or controlled chi proxies. Recompute or extend the E_p registry rather than interpolating it silently.

Run the fixed-sector diagnostics separately to obtain spin and charge gaps and particle/hole pair binding. Check t_perp against all known isolated-ladder scales, not only |E_p|.

## Phase 4 — publication analysis

Report common-functional branch energies, finite-size/chi uncertainty, structure-factor peak positions and scaling, `K_rho`, entanglement fits, and sign-resolved pair correlations. Separate robust findings, candidate phase boundaries, finite-chi enhancement, and unresolved cases. Archive the exact accepted states, configs, hashes, code commit, implementation fingerprint, and selection table used for every figure.
