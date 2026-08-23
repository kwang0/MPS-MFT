# Phases 0 to 4

Status date: 2026-08-22.

## Phase 0 — resource calibration

Status: closed. The CPU matrices remain backend-equivalence and diagnostic
evidence, but the available CPU timings and legacy-GPU timing evidence indicated
roughly a two-order-of-magnitude wall-time disadvantage for CPU production.
The refactored solver, not the legacy SCF implementation, is now the production
path; only its tensor storage and DMRG device move to CUDA.

The retained CPU script remains available for audit, but it is not a gate for
Phase 1. CPU states used `S_z` plus fermion-parity QNs; production GPU states
are dense and use no QNs. This representation difference is explicitly stored
in the numerical fingerprint.

## Phase 1 — refactored GPU fixed-point and periodic branches

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

The initial point is `L=64`, `U=8`, `V=-0.2`, `t0=1.1`, `t_perp=0.1`,
`density=0.9375`, and `chi=200`. Its signed `E_p` is linearly interpolated
between the exact registry entries at `t0=1.0` and `1.2`; all endpoint data are
stored. A 30-minute GPU smoke test precedes the nine 12-hour shared-GPU branch
segments. The initial worst-case reservation is `27.125` node-hours, and four
segments for all nine branches would reserve `108.125`.

The project-wide launcher ledger enforces a conservative cap of 400 additional
node-hours from the user-reported 277-node-hour baseline. It sums requested CPU
and GPU upper bounds even though NERSC maintains separate allocation pools and
does not reclaim unused walltime. Continuations are explicit; no job resubmits
itself.

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
extend the `E_p` registry where interpolation uncertainty matters. Bracketed
`t0` interpolation is permitted for fine transition scans only when its
endpoints and effective coupling `t_perp^2/|E_p|` are stored; extrapolation is
forbidden.

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
