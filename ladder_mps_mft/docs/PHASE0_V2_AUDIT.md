# Phase 0 v2 calibration audit

Status date: 2026-08-22.

## Decision

Run `20260821_phase0_cpu_v2` is valid evidence that the eleven CPU backend
configurations executed the same finite-DMRG calculation reproducibly. It is
**not valid evidence for choosing the production backend**, because neither the
seed nor the timing payload targeted the configured density. Do not submit its
chi=200 validation and do not promote its `serial-t1` recommendation.

The replacement run must use Phase 0 script v1.2.0 (metric schema v3) and a new
immutable run ID. It times the same density-targeting operation used by the SCF
solver, not a fixed-chemical-potential proxy.

## Authoritative evidence

The synced artifacts are under
`output/phase0_calibration/20260821_phase0_cpu_v2/`.

- Seed job `57393191`, all eleven matrix jobs `57393193`--`57393215`, and report
  job `57393217` completed with exit code `0:0` according to `sacct.txt`.
- All metrics have commit
  `acc60f1725ce9647a57ca9256d6813e5c73e0d71`, implementation fingerprint
  `8a8920aa75996298b836f6e584bca6946f0f1ce1009212e53c25ab33b084d2fa`,
  config SHA-256
  `105a0e78f48324ee4d01942590d61a234bf63dab4aabc7239546dcaf79317e59`,
  E_p-registry SHA-256
  `2209bd2ca3c1ad02c0e542d1a9d63ecf90fdfa49120ad9cc3af599a5b4bc1f0e`,
  and seed SHA-256
  `b97dc1f3b8e8943e742422d70d84fa05110f7a08e1283c3c7978bae4ca497f29`.
- The three repeats of every candidate are identical in energy and density
  within the stored precision. Cross-backend differences are at most about
  `1e-15` in energy per physical site and `1.3e-14` in density.

These checks support internal consistency and provenance. They do not repair a
workload-definition error.

## Density failure

The model target is `n_target = 0.9375` particles per physical ladder site.
The seed has `n = 0.5614556102812898`; every timed solve instead reaches
`n = 0.56137567423918...`. The timing-payload error is therefore
`|n-n_target| = 0.3761243257608...`, many orders of magnitude above the new
`5e-4` Phase 0 targeting tolerance.

The cause is structural. The MPS+MF sites conserve total `S_z` and fermion
number parity, but the anomalous pairing field changes particle number by two,
so full `N_f` conservation is intentionally unavailable. The v2 seed and
payload built the grand-canonical Hamiltonian at `mu=0` and called a single
DMRG solve directly. They bypassed `find_mu_for_density`, so the configured
density was only used to form the initial product state and was not enforced.

## Conditional performance observation

At the wrong-density fixed-`mu` workload, the generated report selected
`serial-t1`: median `71.666 s`, timing relative range `0.77%`, MaxRSS
`1.368 GiB`, and a right-sized projection of `4 GiB`, two physical cores, and
`3.1105e-4` shared-QOS node-hours per solve. `strided-t2` was only `3.5%` more
expensive by that projection. `blocksparse-t4` was the fastest in wall time at
`51.318 s` (about `1.40x` speedup) but cost `1.43x` as many projected
node-hours. `strided-t4` was both slow and noisy (about `24%` repeat range).

The actual charge reconstructed from parent-job elapsed times and allocated
CPU cores is approximately `0.101751302083` node-hours for the seed, complete
matrix, and report. This is budget evidence only. Backend rankings can change
when one payload requires multiple density-search DMRGs, so none of the v2
performance ordering is promoted.

## Implemented correction and acceptance gate

Phase 0 v1.2.0 now:

1. creates the common seed with `find_mu_for_density` and records its target,
   achieved density, chemical potential, status, and evaluation count;
2. starts every repetition from the identical seed but reruns the full density
   search, matching the production solver's workload;
3. records every converged chemical potential, density-search status, and DMRG
   evaluation count in metric schema v3;
4. rejects candidates with a target-density miss, failed search, different
   search path, chemical-potential mismatch, backend/thread mismatch, numerical
   mismatch, timing range above 10%, provenance mismatch, or absent MaxRSS; and
5. evaluates chi=200 validation at its own converged chemical potential while
   requiring target density, the selected topology, provenance, and MaxRSS.

Only after a fresh matrix passes these gates should one chi=200 validation be
submitted. Phase 0 still measures performance and numerical reproducibility;
it does not establish SCF convergence, a physical phase, or a CPU/GPU
crossover.

## 2026-08-22 revised interpretation after v3

The preceding decision is preserved as the original audit, but Phase 0's scope
has since been narrowed to choosing a DMRG configuration. Under that scope, v2
is useful screening evidence: all backends performed the same fixed-mu solve,
`blocksparse-t4` was the wall-time winner at `51.318 s`, and `serial-t1` was
the most stable low-charge baseline at `71.666 s`. All other candidates can be
removed from the next matrix.

The v3 density-targeted seed job `57405642` failed after 16 chemical-potential
evaluations with status `maximum_mu_iterations`, reaching density
`0.9843323116910832` instead of `0.9375`. It created no seed or benchmark data.
This is a numerical search failure, not a scheduler or resource failure.

Script v1.3.1 therefore leaves the production density-search algorithm
untouched and performs a focused fixed-mu comparison of only `serial-t1` and
`blocksparse-t4`. It uses `mu=1.8`, `L=64`, `chi=200`, six sweeps, and two
identical repetitions. Only `run_dmrg_ground` is timed; compilation, MPO
construction, MPS copying, GC, density measurement, and chemical-potential
search are excluded. This directly supplies the production-scale evidence that
v2 lacked, so no separate chi=200 validation job is needed.

The v2 result still cannot establish a CPU/GPU crossover. The current legacy
GPU estimate is 35--60 s for a six-sweep `chi=200` solve, derived from saved
`chi=500` and `chi=1000` sweep logs. A matched GPU timing and `sacct` record are
required before making a definitive cost claim.
