# Bare linear-response survey and isolated-ladder backbone

Status date: 2026-09-01. Planning record; no calculation in this file has been run.
This documents the reasoning from the 2026-09-01 review discussion and the protocol
agreed for the next calculations. It supersedes nothing in `docs/PHASES_0_TO_4.md`;
it inserts a cheaper, seed-free step in front of the Phase 1/2 SCF campaigns.

## 1. Why this program

### 1.1 What the stored histories show

Recomputed from `history/fields/{applied,measured}` in the compact `state.h5` files
(residual `r_k = f(x_k) - x_k`, cosine between successive residuals, one-step and
two-step distances of the measured sequence):

| branch | stored class | cos(r_k, r_k-1) | d2/d1 | residual per step |
|---|---|---:|---:|---|
| v3 `unfrustrated__pairing_s1`, iters 43-49 | periodic_candidate p=2 | +1.000 | 2.00 | grows 1.004x |
| Stage A `phase001` / `phase002` (chi 400) | periodic_candidate p=2 | +1.000 | 1.98 | grows 1.02-1.03x |
| v3 `square__sdw_s1`, iters 16-22 | accepted fixed point | +1.000 | 2.01 | shrinks 0.98x |
| square pilot stripe branches, 6 iters | accepted fixed point | +0.998 | 2.6 | shrinks 0.6x |

A genuine 2-cycle has cos near -1 and d2 << d1. Every "period-two candidate" is monotone
drift along one direction, and the unfrustrated pairing lineage is raw-map unstable
(growth factor above 1). Several accepted fixed points contract by only 2% per step, so
their distance to the true fixed point is r/(1-lambda), about 0.2 relative, not the 5e-3
gate they passed. The slow direction is one-dimensional; a two-vector Anderson step or
Aitken extrapolation removes it, which the 20-raw-update policy forbids.

Second observation: in every campaign the field jumps by one to two orders of magnitude
between MF records 1 and 2, independent of the 1e-3 seed. The ladder ground state
already produces Hartree and exchange fields of order 1e-2 to 1e-1, so zero field is
not a fixed point of the map and a seed added to zero is not a small perturbation of
anything physical.

### 1.2 The map factorizes: geometry enters only through a kernel

In the code the geometry never touches the DMRG. It enters only in how measured ladder
correlators are combined into the next applied field (`calculate_mean_fields`,
`density_kernel`). The map is

    x_out = K_geom(g) . F_ladder(x_in),     g = t_perp^2 / |E_p|

with F_ladder identical for all geometries at fixed (t0, V, n). Linearized at a fixed
point, J = K_geom . chi, where chi is the ladder response. The kernels are leg-space
matrices and diagonalize by leg parity. Density channel, from the code:

| geometry | kernel | even (ky=0) eigenvalue | odd (ky=pi) eigenvalue |
|---|---|---:|---:|
| cubic_frustrated | 2g [2 1; 1 2] | +6g | +2g |
| cubic_unfrustrated | 6g [0 1; 1 0] | +6g | -6g |
| square | 2g [0 1; 1 0] | +2g | -2g |

Pair and exchange channels have the same structure with a 4x4 kernel over leg-pair
classes (00, 11, 01, 10): frustrated weights 2 same-leg and 1 cross-leg in units of 2g;
unfrustrated 3 on the swapped leg plus 2 on the cross terms; square 1 on the swapped leg.
These diagonalize in rung-symmetric versus rung-antisymmetric combinations, which is
where s-like and d-like pairing form factors separate.

For a probe direction e with leg parity p and wavevector q, the stability eigenvalue is
lambda(e) = k_geom(p) . chi(q, p). Its meaning:

- |lambda| < 1 for every probe: the normal state is the stable MF solution.
- |lambda| > 1: the normal state is unstable in that channel; the largest |lambda| is
  the dominant instability. lambda is linear in g, so g_c = g / |lambda| and
  t_perp,c = sqrt(g_c |E_p|).
- sign: lambda > 0 is a uniform transverse order (true fixed point); lambda < 0 flips
  sign each update, i.e. a two-sublattice transverse order, which the single-ladder
  iteration shows as a 2-cycle. It is physical only on a bipartite transverse lattice
  (`cubic_unfrustrated`, `square`). The frustrated kernel has both eigenvalues positive,
  so a staggered instability has no consistent single-ladder realization there.
- lambda within a few percent of 1 predicts the slow SCF drift seen in the histories.

Exact sign bookkeeping (response sign of a repulsive Hartree field, the 1/2 reference
density) must follow `build_mf_mpo` and `calculate_mean_fields`, not textbook
conventions: apply probe fields through the same MPO builder and read the response
through the same correlator function.

### 1.3 Why linearize at a fixed point, and the two tiers

f(x0 + delta) = f(x0) + J delta. The eigenvalue interpretation needs f(x0) = x0, else the
response is dominated by the residual f(x0) - x0. Two admissible expansion points:

| tier | expansion point | ladder solves | reuse across geometries | accuracy |
|---|---|---|---|---|
| bare survey | psi_0, isolated ladder, zero field | 10-25 per (t0, V) | yes; only the kernel differs and is applied analytically | J correct to leading order in g |
| dressed check | psi*(geometry), converged alpha=0 fixed point | Step 1 plus 10-25 per (t0, V, geometry) | no | J exact at the array's normal state |

In the bare tier the response is defined as a difference, delta(e) = [F(h e) - F(0)]/h,
so the nonzero F(0) is subtracted rather than treated as a residual. The dressed tier
is needed only where the bare survey puts some |lambda| within about 20% of 1, and
wherever the normal-state reference energy is needed for a ranking (that energy depends
on the geometry through the kernel and cannot be obtained from the bare ladder).

### 1.4 What the original papers did

Checked against arXiv:2207.03754 (PRX 13, 011039) and arXiv:2301.08116
(PRB 111, 125141):

| ingredient | PRX 2023 | PRB 2025 | this program |
|---|---|---|---|
| Hartree channel | omitted (constant density assumed) | included, mu_i = z_c t^2/dE_p (2<n>-1) | included |
| normal fixed point converged first | no | yes for doped CDW seed (alpha=beta=0 at all iterations, App. A.1) | dressed tier, with beta |
| pairing seed | previous solution, bosonization estimate, low-chi guess | strong local pairing, not small | small field on psi_0 or psi*, linearity checked |
| linear stability / critical coupling | no | no | yes |
| energy ranking of solutions | no | no; order read from which order parameter survives | yes, density corrected |
| convergence tolerance, mixing | exponential-fit plateau, none stated | none stated | explicit gates on extrapolated residual |

Their systems (attractive-U chains, robust SC or CDW) tolerated the missing controls;
repulsive ladders with three competing channels and 1e-5/site splittings do not.

## 2. Isolated-ladder backbone protocol

Per ladder point (t0, V) at fixed U=8, t=1, n=0.9375. Start with L=64; add L=96, 128
at the points that matter. All runs are QN-conserving CPU DMRG (N and S_z) on the CPU
pool. Save every MPS. This replaces the legacy `calculate_E_p_ladder.jl`, whose
registry rows carry only energies; its checkpoint files did write `psi_N_*`, so any
surviving scratch checkpoints can warm-start the sectors below.

### 2.1 Sector energies and states

| sector | purpose |
|---|---|
| (N, 0) | ground state psi_0; reference for all probes and correlators |
| (N, 2S_z=2) | spin gap |
| (N-1, 1), (N-2, 0) | hole pair binding E_p (legacy sign convention: negative = bound) |
| (N+1, 1), (N+2, 0) | particle pair binding; charge gap; mu = [E(N+2) - E(N-2)]/4 |

`sector_resolved_gaps` (src/Diagnostics.jl) already computes all six. Extend it to
return and save the MPSs, the per-sweep energy, max discarded weight, and final
maxlinkdim (via `ITensorMPS.measure!`, which receives `spec`). Use chi ladders
400/800/1200 with warm starts from the lower chi, and record the chi dependence of every
energy difference. The DMRG stopping rule must be an absolute energy tolerance per
sweep with the maxdim ramp finished, not the legacy relative 1e-5.

Protocol changes forced by the legacy-log analysis in section 6:

- Initial product state with the holes spread uniformly along the ladder (or a
  low-chi pre-relaxation of about 15 sweeps at chi <= 200 before the ramp). The legacy
  `ladder_initial_state` piled all eight holes on the last rungs, and DMRG then spent
  most of 100 sweeps at full chi transporting them.
- Keep a small nonzero noise (about 1e-8) until the energy stop fires instead of
  switching noise off after sweep 5.
- Stop on an absolute per-sweep energy change with the maxdim ramp complete, and
  record the last-five-sweep energy change in the artifact as a convergence flag.
- Benchmark 16 Julia threads (Slurm `-c 32`, one eighth node) against the legacy 32
  threads on the first point; the legacy per-sweep time grew only 3x from chi 200 to
  chi 1000, so the runs were overhead-bound and the charge may halve at no cost.

### 2.2 Correlators and diagnostics from psi_0

Reuse `compute_ladder_diagnostics` with `full_pair_correlations=true`: density and
spin profiles, connected charge and spin structure factors at ky = 0, pi, K_rho (both
normalizations), entanglement profile and central-charge fit, sign-resolved singlet pair
matrices for rung and leg bonds. Add the equal-time correlators the MF map uses:
`<c_up c_dn>`, `<c^dag c>` per spin (already in `calculate_mean_fields`). Fit pair and
charge correlation exponents in a bulk window with window-movement uncertainty.

Record the zero-field map output F(0) itself: it is the size of the normal-state
dressing and quantifies the bare-versus-dressed error, order g |F(0)|.

### 2.3 Validity map

Weak-coupling MF requires t_perp small against |E_p|, the spin gap, and the charge
gap, and g z chi_max well defined. Tabulate t_perp / |E_p|, t_perp / Delta_s,
t_perp / Delta_c per point; mark points where E_p -> 0 (registry: near t0 = 1.6-1.8 at
V = 0) as outside the method.

## 3. Bare linear-response survey protocol

For each ladder point, with psi_0 and mu from section 2:

1. Choose probe directions e. Build them with the existing matched-mode templates
   (`matched_mode_profile`, `_matched_pairing_template!`, the stripe/SDW/CDW branches of
   `_matched_mode_initial_fields`) at unit norm:
   - charge, even leg parity: q = 2 pi (1-n) (mode 8 at L=64, the four-k_F value) and
     its two neighbors;
   - spin, odd leg parity: q near pi (1-delta) (modes 58, 59 at L=64) and q = pi;
   - pair, q = 0: form factors onsite_s, rung_s, leg_s, extended_s, d_wave;
   - optional pair at finite q (pair-density-wave check) later.
   Exchange is not an order channel; it enters only through the normal dressing.
2. For each e and two field strengths h, h/2 (small enough that the response is linear
   to 5%; start at field norm 1e-4 per site and adjust), build the MPO from zero field
   plus h e with `build_mf_mpo` and solve by warm-started DMRG from psi_0. QN choices:
   - charge and spin probes: N and S_z conserved (the field is number conserving);
   - pair probes: the field breaks N; run with S_z and fermion parity conserved, or
     dense CPU, at fixed mu from 2.1, warm started from `dense(psi_0)` if the site QNs
     differ. Solve psi_0 once more in that representation as the h = 0 reference.
3. Measure with `calculate_mean_fields` (before any kernel, i.e. the raw correlators,
   or equivalently divide the geometry prefactors back out) and form
   delta(e) = [F(h e) - F(0)] / h. Project: chi(e) = <e, delta(e)> / <e, e>; also keep the
   leakage into other q and parity as a block-diagonality check.
4. Apply the three kernels analytically: lambda_geom(e) = k_geom(p) chi(e) at the
   physical g = t_perp^2 / |E_p|; report lambda, sign, g_c, t_perp,c per channel and
   geometry.
5. Deliverables per ladder point: one HDF5 with all sector MPSs and probe MPSs on
   scratch, a compact mirror (existing `mirror_stateless_tree`), and rows in an
   extended registry CSV: L, U, V, t0, density, chi, E(N), E_p hole, E_p particle,
   spin gap, charge gap, mu, K_rho, pair and charge exponents, F(0) norm, and
   lambda per (channel, q, parity) per geometry with the h and h/2 values.

Cost is roughly six sector solves plus 20-25 warm-started probe solves per ladder
point, all CPU QN DMRG. Probes are short because psi_0 is a good start; the sector
solves dominate. Section 6 gives the measured basis for the estimate and the agreed
scope.

## 4. What comes after the survey

- One channel unstable at a point: seed that channel with delta(e), run the SCF with
  acceleration (adaptive probe, Anderson once the residual direction is fixed), confirm.
- Two or more unstable: seed each and their combination, converge all, rank by the
  density-corrected canonical energy E + mu (N_target - N) against the dressed normal
  reference from the dressed tier.
- All |lambda| < 1 with one within ~20%: dressed tier at that point plus one
  finite-amplitude seeded SCF as a first-order check.
- CDW claims additionally need the bulk four-k_F modulation amplitude scaled in L
  against the isolated-ladder profile from section 2.2 (Friedel control).

## 5. Implementation notes (for the later code change; not done)

- New `scripts/run_ladder_backbone.jl`: sectors, states, diagnostics, registry row.
- New `scripts/run_bare_response.jl`: probes 1-4 above, reusing `build_mf_mpo`,
  `calculate_mean_fields`, `density_kernel`, and the matched-mode templates.
- `sector_resolved_gaps` to save MPSs and truncation evidence; `write_sector_gaps`
  schema version 2.
- Launcher: a CPU-pool action mirroring `submit-ep` with its own sub-ledger.
- Fingerprints: a solver-only implementation fingerprint (src/ plus Manifest) so
  survey and SCF artifacts remain comparable across launcher edits.

## 6. Cost basis from the legacy E_p logs and agreed scope (2026-09-01)

### 6.1 What the legacy scan cost and why

Parsed from the 55 `logs_julia/E_p_ladder_L_64_U_8.0_*.log` files (per-sweep
`After sweep` lines) and `submit_E_p_ladder.sh` (64 Slurm CPUs, shared QOS, one
quarter node, 32 Julia threads, block-sparse):

| item | legacy value |
|---|---:|
| DMRG wall time summed over the U=8 logs | 893 h, about 223 node-hours |
| sectors per point | 3 (N, N-1, N-2) |
| sweeps per sector at chi 1000 | 100, always the cap; the relative early stop never fired |
| wall per sweep | chi 200: 59 s; chi 1000: 184 s; chi 1500: 256 s |
| wall per point | 14-16 h, about 3.9 node-hours |
| energy above final at sweeps 20 / 40 / 60 | about 1.0 / 0.3 / 1e-2 (median over sectors) |

The slow convergence is hole transport, not DMRG cost: the legacy initial state fills
rungs 1-60 with Up/Dn pairs and leaves all eight holes on rungs 61-64, and the noise
term is zero after sweep 5. Roughly three quarters of the legacy node-hours went into
relaxing the density profile at full bond dimension. The weak chi dependence of the
per-sweep time shows the 32-thread runs were overhead-bound.

Registry reliability: 68 of 162 sectors were still changing by more than 1e-4 over
their last five sweeps. The chi 1000 rows at V=0, t0 >= 1.4 moved by 1e-2 to 5e-2 per
sector in the last five sweeps, comparable to E_p itself; the chi 1500 reruns of those
points are at the 1e-3 level. The rows used so far in Phase 1 (t0=1.0 at V=-0.2 and
-0.4) were converged. Logs for the t0=1.2 and 1.4 rows at V != 0 are not in this
checkout and were not checked. Every backbone row must carry its own last-five-sweep
energy change and truncation error so this cannot recur silently.

### 6.2 Estimate per ladder point (L=64, chi 1000, quarter node)

Assumes the protocol changes in section 2.1 (uniform holes, low-chi pre-relaxation,
small noise, absolute stop) and the measured 184 s per chi-1000 sweep.

| component | solves | sweeps at chi 1000 | wall | node-hours |
|---|---:|---:|---:|---:|
| six sectors with states | 6 | 15-25 each | 6-9 h | 1.5-2.3 |
| response probes, warm from psi_0 | 20-25 | 4-8 each | 4-10 h | 1-2.5 |
| pair probes without N conservation (larger blocks) | included | | | +0.5-1 |
| correlators, structure factors, entanglement, pair matrices | | | 1-2 h | 0.3-0.5 |
| total per point | | | | 4-7 |

The pessimistic end assumes hole transport still needs 40 sweeps per sector at chi
1000. Halve the node-hours if the 16-thread benchmark matches the 32-thread sweep time.

### 6.3 Agreed scope

- Scope for now: the 3x3 square grid only, t0 = {1.0, 1.2, 1.4} by V = {0, -0.2, -0.4}
  at L=64, U=8, n=0.9375. The frustrated line t0 = 1.0:0.025:1.2 is deferred.
- Estimated total: 9 points at 4-7 node-hours, 36-63 CPU node-hours, in the CPU pool
  (`m4863`), which NERSC accounts separately from the GPU pool.
- First calculation: the single point t0 = 1.0, V = -0.2, which has a converged legacy
  reference (registry E_p = -0.1545120066237189, chi 1000, last-five-sweep change below
  1e-4 in all three sectors). Run the full backbone plus survey there, record measured
  sweep counts and per-sweep times for 32 and 16 threads, then budget the remaining
  eight points from those numbers before submitting them.
- L = 96 and 128 controls are not part of this scope; they are budgeted later at a few
  points only, at roughly 2-3x the per-point cost.
