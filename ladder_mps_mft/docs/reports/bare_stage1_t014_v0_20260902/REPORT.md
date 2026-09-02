# Bare-Ladder Stage 1 at V=0, t0=1.4

**Decision:** proceed to the gated Stage 2 discovery calculation, but treat its
instability eigenvalues as exploratory rather than a controlled weak-coupling
prediction.

The six-sector L=64, chi=1200 backbone is numerically usable. Its final spin
gap is 0.153361332, charge gap 0.010858301, hole pair
binding -0.146510221, and particle pair binding
-0.148542748. The rung-pair correlation decays more
slowly than the charge correlation: exponents 0.7614 +/- 0.0215
and 1.2158 +/- 0.0830, respectively. The pair fits are much
more stable (R2 0.953--0.982) than the charge fits (R2 0.527--0.707).

This is not yet a susceptibility result. The equal-time covariance spectra are
broad: participation ranks range from 48.0
to 63.4. They therefore argue
against assuming a tiny low-rank response, while still supplying useful
data-driven candidate directions. The Stage 2 bank combines 11 motivated names
with three covariance additions. Exact orthogonalization reduces those 14 names
to 12 independent field directions: nine normal and three pairing.

## Numerical convergence

- All six final chi=1200 sectors pass the saved convergence gates.
- From chi=800 to 1200, the spin gap moves -0.900%,
  the charge gap -0.029%, and
  the magnitudes of the hole and particle bindings by less than 0.03%.
- The chi=200 derived gaps are not physical results: not all sectors were
  converged at that stage.
- The central-charge fit gives c=2.009 but
  R2=0.155; it is unusable as a central-charge
  estimate.

## Physics interpretation

- K_rho is 1.4247 with the rung normalization
  and 0.7124 with the site normalization.
  The convention must be named whenever this number is quoted.
- The open-boundary density profile has its strongest Fourier component at
  q/pi=0.1250, the expected four-kF-scale Friedel modulation.
- The leading rung and leg pair covariance vectors are q=0-like, with uniform
  overlaps 0.980 and 0.975.
- The two largest spin-odd covariance vectors are boundary modes (edge weights
  0.980 and 0.980); the first bulk spin-odd
  candidate is rank 3, with covariance eigenvalue 0.675196.
- In charge-even covariance, the first mode whose dominant Fourier component is
  q/pi=0.125 appears only at rank 11, with eigenvalue
  0.129339. A top-six-only rule would have missed it.
- Separate covariance matrices cannot decide an extended-s versus d-wave
  rung/leg mixture. That is exactly the mixing Stage 2 will determine.

## Method-validity boundary

At tp=0.1, tp/|Ep|=0.683,
tp/Delta_s=0.652, and
tp/Delta_c=9.210. The charge ratio is especially
large. The bare MPS+MF expansion is therefore a diagnostic ordering tendency,
not a controlled weak-coupling prediction at this point. With
g=tp^2/|Ep|=0.068255, g||F(0)||=0.300; normal-state dressing is
not parametrically negligible.

## Efficiency

The logs contain 351 DMRG sweeps and 27.037 summed
DMRG wall-hours. Because the six sector jobs ran concurrently, the ideal
sector-array critical path is 9.051 hours. The spin-excited
sector is the bottleneck. The full scratch tree represented in the compact
manifest is 6.725 GiB; the stateless mirror is
2.046 MiB, a 3366x reduction.
Final MPS data are intentionally present in the assembled backbone, final
sector files, and chi=1200 checkpoints; that triplication accounts for
79.6% of the full tree.

No `sacct` or `/usr/bin/time -v` record was synchronized. Consequently this
report does not claim measured CPU utilization, charged node-hours, or peak
resident memory. The four-thread block-sparse topology remains the only
production choice supported by the repository's calibration; Stage 2 obtains
parallelism across independent probe jobs instead of adding unbenchmarked
threads inside each DMRG solve.

## Stage 2 pilot now prepared

Discovery performs one finite-field solve at h=1e-4 for each of the 12 basis
directions, plus two representation-matched zero-field references. The strict
number-conserving zero-field re-solve is essential: the saved backbone's
last-five energy spread is much larger than h times the desired response
accuracy, so an unrelaxed baseline would create a common O(residual/h)
contamination. The pairing reference drops only Nf while preserving fermion
parity and Sz.

The nine normal probes and the pairing-reference job run concurrently after
the strict zero-field reference. Three parity-only pairing probes follow that
reference. Each measured ladder response is reused for cubic frustrated,
cubic unfrustrated, and square kernels. Their prefactor uses the backbone's
measured hole binding, -0.146510221, rather than the older
registry interpolation. Discovery must pass DMRG convergence,
5% within-block reciprocity, and a 5% normal/pair cross-block gate before it
emits the validation plan. Validation is a separate submission: three selected
eigenvectors are checked at h and h/2, and the final response uses Richardson
extrapolation only if the 5% linearity gate passes.

The launcher's conservative reservation ceilings are 18.656 CPU node-hours for
discovery and 9.609 for the optional validation. These are walltime-memory
reservation bounds, not measured charges.

## Evidence boundary

All quantitative claims above are derived from `20260901_bare_t014_v0_stage1`'s synchronized
compact artifacts and logs. The 33 manifest rows were present, and the compact
backbone and Stage 1 HDF5 files were hash-checked during this report build. The
full restartable MPS tree remains on Perlmutter scratch and was not locally
opened. No Perlmutter connection, transfer, scheduler query, or submission was
performed.
