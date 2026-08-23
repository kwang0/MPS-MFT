# Configuration guide

`phase0_timing.toml` is a timing-only L=64, chi=64 density-search
payload, with two DMRG sweeps per chemical-potential evaluation.
`phase0_validation.toml` expands the same model and seed to chi=200 and six
sweeps per evaluation. The Phase 0 driver requires every payload repetition to
retarget the configured density because anomalous pairing conserves fermion
parity, not full particle number. Neither config runs an SCF loop or constitutes
a publishable state.

`example_scf.toml` is a production-shaped template. Copy or generate variants before changing it. The `scripts/prepare_branch_scan.jl` helper makes SC, SDW, and CDW configurations that differ only in run lineage and initial seed.

The `[convergence]` defaults reserve the first 20 iterations for an unmixed raw-map probe, accept periods 1 and 2, and only then enable Anderson mixing. This prevents the mixer from averaging away the physical period-two CDW construction of Bollmark et al. (2025). Raise `probe_max_period` and `probe_iterations` before enabling a higher physical period, and document its transverse-sublattice interpretation.

The `[model]` density is particles per physical ladder site, so the target particle number is approximately `2L*density`. `geometry` is one of `cubic_frustrated`, `cubic_unfrustrated`, or `square`.

The `[pair_binding]` registry lookup is exact in `(L,U,V,t0,density)` by default. The selected row uses the largest available chi, then the smallest reported relative difference. Set `allow_interpolation=true` only to linearly interpolate signed `E_p` between the nearest bracketing `t0` values at identical `(L,U,V,density)`; extrapolation and sign-changing brackets are rejected. The mode, endpoints, endpoint chi values, weight, signed result, and `t_perp^2/abs(E_p)` are saved and fingerprinted. `abs(E_p)` is used as the perturbative denominator only for a bound pair unless the explicitly unsafe `allow_unbound_ep=true` override is set.

`phase1_gpu_base.toml` is the representative Phase 1 point. It uses dense CUDA tensors and therefore sets both `conserve_sz=false` and `conserve_nfparity=false`. Do not turn QNs back on in this GPU campaign: that is a materially different, uncalibrated block-sparse CUDA backend.

For a continuation, set `parent_checkpoint` and `parent_sha256`; a nearby model is allowed, and the parent is only a seed. For a same-model restart, set `resume_checkpoint` and `resume_sha256`; the model fingerprint must match. Do not set both.
