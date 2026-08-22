# Configuration guide

`phase0_timing.toml` is a timing-only L=64, chi=64, two-sweep payload. `phase0_validation.toml` expands the same model and seed to chi=200 and six sweeps. Neither runs an SCF loop or constitutes a publishable state.

`example_scf.toml` is a production-shaped template. Copy or generate variants before changing it. The `scripts/prepare_branch_scan.jl` helper makes SC, SDW, and CDW configurations that differ only in run lineage and initial seed.

The `[convergence]` defaults reserve the first 20 iterations for an unmixed raw-map probe, accept periods 1 and 2, and only then enable Anderson mixing. This prevents the mixer from averaging away the physical period-two CDW construction of Bollmark et al. (2025). Raise `probe_max_period` and `probe_iterations` before enabling a higher physical period, and document its transverse-sublattice interpretation.

The `[model]` density is particles per physical ladder site, so the target particle number is approximately `2L*density`. `geometry` is one of `cubic_frustrated`, `cubic_unfrustrated`, or `square`.

The `[pair_binding]` registry lookup is exact in `(L,U,V,t0,density)`. If no row exists, the run stops. The selected row uses the largest available chi, then the smallest reported relative difference. The signed value is saved, while `abs(E_p)` is used as the perturbative denominator only when the registry value indicates a bound pair, unless the explicitly unsafe `allow_unbound_ep=true` override is set.

For a continuation, set `parent_checkpoint` and `parent_sha256`; a nearby model is allowed, and the parent is only a seed. For a same-model restart, set `resume_checkpoint` and `resume_sha256`; the model fingerprint must match. Do not set both.
