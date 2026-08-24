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

There are three mutually exclusive lineage modes:

- `inherit_from` plus `inherit_sha256` reproduces the legacy field-only
  inheritance contract. It reads `alpha`, `beta`, `mu_cdw` (or zeros when an
  older legacy file has no `mu_cdw`), and `mu`, but deliberately creates a
  fresh product MPS and fresh site indices. Both legacy top-level HDF5 files
  and refactored `fields/restart` states are accepted.
- `parent_checkpoint` plus `parent_sha256` is a continuation seed that reuses
  both the MPS and fields; a nearby model is allowed.
- `resume_checkpoint` plus `resume_sha256` is a same-model restart that reuses
  both the MPS and fields and requires the model fingerprint to match.

Every SHA is mandatory, and only one mode may be set. New schema-v5 states
store the exact seed in `fields/initial` plus the full
`history/fields/applied` and `history/fields/measured` arrays at every SCF
iteration; the final dimension matches `history/iteration`.

Generate a validated, SHA-pinned field-only inheritance config without editing
the source artifact:

```bash
julia --project=. scripts/prepare_field_inherit.jl \
  /path/to/legacy_state.h5 base_config.toml inherited_run.toml
```
