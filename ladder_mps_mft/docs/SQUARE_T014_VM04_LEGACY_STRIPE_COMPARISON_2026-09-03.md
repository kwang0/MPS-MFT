# Square `(t0,V)=(1.4,-0.4)` legacy-stripe comparison

## Scientific question

This two-branch `L=64`, `U=8`, `tp=0.1`, density `0.9375`, square-geometry,
`chi=200` loose campaign asks whether the high-amplitude stripe basin found by
the legacy square calculation at `(t0,V)=(1.0,0.0)` remains a self-consistent
minimum after a direct parameter jump to `(1.4,-0.4)`. If it does, the paired
current-solver control supplies the same-campaign energy reference needed to
decide which accepted basin has lower canonical variational energy.

The branches are:

1. `square__smooth_pairing_current_t014_vm04_chi200_loose`: a fresh product MPS
   and the established center-of-mass-uniform `legacy_pairing` matched-mode
   field seed with total norm `1e-3` and RNG seed `1404`.
2. `square__legacy_t010_v000_stripe_inherit_t014_vm04_chi200_loose`: a fresh
   product MPS and the terminal physical `alpha`, `beta`, `mu_cdw`, and `mu`
   from the exact legacy `(1.0,0.0)` stripe artifact. Its amplitude and spatial
   texture are not renormalized or smoothed.

Both branches use a fresh MPS, so this is a field-basin test rather than an MPS
checkpoint continuation. The exact time-zero fields are saved in the MF
history before the first update.

## Immutable source and field-only derivative

The required legacy source is:

`$CFS/m4863/MPS-MFT/stateless_data/results_L_64_U_8.0_V_0.0_t0_1.0_t_p_0.1_geometry_square_chi_200_density_0.9375_gpu.h5`

Its required SHA-256 is
`ae6a3bfe76ca8f06f2396fd731b18bca8539e0b7ee68df016cc9156fdceeb074`.
Preparation refuses any other file. The source is read only and is neither
modified nor copied wholesale.

The legacy tensor contains 256 same-physical-site `beta` entries with maximum
absolute value `0.1282449874`. Those entries are skipped by both current MPO
construction and the canonical energy functional, while a raw all-field
residual would count them. Preparation therefore writes a small derived HDF5
seed under the new CFS run tree and zeros exactly those inactive entries. It
verifies that every other `beta` entry and all `alpha` and `mu_cdw` entries are
bitwise unchanged. The original and derived paths and hashes and the named
sanitization policy are recorded in `manifest.tsv`.

The source's physical field scales provide a guard against accidentally using
a weak or wrong state: `max|alpha|=2.05515e-6`, active
`max|beta|=0.03440698`, `max|mu_cdw|=0.05339365`, and
`mu=1.6586343178`.

## Numerical and physical policy

- Exact signed `E_p=-0.24962435880865996`; no interpolation or new isolated-
  ladder calculation.
- Float64 CUDA, 12 DMRG sweeps, `chi=200`, cutoff `1e-10`, and DMRG energy
  tolerance `1e-6`.
- Inner and outer density tolerance `1e-3`, with the carried positive
  compressibility slope and `1e-8` warm-start noise for chemical-potential
  re-solves.
- Loose field gate: absolute residual `1e-6` or relative residual `5e-3`, plus
  the oscillation and slow-mode extrapolation acceptance gates.
- Up to 20 raw-map updates before Anderson acceleration. A physical raw
  recurrence is preserved and classified separately from mixer behavior.
- Full MPS artifacts go to scratch; only MPS-free results and the field-only
  seed remain on CFS.

## Interpretation and energy boundary

The inherited stripe is operationally stable only if the current solver
produces an accepted period-one or accepted physical period-two result that
retains the stripe texture. Basin escape to the paired endpoint is evidence
against stability under this protocol. A time limit, stagnation, or unaccepted
candidate leaves the question unresolved.

An energy comparison is authorized only if both new endpoints are accepted and
their geometry, model, numerical, implementation, scalar, and `E_p` provenance
match. Rank with the canonical variational energy including all double-counting
terms; preserve orbit phases separately if either endpoint is periodic. The
legacy stored `E=-267.99503` is an effective-Hamiltonian-era quantity and is
not rankable. The six prior `(1.4,-0.4)` endpoints are useful qualitative
context, but they predate the current numerical/provenance fingerprints and
must not enter the formal two-state ranking.

## Perlmutter handoff

Preparation is local to Perlmutter CFS/scratch metadata. It submits nothing and
does not reserve node-hours:

```bash
cd "$CFS/m4863/MPS-MFT/ladder_mps_mft"
bash slurm/phase1_gpu.sh plan-square-legacy-stripe-compare

STRIPE_SOURCE="$CFS/m4863/MPS-MFT/stateless_data/results_L_64_U_8.0_V_0.0_t0_1.0_t_p_0.1_geometry_square_chi_200_density_0.9375_gpu.h5"
STRIPE_RUN=20260903_phase1_square_t014_vm04_legacy_stripe_compare_chi200_loose
bash slurm/phase1_gpu.sh prepare-square-legacy-stripe-compare \
  "$STRIPE_SOURCE" "$STRIPE_RUN"

column -ts $'\t' "output/phase1_gpu/$STRIPE_RUN/manifest.tsv" | less -S
bash slurm/phase1_gpu.sh budget
```

After inspection, direct submission is:

```bash
bash slurm/phase1_gpu.sh submit "$STRIPE_RUN"
bash slurm/phase1_gpu.sh status "$STRIPE_RUN"
```

No standalone smoke is created. Two 12-hour, one-of-four-GPU requests reserve
`6.000` node-hours. Completed old target-point branches suggest roughly
`0.2133--0.2469` actual node-hours for two analogous jobs, but the inherited
stripe may take longer if it drifts or escapes its basin. Perlmutter `sacct`
and the live append-only project ledger are authoritative.
