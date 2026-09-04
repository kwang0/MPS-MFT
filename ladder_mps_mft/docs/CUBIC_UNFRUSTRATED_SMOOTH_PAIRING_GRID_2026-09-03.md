# Cubic-unfrustrated smooth-pairing eight-point grid fill

Status date: 2026-09-03. This is a loose-accuracy Phase 1 survey. It is not a
thermodynamic phase assignment and does not replace bond-dimension, length, or
same-point basin convergence.

## Question and point inventory

The target grid is `t0={1.0,1.2,1.4}` by `V={-0.4,-0.2,0.0}` at
`L=64`, `U=8`, `t_perp=0.1`, density `0.9375`, and
`cubic_unfrustrated` transverse geometry. The legacy evidence already covers
`(t0,V)=(1.0,0.0)`. This campaign independently starts the other eight cells:

| point ID | t0 | V | exact signed E_p |
|---|---:|---:|---:|
| `cubic_unfrustrated_t010_vm04` | 1.0 | -0.4 | -0.17882744409052975 |
| `cubic_unfrustrated_t010_vm02` | 1.0 | -0.2 | -0.1545120066237189 |
| `cubic_unfrustrated_t012_vm04` | 1.2 | -0.4 | -0.25124588461187614 |
| `cubic_unfrustrated_t012_vm02` | 1.2 | -0.2 | -0.21453418655934797 |
| `cubic_unfrustrated_t012_v000` | 1.2 | 0.0 | -0.17989619749147323 |
| `cubic_unfrustrated_t014_vm04` | 1.4 | -0.4 | -0.24962435880865996 |
| `cubic_unfrustrated_t014_vm02` | 1.4 | -0.2 | -0.2068002629740704 |
| `cubic_unfrustrated_t014_v000` | 1.4 | 0.0 | -0.14653773091916378 |

Every value is an exact row of `data/E_p_values.csv`; preparation rejects
interpolation. No isolated-ladder CPU job is needed.

## Seed and numerical contract

This deliberately mirrors the submitted square-grid protocol. Every point uses
the `legacy_pairing` seed under `matched_mode`, total field norm `1e-3`, and
common RNG seed `1404`. One Gaussian coefficient is drawn for each relative
rung offset and ordered leg-pair class and copied along every allowed rung.
Thus the anomalous field opens the pairing sector without center-of-mass
spatial noise or seed domain walls. `beta=mu_cdw=0` initially, and there is no
field inheritance, parent MPS, or checkpoint resume. The time-zero seed is
saved by the schema-v7 history path. This is a controlled smooth-pairing access
seed, not a mathematically symmetry-unbiased or exhaustive basin search.

All branches share one loose numerical and seed fingerprint:

| control | value |
|---|---:|
| bond dimension and sweeps | 200, 12 |
| DMRG cutoff and energy tolerance | 1e-10, 1e-6 |
| inner and outer density tolerance | 1e-3 |
| field gate | absolute 1e-6 or relative 5e-3 |
| variational-energy stability per site | 1e-6 |
| raw-map probe | up to 20 MF updates |
| total MF limit | 80 updates |

Starting chemical potentials are performance guides: `0.55`, `1.10`, and
`1.65` for `V=-0.4`, `-0.2`, and `0`, respectively. Density is solved at every
update using the carried positive compressibility slope and `1e-8` warm-start
noise. The unmixed physical map comes first; any period-two orbit is preserved
separately before Anderson acceleration. The oscillation classifier and
slow-mode extrapolated-residual gate remain active.

Different `(t0,V)` cells have different model fingerprints and are not a
variational ranking. Never compare their absolute energies with one another or
with square-geometry energies. At a fixed cell, ranking still requires accepted
states with matching model, numerical, implementation, and `E_p` fingerprints
and the canonical variational functional including double counting.

## Storage, concurrency, and accounting

The cubic grid has its own CFS control tree and scratch result tree. Full MPS
objects remain on scratch; CFS receives compact stateless mirrors. It does not
read or write the active square-grid or square `chi=400` campaign directories.
Launcher v1.17.0 remains worker-compatible with their v1.16.0 and v1.15.0 run
environments because solver `src/` and the GPU Manifest are unchanged.

Each branch requests one of four GPUs for 12 hours: `3.000` requested node-hours
per branch and `24.000` for eight first segments. Four segments for all eight
would be `96.000` node-hours and are not pre-authorized. The identical seed at
two square endpoints scales to `1.406388888` actual node-hours for eight jobs,
but older fresh cubic-unfrustrated `chi=200` campaigns scale to
`6.842777776--10.011481480`. Use the latter range for conservative planning;
only Perlmutter `sacct` is authoritative.

The last synchronized active ledger was `42.024097222`. Adding the active
square `chi=400` ceiling (`6.000`), submitted square grid (`15.000`), and this
cubic first-segment envelope (`24.000`) gives an illustrative active total of
`87.024097222`, leaving `312.975902778` under the 400-hour project cap. The
Perlmutter plan and budget commands recompute from the live ledgers and
supersede this illustration. If all 15 scientific branches run simultaneously,
their combined instantaneous fractional rate is `3.75` node-hours per wall
hour; this does not change each job's guarded ceiling.

## Perlmutter handoff

After synchronizing this revision, run on Perlmutter:

```bash
cd "$CFS/m4863/MPS-MFT/ladder_mps_mft"

bash slurm/phase1_gpu.sh plan-cubic-unfrustrated-smooth-pairing-grid

CUBIC_GRID_RUN=20260903_phase1_cubic_unfrustrated_grid_smooth_pairing_chi200_loose
bash slurm/phase1_gpu.sh prepare-cubic-unfrustrated-smooth-pairing-grid "$CUBIC_GRID_RUN"
column -ts $'\t' "output/phase1_gpu/$CUBIC_GRID_RUN/manifest.tsv" | less -S
bash slurm/phase1_gpu.sh budget
```

Preparation neither submits nor reserves. When the manifest and live budget
look correct, the guarded direct submission creates eight scientific jobs and
no smoke job:

```bash
bash slurm/phase1_gpu.sh submit "$CUBIC_GRID_RUN"
bash slurm/phase1_gpu.sh status "$CUBIC_GRID_RUN"
```

After all eight allocation rows are terminal:

```bash
bash slurm/phase1_gpu.sh reconcile "$CUBIC_GRID_RUN"
bash slurm/phase1_gpu.sh budget
```
