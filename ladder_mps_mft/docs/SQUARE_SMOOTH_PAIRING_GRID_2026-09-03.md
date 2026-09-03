# Square smooth-pairing five-point grid fill

Status date: 2026-09-03. This is a loose-accuracy Phase 1 survey, not a
thermodynamic phase assignment or a substitute for bond-dimension and length
convergence.

## Question and scope

The target square grid is

```text
t0 = {1.0, 1.2, 1.4}
V  = {-0.4, -0.2, 0.0}
```

The working evidence set already represents `(t0,V)=(1.0,0.0)`,
`(1.4,-0.4)`, `(1.4,-0.2)`, and `(1.4,0.0)`. This campaign fills only the five
missing coordinates:

| point ID | t0 | V | exact signed E_p |
|---|---:|---:|---:|
| `square_t010_vm04` | 1.0 | -0.4 | -0.17882744409052975 |
| `square_t010_vm02` | 1.0 | -0.2 | -0.1545120066237189 |
| `square_t012_vm04` | 1.2 | -0.4 | -0.25124588461187614 |
| `square_t012_vm02` | 1.2 | -0.2 | -0.21453418655934797 |
| `square_t012_v000` | 1.2 | 0.0 | -0.17989619749147323 |

Every value is an exact row of `data/E_p_values.csv`; interpolation and new
pair-binding jobs are forbidden by the preparation contract.

## Common seed

Each point uses the previously tested `legacy_pairing` seed under the
`matched_mode` protocol:

- total initial field norm per physical site: `1e-3`;
- field RNG and product-MPS RNG seed: `1404`;
- one Gaussian coefficient per relative rung offset and leg-pair class;
- that coefficient is copied along all allowed center-of-mass positions;
- `beta=0` and `mu_cdw=0` at time zero;
- no inherited fields, parent MPS, or resumed checkpoint.

This is therefore a smooth mixed-pairing seed: it samples relative pairing
components without injecting random signs or amplitudes along the ladder, so
the seed itself contains no center-of-mass domain walls. It is not literally
symmetry-unbiased. Any nonzero anomalous field opens the pairing sector, while
an exactly zero anomalous field would leave the finite-ladder DMRG in the
number-conserving `alpha=0` invariant subspace. The choice is a controlled
pairing-access survey, not proof that other basins are absent.

The time-zero seed is included in the saved MF history by the current solver.
Because all five points have the same `L`, density, range, protocol, amplitude,
and RNG seed, they receive one common initial-seed fingerprint. Their model
fingerprints are necessarily different because `(t0,V,E_p)` differs.

## Numerical contract

All five branches use the same loose numerical fingerprint:

| control | value |
|---|---:|
| geometry | square |
| L, U, t_perp, density | 64, 8, 0.1, 0.9375 |
| bond dimension | 200 |
| DMRG sweeps | 12 |
| DMRG cutoff | 1e-10 |
| DMRG energy tolerance | 1e-6 |
| inner and outer density tolerances | 1e-3 |
| field acceptance | absolute 1e-6 or relative 5e-3 |
| variational-energy stability per site | 1e-6 |
| raw-map probe | up to 20 MF updates |
| overall MF limit | 80 updates |

The starting chemical potential is a performance seed, not a fixed physical
parameter. It is `0.55` for `V=-0.4`, `1.10` for `V=-0.2`, and `1.65` for
`V=0`, using the current `t0=1.4` endpoints and their midpoint as rounded
bracketing guides. The solver still independently fixes the density at every
point. It carries a positive compressibility slope across SCF updates and uses
the `1e-8` warm re-solve noise schedule.

The first 20 updates are the unmixed physical map. A raw-map recurrence is
recorded independently of any later Anderson acceleration. The oscillation
criterion and slow-mode extrapolated residual gate remain active. Only accepted
states with matching geometry, model, numerical, implementation, and pair-
binding fingerprints may be variationally ranked. In particular, energies at
different grid coordinates are not a same-Hamiltonian ranking.

## Storage and concurrency

The campaign has its own run ID and output roots. It neither reads from nor
writes to the running square `chi=400` two-lineage campaign. Full MPS artifacts
go to the campaign's Perlmutter scratch tree; CFS receives stateless analysis
mirrors. If the repository is synchronized while a launcher-v1.15.0 `chi=400`
job is still pending, launcher v1.16.0 explicitly retains worker compatibility.
An explicit continuation of that one active campaign is also allowed because
the solver `src/` and GPU Manifest are unchanged.

Preparing the grid advances the convenience pointer `latest_run.txt`. This
does not alter the `chi=400` campaign, but status and continuation commands for
both campaigns should keep passing their explicit run IDs.

The live ledger is checked under its lock immediately before the five jobs are
submitted. Consequently every unreconciled `chi=400` reservation already
recorded on Perlmutter is counted; there is no separate or stale budget
allowance for this grid. The locally synchronized copy may lag the running
campaign and is not authoritative.

## Cost estimate

Each job requests one of four GPUs for 12 hours, so the conservative ledger
ceiling is `3.000` node-hours per branch and `15.000` node-hours for five first
segments. Four segments for all five would be `60.000` node-hours and are not
pre-authorized.

The same smooth mixed-pairing branch measured:

| prior point | elapsed | measured fractional charge |
|---|---:|---:|
| `t0=1.4,V=-0.4` | 1778 s | 0.123472222 node-hours |
| `t0=1.4,V=0.0` | 3285 s | 0.228125000 node-hours |

Scaling their mean to five branches gives `0.878993055` node-hours. Scaling
the two endpoint rates separately gives `0.617361110` to `1.140625000`
node-hours. The broader mean over all twelve successful branches at those two
points gives `1.010243055` node-hours for five jobs. A useful actual-charge
expectation is therefore roughly `0.9--1.0` node-hours, but the ledger must
reserve `15.000` until terminal `sacct` reconciliation. The new coordinates
may take a different number of MF updates, so only Perlmutter accounting is
authoritative.

With the last synchronized local active total of `42.024097222` node-hours and
the running `chi=400` campaign's `6.000`-node-hour unreconciled ceiling, the
illustrative total after grid submission is `63.024097222`, leaving
`336.975902778` under the 400-hour project cap. The plan command recomputes
this from the live Perlmutter ledgers and supersedes the illustration.

## Perlmutter handoff

Run these commands on Perlmutter from the managed checkout. Preparation does
not submit or reserve:

```bash
cd "$CFS/m4863/MPS-MFT/ladder_mps_mft"

bash slurm/phase1_gpu.sh plan-square-smooth-pairing-grid

GRID_RUN=20260903_phase1_square_grid_smooth_pairing_chi200_loose
bash slurm/phase1_gpu.sh prepare-square-smooth-pairing-grid "$GRID_RUN"
column -ts $'\t' "output/phase1_gpu/$GRID_RUN/manifest.tsv" | less -S
bash slurm/phase1_gpu.sh budget
```

The run ID is distinct from the active `chi=400` comparison. When ready, the
single guarded submission action creates only the five scientific jobs; there
is no smoke job:

```bash
bash slurm/phase1_gpu.sh submit "$GRID_RUN"
bash slurm/phase1_gpu.sh status "$GRID_RUN"
```

After all five allocation records are terminal, reconcile their unused
requested ceilings:

```bash
bash slurm/phase1_gpu.sh reconcile "$GRID_RUN"
bash slurm/phase1_gpu.sh budget
```
