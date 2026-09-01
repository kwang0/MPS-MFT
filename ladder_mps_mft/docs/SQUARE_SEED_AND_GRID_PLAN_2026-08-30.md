# Square seed reconnaissance and grid plan

Status date: 2026-08-30. This is an exploratory Phase 1 plan, not a phase
assignment or a production convergence claim.

## New legacy evidence

Two independently initialized legacy square-ladder calculations are present
locally at `L=64`, `U=8`, `t0=1.4`, `t_perp=0.1`, density `0.9375`, and
`chi=200`:

| V | SHA-256 | outer updates | final density error | stored d-wave proxy | correlation-based dominant pairing |
|---:|---|---:|---:|---:|---:|
| -0.2 | `761a14d5248507abc4bc7092f3960302dc915866ce5c602c77bfe787a3c05be` | 4 | `1.143e-3` | `0.07952` | `0.08396` |
| -0.4 | `3100916863023c01ead6ae3edd77beda60a7a2c1e2b9991a2d1dad27ee7b75b0` | 3 | `1.226e-3` | `-0.08650` | `0.08853` |

Neither file declares `inherit_from`, a parent checkpoint, or a resume
checkpoint. In contrast, the earlier square points were field-inherited from
the `t0=1,V=0` solution. The appearance of a substantial d-wave proxy in both
new independent starts is therefore evidence that basin access depends on the
initial condition. It is not evidence that either file is a converged
refactored solution: the legacy files have only three or four MF updates,
their density errors are slightly above `1e-3`, and their stored energies are
not the canonical refactored functional with its double-counting terms.

The subsequently supplied clean `V=0,t0=1.4` profile shows that treating SDW
and CDW as unrelated sources is physically artificial. Its spin texture is an
antiferromagnetic carrier with a slow signed envelope, while its charge minima
sit at the spin antiphase nodes. On the 64-rung open grid this is represented
by envelope mode `m=4`, physical spin mode `n_s=63-m=59`, and charge second
harmonic `n_c=2m=8`. The adjacent `m=5` member (`n_s=58,n_c=10`) is declared
before any new energy is inspected to test nearby finite-size wavevector
competition.

The stripe source uses odd transverse parity for spin and even parity for
charge. Spin and charge templates are normalized separately, combined at a
declared source-norm ratio `charge:spin=0.2`, and then included in the common
full-field normalization. The ratio is a coarse texture-inspired source
choice, not a fit equating mean-field source magnitude to observable density
amplitude.

Direct inspection of the actual fresh-run block in the legacy driver corrects
an earlier recollection about its seed. It initialized `beta=0` and
`mu_cdw=0`. For `alpha`, it drew one Gaussian coefficient for each relative
rung offset and leg-pair class and copied that coefficient along every allowed
rung. The legacy source was therefore random in relative pairing form factor
but already translation-invariant—and hence maximally smoothed—in the
center-of-mass direction. The representative bank includes a separately
labeled `legacy_pairing` control with that spatial structure, a deterministic
field RNG stream keyed by seed `1404`, and the same matched total norm as the
other starts. It does not reproduce the legacy amplitude convention.

## Representative calculation now staged in code

The next prepared calculation is the square point `t0=1.4,V=-0.4`. It uses
six independent starts, no inherited fields, and no parent or resumed MPS:

- a uniform d-wave pairing symmetry-subspace control;
- a legacy-like translation-invariant random relative-bond pairing control;
- normal combined SDW/CDW stripe controls at envelope modes `m=4` and `m=5`;
- combined stripe plus uniform d-wave starts at `m=4` and `m=5`.

The last two starts are essential: an exactly zero anomalous field leaves a
number-conserving Hamiltonian and can lock a normal stripe branch into the
`alpha=0` invariant subspace. The mixed starts allow pairing either to survive
or decay and therefore test coexistence without forcing it. Conversely, the
pure pairing and pure normal-stripe branches remain useful symmetry-subspace
controls, but are not by themselves unrestricted basin searches.

All six use matched total field norm `1e-3`, phase `0`, and common
product-state random seed `1404`. In mixed starts the separately normalized
pairing and spin source norms have ratio `1`; the pairing form factor is
uniform d-wave. The numerical settings are intentionally exploratory:

| control | exploratory value |
|---|---:|
| bond dimension | `chi=200` |
| DMRG sweeps | `12` |
| DMRG energy tolerance | `1e-6` |
| truncation cutoff | `1e-10` |
| inner density tolerance | `1e-3` |
| outer density tolerance | `1e-3` |
| initial chemical-potential bracket step | `0.01` |
| bracket growth | `3` |
| variational-energy stabilization tolerance | `1e-6` per site |
| maximum MF updates | `80` |

The common initial chemical potential is `0.55`, close to the independent
legacy `V=-0.4` result (`0.54634`) but without importing any legacy field or
MPS. The exact registry entry is used without interpolation:
`E_p=-0.24962435880865996`.

The first 20 updates apply the unmixed physical map. An accepted period-one
or all-phase period-two solution may terminate there. If the raw probe ends
without acceptance, or archives an unaccepted raw recurrence, the solver may
then use Anderson acceleration. A later Anderson fixed point is a numerical
fixed-point result; it does not erase or reclassify an archived physical
raw-map orbit. A mixer-dependent recurrence receives a fresh raw-map probe
before it can be retained as a candidate.

The config is `configs/phase1_gpu_square_seed_pilot_chi200_loose.toml`; the
preparer is `scripts/prepare_phase1_square_seed_pilot.jl`. Preparation writes
full-result destinations below Perlmutter scratch and MPS-free stateless
destinations below CFS. It refuses parameter drift, interpolation, lineage,
or overwriting a prepared run.

## Interpretation and energy boundary

The six branches share one square geometry, model fingerprint, numerical
fingerprint, implementation fingerprint, and exact `E_p` selection. If more
than one branch is accepted, their canonical variational energies may be
compared within this campaign as a preliminary basin ranking. The loose
tolerances and `chi=200` make that ranking reconnaissance only.

If the `m=4` and `m=5` mixed starts converge to the same accepted state, that
is evidence of basin robustness within this predeclared two-mode bank. If they
remain distinct and accepted, retain and rank both rather than selecting one
by appearance. Phase zero is boundary-compatible with the supplied profile;
additional phase offsets are conditional controls if the two modes remain
nearly degenerate or retain different wall counts, not part of this first
six-branch allocation.

Do not rank a new branch against either legacy energy, a different grid point,
another transverse geometry, or a state with a different numerical or
implementation fingerprint. Any scientifically retained basin must be rerun
or continued into a common tighter fingerprint before a production ranking.

## Compute ledger

The synced project ledger contains `135.750` conservatively reserved
node-hours and `264.250` unreserved under the 400-additional-node-hour cap.
Perlmutter accounting remains authoritative and must be rechecked immediately
before submission.

Six first segments request `6*3=18.000` node-hours. If authorized under the
direct-submission policy, the ledger would move to `153.750` reserved and
`246.250` unreserved. The four-segment emergency ceiling is `72.000`; it is not
pre-authorized. Finalized early completions may release unused requested
ceilings only through the append-only `sacct` reconciliation action.

The intended square grid is `t0={1.0,1.2,1.4}` by `V={0,-0.2,-0.4}`. Running
this full six-branch reconnaissance at the representative point and a
provisional three-branch bank at each of the other eight points would reserve
`18.000 + 8*9.000 = 90.000` first-segment node-hours. That would project the
current ledger to `225.750` reserved and leave `174.250`. Repeating all six
branches at all nine points would instead reserve `162.000`, project the
ledger to `297.750`, and leave only `102.250`; that is explicitly not
recommended. The later three-branch bank is
not yet locked or authorized: use this representative result to prune it while
retaining compute for higher bond dimension, length scaling, and any later
cubic-unfrustrated work.

All nine grid coordinates already have exact chi-1000 `E_p` registry rows, so
the later grid does not require interpolation:

| V / t0 | 1.0 | 1.2 | 1.4 |
|---:|---:|---:|---:|
| 0.0 | `-0.13251724` | `-0.17989619749147323` | `-0.14653773091916378` |
| -0.2 | `-0.1545120066237189` | `-0.21453418655934797` | `-0.2068002629740704` |
| -0.4 | `-0.17882744409052975` | `-0.25124588461187614` | `-0.24962435880865996` |

## Accuracy ladder

1. Use this loose `chi=200` point to determine whether the two stripe harmonics
   reach distinct basins, whether pairing survives the unrestricted mixed
   starts, and whether the density-search changes produce enough MF updates.
2. Warm-start only accepted or clearly contracting relevant basins into a
   separately prepared, SHA-pinned campaign with one common tighter `chi=200`
   fingerprint (at least density `1e-4`, cutoff `1e-11`, and a tighter DMRG
   energy tolerance). Do not mix those energies with the loose fingerprint.
3. Recheck the surviving competitors at `chi=400`, then reserve `chi=800` and
   length controls for the small set that can affect the conclusion.
4. Expand to the remaining square grid points conditionally. Treat a later
   cubic-unfrustrated grid as a separate transverse geometry and never compare
   its absolute energies to square states.

## Perlmutter staging commands

Run the plan first against the live ledger. Preparation itself does not submit
or reserve:

```bash
cd "$CFS/m4863/MPS-MFT/ladder_mps_mft"
bash slurm/phase1_gpu.sh plan-square-seed-pilot

SQUARE_RUN=20260830_phase1_square_t014_vm04_seed_chi200_loose
bash slurm/phase1_gpu.sh prepare-square-seed-pilot "$SQUARE_RUN"
sed -n '1,7p' "output/phase1_gpu/$SQUARE_RUN/manifest.tsv"
bash slurm/phase1_gpu.sh budget
```

Only after reviewing those outputs should the guarded direct action be used:

```bash
bash slurm/phase1_gpu.sh submit "$SQUARE_RUN"
bash slurm/phase1_gpu.sh status "$SQUARE_RUN"
```
