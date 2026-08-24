# Phase 1 GPU v2 scientific and numerical audit

## Decision

Run `20260823_phase1_gpu_v2` is useful screening and warm-start evidence, but it
does not contain a converged state that can enter a phase diagram or publication
energy comparison. All nine branch jobs completed at the scheduler level; zero
states passed the scientific acceptance gates.

The reproducible audit is generated with:

```bash
julia --project=. scripts/audit_phase1_campaign.jl \
  output/phase1_gpu/20260823_phase1_gpu_v2
```

It writes exact source paths and gates to `audit/states.tsv` and a compact
summary to `audit/report.md` without modifying any immutable state.

## Data-quality result

- Outcomes: three raw-map period-two candidates, five mixer-dependent
  period-two candidates, one stagnated branch, and zero accepted solutions.
- Every saved production MPS is Float32. The cause is the use of `CUDA.cu`,
  whose documented behavior eagerly converts floating-point arrays to Float32.
- The Hamiltonian-identity errors are `1.52e-5`--`2.77e-5` per site and the
  effective-eigenvalue errors are `1.14e-5`--`3.34e-5` per site. All nine fail
  the configured `1e-9` and `1e-6` gates, respectively. Loosening those gates
  would erase resolution needed for the close frustrated and symmetry-seeded
  comparisons; the correction is Float64, not post-hoc acceptance.
- The original control flow stopped raw candidates as soon as a loose recurrence
  was visible and stopped mixer-dependent recurrences before launching the raw
  probe they explicitly required. Frustrated pairing/CDW stopped after only
  nine raw updates while their residuals were growing. Those labels are
  transient candidates, not evidence for a physical period-two orbit.
- Stored MF-iteration time sums to `40.406` GPU-hours, equivalent to about
  `10.102` node-hours for one of four GPUs before compilation, scheduler, and
  non-iteration overhead. Exact charge still requires synchronized `sacct`
  TRES/elapsed rows. The conservative project ledger remains authoritative for
  the hard cap.

## Screening-level physics signals

The following are hypotheses for the Float64 recovery campaign, not phase
claims.

| Geometry | Screening signal | Candidate energy separation |
|---|---|---:|
| Cubic frustrated | All three seeds retain a sizeable pairing field, `max|alpha| = 0.0110`--`0.0112`, while the bulk spin-Hartree variation is only `1.1e-5`--`2.2e-4`. Pairing and CDW seeds end only `1.15%` apart in the full measured-field norm. | The mixer-dependent SDW-seeded candidate is lower than CDW/pairing by `0.0091`/`0.0100` total (`7.1e-5`/`7.8e-5` per site), but this is only a few times the Float32 consistency floor and is not a ranking. |
| Cubic unfrustrated | Pairing retains a distinct `max|alpha|=0.0193` basin. SDW/CDW seeds have `max|alpha|<8e-6` and strong spin modulation, so the paired state is a plausible metastable coexistence branch rather than the screened ground state. | The paired candidate is higher by `0.5759` total (`4.50e-3` per site). SDW and CDW candidates differ by only `0.000716` total (`5.6e-6` per site), below the numerical floor. |
| Square | All seeds end with `max|alpha|<5e-6`; even the pairing seed flows to a non-paired magnetic/charge-modulated state. | Pairing and SDW seeds are unresolved (`0.00126` total), while the CDW-seeded candidate is higher than pairing by `0.0591` total (`4.62e-4` per site). |

A preliminary central-half Fourier transform of the one-point densities uses
`q_x = 2*pi*k/N_bulk`, `q_y in {0,pi}`, and amplitude
`|sum exp(-i q_x r - i q_y leg) O(r,leg)|/(2 N_bulk)`, after subtracting the
bulk mean. Several square/unfrustrated branches place charge weight near
`|q_x|=pi/8` and spin weight near `q_x=pi +/- pi/16`, `q_y=pi`, the familiar
finite-doping stripe relation at hole density `delta=1/16`. The precise peak is
seed-dependent and grid-locked, and these are one-point profiles under
self-consistent fields rather than connected structure factors. Publishable
stripe evidence therefore requires accepted states, full correlation-based
`S_charge(q)`/`S_spin(q)`, larger `L`, multiple bulk windows, and chi scaling.

## Implemented recovery

1. `runtime.tensor_scalar_type` is explicit and part of the numerical
   fingerprint and provenance. Phase 1 requires Float64.
2. MPS/MPO tensors are promoted tensor-by-tensor and transferred with
   NDTensors' type-preserving CUDA adaptor. Float32 v2 parents are promoted on
   load.
3. The smoke artifact records its requested scalar type and the launcher reads
   the saved MPS storage to require actual Float64 before matrix submission.
4. An early raw recurrence no longer shortens the 20-update probe. An
   unaccepted initial recurrence may proceed to Anderson only after the full
   probe. A mixer-dependent recurrence automatically triggers one fresh raw-map
   probe; a failed controlled probe stops for inspection rather than being
   damped away.
5. `prepare-recovery SOURCE_RUN NEW_RUN` creates a new immutable campaign whose
   manifest hashes and records every v2 parent without submitting a job;
   `submit NEW_RUN` then allocates only the gated smoke. The one-command
   `submit-recovery` alias remains available. Neither path mutates or falsely
   continues the old numerical fingerprint.
6. Schema-v5 states retain both the applied and measured full MF fields at
   every iteration. A SHA-guarded `inherit_from` mode also restores the legacy
   field-only warm start without importing the legacy MPS.

## Next allocation step

On Perlmutter, push/pull the correction and run:

```bash
SOURCE_RUN_ID=20260823_phase1_gpu_v2
RUN_ID=20260824_phase1_gpu_v3_float64
bash slurm/phase1_gpu.sh prepare-recovery "$SOURCE_RUN_ID" "$RUN_ID"
bash slurm/phase1_gpu.sh submit "$RUN_ID"
bash slurm/phase1_gpu.sh status "$RUN_ID"
grep -E 'gpu_smoke_path|linalg_preflight_dimension|tensor_scalar_type|system path' \
  "output/phase1_gpu/$RUN_ID"/logs/smoke-*.out
```

The smoke reserves `0.125` node-hours. Only after it reports dimension `256`,
Float64, and no system-path warning, submit the nine warm-start branches:

```bash
bash slurm/phase1_gpu.sh submit-matrix "$RUN_ID"
```

That adds `27.0` conservative node-hours. From the current ledger value
`54.25`, the total becomes `81.375`, leaving `318.625` under the project's
400-additional-node-hour cap. Phase selection begins only after accepted
Float64 states exist; compare energies within a geometry and common numerical
fingerprint, then run correlation diagnostics and `L`/chi scaling on the
surviving branches.
