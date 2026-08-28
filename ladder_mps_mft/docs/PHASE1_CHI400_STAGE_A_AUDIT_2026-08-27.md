# Phase 1 chi=400 recurrence Stage A audit

## Decision

Campaign `20260826_phase1_unfrustrated_pairing_recurrence_chi400` did not pass
the conditional Stage B gate. Both phase-parent branches reproduced a
pairing-bearing raw-map period-two recurrence, but neither passed the
variational-energy recurrence gate. The independent `pairing_s2` branch
reached the 12-hour limit after only nine raw-map records and is incomplete.

Do not prepare or submit the SDW/CDW Stage B controls from these results. The
smallest useful next calculation is one explicit continuation segment for
`unfrustrated__pairing_s2_chi400`, solely to finish or further resolve its
20-update raw probe. Do not continue either completed phase-parent candidate
at the same controls.

## Reproducible local audit

The branch-level compact verifiers passed all 14 synced artifacts:

```text
full bytes represented: 11,826,906,884
compact bytes:             163,304,236
full artifacts verified: false
```

The full scratch sources were not mounted on Windows. This establishes compact
hashes, sizes, stateless markers, full-artifact links, and absence of MPS
objects; it does not establish the current readability or hash equality of the
full Perlmutter files.

The local audit generated at `2026-08-27T18:23:51.504` UTC reported:

- accepted states: `0 / 3`;
- raw-map periodic candidates: `2`;
- mixer-dependent periodic candidates: `0`;
- Hamiltonian-identity failures: `0 / 3`;
- effective-energy failures: `0 / 3`; and
- stored MF-iteration time: `20.723` GPU-hours, or `5.181` one-of-four-GPU
  node-hours before scheduler and compilation overhead.

The reproduced files are in the ignored directory
`output/phase1_gpu/20260826_phase1_unfrustrated_pairing_recurrence_chi400/audit-win-stagea-20260827`.

## State-level result

| Branch | Outcome | Iterations | Period | E solution | Relative residual | dE/site | Cycle residual | max abs pairing |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| phase 001 parent | raw-map candidate | 21 | 2 | -103.854613005 | 9.585e-4 | 3.171e-5 | 1.892e-3 | 1.873e-2 |
| phase 002 parent | raw-map candidate | 21 | 2 | -103.852203154 | 9.781e-4 | 3.370e-5 | 1.929e-3 | 1.872e-2 |
| independent pairing s2 | time limit | 9 | 0 | NaN | 3.923e-1 | 4.061e-3 | not established | 1.104e-2 |

Both phase-parent branches pass density, Hamiltonian-identity, and effective-
energy consistency. Their configured variational recurrence tolerance is
`1e-7` per site, so their `dE/site` values miss that gate by factors of about
317 and 337. Their period-two phases remain separate and are not eligible for
energy ranking.

At chi=200 the corresponding v3 candidate had `dE/site=2.087e-5`, cycle
residual `1.225e-3`, and `max|alpha|=1.925e-2`. Raising chi to 400 preserved the
pairing-bearing recurrence but did not improve either recurrence metric. This
supports persistence of the raw-map behavior, not acceptance of a physical
orbit or a thermodynamic phase.

The independent seed initially reduced its relative residual from `1.000` to
`7.877e-3` by record 6, then grew to `2.612e-2`, `1.203e-1`, and `3.923e-1` at
records 7--9. It has not yet produced enough contiguous raw records to classify
its destination basin.

## Prepared next operator step and budget boundary

On Perlmutter, first confirm the live status and ledger:

```bash
RECURRENCE_RUN=20260826_phase1_unfrustrated_pairing_recurrence_chi400
bash slurm/phase1_gpu.sh status "$RECURRENCE_RUN"
bash slurm/phase1_gpu.sh budget
```

The exact next submission is only:

```bash
bash slurm/phase1_gpu.sh continue \
  "$RECURRENCE_RUN" unfrustrated__pairing_s2_chi400
```

`continue` prepares the hash-pinned segment-002 resume configuration from the
full scratch state and immediately submits that one job. It should therefore
be run only when submission is intended. The requested upper-bound charge is
`3.000` node-hours. From the synced authoritative ledger value `123.625`, it
would project to `126.625` reserved and `273.375` unreserved under the
400-additional-node-hour cap. No third segment is pre-authorized; inspect the
segment-002 result first. The unused allowance remains reserved for later
higher-bond-dimension and scaling work.

Stage B remains closed even if the independent continuation becomes accepted,
because neither phase-parent lineage currently supplies the other required
accepted pairing-bearing survivor.
