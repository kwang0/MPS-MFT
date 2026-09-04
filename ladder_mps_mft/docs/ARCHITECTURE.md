# Architecture and workflow map

This is a stable orientation map. Detailed scientific rules are owned by the
linked method documents and should not be redefined here.

## End-to-end flow

```text
versioned config + E_p registry + seed/lineage contract
                         |
                         v
             preparation script + manifest
                         |
                         v
          user-run Slurm launcher on Perlmutter
                         |
                         v
       SCF driver -> density-fixed DMRG -> MF raw map
                         |
           +-------------+-------------+
           |                           |
           v                           v
 full immutable state on scratch   compact stateless mirror on CFS
 (MPS, restart/checkpoints)         (fields, histories, metrics, hashes)
           |                           |
           +-------------+-------------+
                         v
       verifier -> classifier -> accepted-only selection
                         |
                         v
              plots, audits, and reports
```

The user operates transfers and Perlmutter. Codex prepares and validates local
code and analyzes synchronized evidence; it does not connect to NERSC or
operate Slurm.

## Source ownership

| Area | Responsibility |
|---|---|
| `src/Types.jl`, `src/Config.jl` | Versioned model, numerical, output, and seed contracts |
| `src/Geometry.jl`, `src/EpRegistry.jl` | Ladder geometry and exact/interpolated pair-binding provenance |
| `src/MeanField.jl`, `src/Mixing.jl` | MF fields, seed construction, raw-map updates, and fixed-point acceleration |
| `src/Solver.jl`, `src/Device.jl` | Density-fixed DMRG/SCF execution and CPU/CUDA representation |
| `src/Convergence.jl` | Fixed-point, oscillation, slow-mode, and recurrence classification |
| `src/Variational.jl`, `src/Selection.jl` | Canonical functional and accepted-only same-fingerprint ranking |
| `src/Provenance.jl`, `src/Storage.jl` | Fingerprints, immutable HDF5, histories, hashes, and full/compact lineage |
| `src/Diagnostics.jl` | Correlations, structure factors, entanglement, and diagnostic observables |
| `scripts/` | Reproducible preparation, execution, verification, audit, analysis, and plotting entry points |
| `configs/` | Versioned base numerical and campaign templates |
| `slurm/` | User-run Perlmutter planning, submission, continuation, status, and accounting wrappers |
| `output/` | Ignored local or synchronized campaign evidence; never Git authority |

## State and storage boundaries

- A full state contains restartable MPS data and lives scratch-first. Scratch is
  not archival storage.
- A stateless mirror removes the MPS while retaining analysis fields,
  histories, diagnostics, full-source path, size, and SHA-256. It is not a
  restart source.
- `state.h5` is immutable. Repairs or transformations create a new artifact
  with explicit provenance.
- Time-zero seeds, every applied/measured MF record, and physical orbit phases
  remain separate in the history.

See `docs/PERLMUTTER_STORAGE.md` and
`docs/PROVENANCE_AND_SELECTION.md`.

## Physics and numerical boundaries

- The unmixed raw MF map determines fixed points and physical periodic orbits.
  Anderson mixing may accelerate a fixed point but cannot establish, average,
  or erase raw-map physics.
- Acceptance includes density, field, oscillation, slow-mode extrapolation,
  and energy-stability gates.
- Energies are compared through the canonical zero-temperature variational
  functional including double-counting terms, and only after acceptance and
  matching provenance gates.
- Different transverse geometries and different `(t0,V)` points are different
  Hamiltonians, not branches in one authorized energy ranking.

See `docs/CONVERGENCE.md`, `docs/VARIATIONAL_FUNCTIONAL.md`, and
`docs/PHASE1_NUMERICAL_ERROR_BUDGET.md`.

## Continuity ownership

- `AGENTS.md`: durable rules.
- `docs/PROJECT_STATE.md`: current snapshot and exact next action.
- `docs/plans/ACTIVE.md`: active completion sequence.
- Dated campaign docs: frozen question and numerical contract.
- `docs/RUN_LOG.md`: append-only evidence and decision history.
- Git commits: implementation history and reviewable changes.

