# Provenance, lineage, and selection

Every state records the git commit/branch/dirty flag; Julia and package versions; model, numerical, and implementation SHA-256 fingerprints; configuration and E_p-registry hashes; threading state; Slurm identifiers; seed label; random seed; preparation; scan direction; and parent/restart paths and hashes.

The implementation fingerprint hashes Julia, TOML, CSV, and shell inputs below this subproject while excluding generated output. This retains exact-code provenance even when the surrounding repository has unrelated dirty files.

`parent_checkpoint` is a hash-checked continuation seed and may belong to a nearby model point. `resume_checkpoint` is a hash-checked same-model restart and must match the model fingerprint. The restart field stored in HDF5 is the exact next field that would have been applied, not an implicit substitution of the last measured field.

An optional `parent_orbit_phase` selects one phase from a full hash-checked
orbit parent. Its phase index is stored in provenance, its MPS is loaded from
`cycle_members/NNN/psi`, and its measured field is used as the next raw-map
input. The sibling phases remain unchanged in the immutable source artifact.

The HDF5 schema distinguishes:

- `process_completed`: the process reached a terminal numerical status;
- `accepted`: either all period-one gates passed or an explicitly allowed periodic orbit passed the unmixed all-phase gates;
- `completed`: an alias for accepted, retained for conservative plot selection;
- `status` and `fundamental_period`: the actual outcome.

Schema v3 additionally records `solution_kind`, raw-orbit validation, the update mode of every iteration, orbit-averaged energy, phase-energy spread, and density contrast. Every phase of an accepted orbit contains its fields, correlators, energy decomposition, chemical potential, density, and MPS.

Rolling `checkpoint_latest.h5` and `checkpoint_best.h5` may be replaced within one run directory. Final `state.h5`, detected-orbit artifacts, Phase 0 seed, and metric files refuse overwrite.

Production Phase 1 full artifacts are scratch-resident. Their CFS analysis
mirrors recursively omit all MPS objects and record the full path, full SHA-256,
full byte count, compact SHA-256, and omitted paths. Compact states remain valid
inputs to selection, plotting, variational comparison, and field-only
inheritance. They are explicitly non-restartable; parent and resume lineage
always resolves a hash-checked full artifact.

`select_completed_runs` recursively selects only final `state.h5` artifacts and, by default, accepts both gated fixed points and unmixed validated periodic solutions. `--include-incomplete` exposes other terminal states with `plot_style=hatched`. Diagnostics HDF5 files and rolling checkpoints are not accidentally treated as final runs.

The comparison tool uses the last-phase energy for a fixed point and the stored phase average for a periodic solution. A mixed-history recurrence cannot enter a ranking. Physical periods beyond two require an explicit `accepted_periods` configuration and a documented mapping of orbit phases to transverse sublattices.

For hysteresis scans, keep independent, forward, and reverse branches side by side. Preserve the forward metastable continuation as its own labeled branch; a lower-energy state does not erase its lineage.
