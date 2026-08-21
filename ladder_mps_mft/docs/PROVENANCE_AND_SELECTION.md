# Provenance, lineage, and selection

Every state records the git commit/branch/dirty flag; Julia and package versions; model, numerical, and implementation SHA-256 fingerprints; configuration and E_p-registry hashes; threading state; Slurm identifiers; seed label; random seed; preparation; scan direction; and parent/restart paths and hashes.

The implementation fingerprint hashes Julia, TOML, CSV, and shell inputs below this subproject while excluding generated output. This retains exact-code provenance even when the surrounding repository has unrelated dirty files.

`parent_checkpoint` is a hash-checked continuation seed and may belong to a nearby model point. `resume_checkpoint` is a hash-checked same-model restart and must match the model fingerprint. The restart field stored in HDF5 is the exact next field that would have been applied, not an implicit substitution of the last measured field.

The HDF5 schema distinguishes:

- `process_completed`: the process reached a terminal numerical status;
- `accepted`: all period-1 scientific gates passed;
- `completed`: an alias for accepted, retained for conservative plot selection;
- `status` and `fundamental_period`: the actual outcome.

Rolling `checkpoint_latest.h5` and `checkpoint_best.h5` may be replaced within one run directory. Final `state.h5`, detected-cycle artifacts, Phase 0 seed, and metric files refuse overwrite.

`select_completed_runs` recursively selects only final `state.h5` artifacts and, by default, only accepted fixed points. `--include-incomplete` exposes other terminal states with `plot_style=hatched`. Diagnostics HDF5 files and rolling checkpoints are not accidentally treated as final runs.

For hysteresis scans, keep independent, forward, and reverse branches side by side. Preserve the forward metastable continuation as its own labeled branch; a lower-energy state does not erase its lineage.
