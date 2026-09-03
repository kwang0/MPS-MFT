# Frozen legacy-field energy diagnostic

This campaign evaluates the terminal fields from the legacy square-ladder file at
`L=64`, `U=8`, `V=0`, `t0=1.4`, `t_perp=0.1`, and target density `0.9375`.
It performs exactly one fresh `chi=200` Float64-CUDA DMRG at the saved legacy
chemical potential. It does not run the chemical-potential search, apply a
mean-field update, or enter an SCF loop.

The DMRG result is measured with the current code and written with:

- the current canonical variational functional and all double-counting terms;
- the target-density correction `mu * (N_target - N)`;
- the outgoing mean fields and one-step raw-map residual;
- density, spin/charge structure factors, entanglement diagnostics, discarded
  weight, and realized link dimension; and
- a comparison table containing the six accepted same-fingerprint reference
  states plus the frozen legacy diagnostic.

The six accepted states retain the formal variational ranking. The frozen row is
always stored with `accepted=false`, `solution_kind=diagnostic`, and
`selection_eligible=false`: one map evaluation cannot establish an SCF fixed
point. Its energy is therefore a conditional diagnostic under the explicit
assumption that the supplied legacy fields represent the intended legacy basin.

As a read-only preview, reconstructing the current functional from the legacy
file's saved Float32 effective eigenvalue and final correlations gives a
target-density-corrected energy of `-84.6163371905`. That is `0.0257361851`
below the lowest of the six current accepted energies (`-84.5906010053`), or
`2.01064e-4` per physical site. This preview is not the requested comparison:
it inherits the legacy DMRG state, precision, and effective-energy consistency.
The fresh one-shot DMRG is designed to test whether the difference survives a
current Float64 solve at exactly the frozen legacy fields.

The legacy file itself is read-only and its SHA-256 is recorded. The full new MPS
is written below Perlmutter scratch; the launcher automatically mirrors an
analysis-ready HDF5 copy without MPS tensors to CFS.

## Perlmutter commands

Run from the user-managed Perlmutter checkout after synchronizing the code and
making the legacy HDF5 file available at the path below:

```bash
cd "$CFS/m4863/MPS-MFT/ladder_mps_mft"

SOURCE_RUN=20260902_phase1_square_t014_v000_seed_chi200_loose_cuda130
LEGACY_H5="$CFS/m4863/MPS-MFT/stateless_data/results_L_64_U_8.0_V_0.0_t0_1.4_t_p_0.1_geometry_square_chi_200_density_0.9375_gpu.h5"
FROZEN_RUN=20260903_phase1_square_t014_v000_legacy_frozen_dmrg_chi200

bash slurm/phase1_gpu.sh plan-frozen-legacy
bash slurm/phase1_gpu.sh prepare-frozen-legacy \
  "$SOURCE_RUN" "$LEGACY_H5" "$FROZEN_RUN"
bash slurm/phase1_gpu.sh submit "$FROZEN_RUN"
bash slurm/phase1_gpu.sh status "$FROZEN_RUN"
```

Preparation performs no submission and makes no ledger reservation. Submission
requests one of four GPUs for three hours, reserving `0.75` node-hours. No smoke
job is created. Before submission, the plan and submit actions read the
authoritative Perlmutter ledger and enforce the 400-additional-node-hour cap.

After completion, `energy_comparison.tsv` and `run_summary.md` are mirrored with
the compact result. `state.h5` contains the full energy breakdown, fields,
correlations, DMRG sweep evidence, source hashes, and diagnostic classification.
