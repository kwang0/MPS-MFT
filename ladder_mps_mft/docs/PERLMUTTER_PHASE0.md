# Perlmutter Phase 0 CPU calibration

No jobs were submitted during implementation.

The QSL Project B result that favored a small block-sparse CPU configuration is a useful hypothesis only. Ladder finite-DMRG has different contractions, memory growth, and sweep structure, so this workflow remeasures the optimum.

## Workflow

On Perlmutter, first make Julia and the project depot available, then inspect:

```bash
cd /path/to/MPS-MFT/ladder_mps_mft
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
bash slurm/phase0_calibrate_cpu.sh plan
```

Review the account, shared QOS, 32G candidate request, one-hour limit, three-node-hour cap, and generated 1.511718750-node-hour worst-case reservation. Submission is an explicit separate action:

```bash
bash slurm/phase0_calibrate_cpu.sh submit 20260821_phase0
bash slurm/phase0_calibrate_cpu.sh status 20260821_phase0
bash slurm/phase0_calibrate_cpu.sh show 20260821_phase0
```

The seed job creates one immutable, non-scientific chi=64 warm-start MPS. All 11 candidates read that exact state and run the same two-sweep payload three times. BLAS, Strided, and block-sparse threading are mutually exclusive. `/usr/bin/time -v` supplies MaxRSS.

The report rejects a candidate unless it completes and agrees with `serial-t1` within `1e-8` energy per physical site and `1e-6` density. Eligible candidates are ranked by projected shared-QOS node-hours per solve after adding 30% to MaxRSS, rounding memory upward to 2 GiB, and applying Perlmutter's CPU/memory charging rule.

Then submit one larger validation:

```bash
bash slurm/phase0_calibrate_cpu.sh validate 20260821_phase0
```

This uses the winner at chi=200 and six sweeps and schedules a follow-up report. Inspect its terminal state, log, MaxRSS, energy/density, and metric hash. Only then update a production config's Julia, Slurm CPU, and memory choices.

## What Phase 0 does not establish

It does not compare CPU with the legacy GPU implementation, demonstrate SCF convergence, validate observables, or establish a phase. A later CPU/GPU crossover test should use the same accepted checkpoint, chi, sweeps, tolerance, and charged node-hour accounting.

After every run, append job IDs, state, resource recommendation, metric hashes, actual charge from `sacct`, and the validation boundary to `docs/RUN_LOG.md`.
