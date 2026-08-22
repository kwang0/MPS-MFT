# Perlmutter Phase 0 CPU calibration

The first successful matrix, `20260821_phase0_cpu_v2`, completed but exposed a
workload-definition error: its fixed-`mu=0` solves reached `n=0.5614` instead
of the configured `n=0.9375`. Preserve that run as audit evidence, do not
validate or promote its recommendation, and use script v1.2.0 with a new run
ID. See `PHASE0_V2_AUDIT.md`.

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
bash slurm/phase0_calibrate_cpu.sh submit 20260822_phase0_cpu_v3
bash slurm/phase0_calibrate_cpu.sh status 20260822_phase0_cpu_v3
bash slurm/phase0_calibrate_cpu.sh show 20260822_phase0_cpu_v3
```

The seed job creates one immutable, density-targeted, non-scientific chi=64
warm-start MPS. All 11 candidates read that exact state. Each repetition starts
from the same seed and runs `find_mu_for_density`, whose individual DMRG
evaluations use two sweeps. This mirrors the operation used inside the SCF
solver and prevents a fixed-chemical-potential solve from benchmarking the
wrong density. BLAS, Strided, and block-sparse threading are mutually
exclusive. `/usr/bin/time -v` supplies MaxRSS.

The report rejects a candidate unless every repetition completes within
`5e-4` of the target density, follows the same density-search path, and agrees
with `serial-t1` within `1e-8` energy per physical site, `1e-6` density, and
`1e-8` chemical potential. It also enforces metric schema, exact seed/config/
implementation provenance, exclusive thread topology, and measured MaxRSS.
Candidates whose three timing repeats span more than 10% of their median are
marked unstable rather than ranked.
Eligible candidates are ranked by projected shared-QOS node-hours per complete
density-targeting call after adding 30% to MaxRSS, rounding memory upward to 2
GiB, and applying Perlmutter's CPU/memory charging rule.

Then submit one larger validation:

```bash
bash slurm/phase0_calibrate_cpu.sh validate 20260822_phase0_cpu_v3
```

This uses the winner at chi=200 and six sweeps per density-search evaluation and
schedules a follow-up report. The validation may converge at a different
chemical potential from chi=64; acceptance instead requires its own successful
density target, the selected exclusive topology, matched provenance, and
MaxRSS. Inspect its terminal state, log, metric, and report acceptance. Only
then update a production config's Julia, Slurm CPU, and memory choices.

## What Phase 0 does not establish

It does not compare CPU with the legacy GPU implementation, demonstrate SCF convergence, validate observables, or establish a phase. A later CPU/GPU crossover test should use the same accepted checkpoint, chi, sweeps, tolerance, and charged node-hour accounting.

After every run, append job IDs, state, resource recommendation, metric hashes, actual charge from `sacct`, and the validation boundary to `docs/RUN_LOG.md`.
