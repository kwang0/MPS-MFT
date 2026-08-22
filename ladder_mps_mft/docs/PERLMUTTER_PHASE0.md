# Perlmutter Phase 0 CPU calibration

Phase 0 now has one narrow purpose: choose the CPU configuration for the
production-scale finite-DMRG solve. The complete v2 matrix is sufficient to
shortlist `serial-t1` and `blocksparse-t4`. The failed v3 density search is
preserved as diagnostic evidence but is no longer on the critical path.

The focused script v1.3.0 compares those two candidates at `L=64`, `chi=200`,
six sweeps, and fixed `mu=1.8`. It does not time or modify the legacy density
search.

## What is timed

Each candidate performs two identical solves from one immutable warm-start MPS.
The timed region is exactly the call to `run_dmrg_ground`. The following are
outside the timer:

- Julia/ITensor compilation warmup;
- construction of the mean-field MPO;
- copying the common initial MPS;
- explicit garbage collection;
- post-solve density measurement; and
- any chemical-potential search.

The seed job performs one untimed two-sweep, at-most-chi=64 DMRG at the same
fixed chemical potential. It exists only to give both candidates the same
realistic initial state. The benchmark result is a timing/resource result, not
a density-convergence or scientific-state claim.

## Submission

After pushing this branch and pulling it on Perlmutter:

```bash
cd /global/cfs/cdirs/m4863/MPS-MFT/ladder_mps_mft
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

bash slurm/phase0_calibrate_cpu.sh plan
bash slurm/phase0_calibrate_cpu.sh submit-seed 20260822_phase0_cpu_v4
bash slurm/phase0_calibrate_cpu.sh status 20260822_phase0_cpu_v4
bash slurm/phase0_calibrate_cpu.sh show-seed 20260822_phase0_cpu_v4
```

Only after the seed is `COMPLETED`, submit the two candidates:

```bash
bash slurm/phase0_calibrate_cpu.sh submit-matrix 20260822_phase0_cpu_v4
bash slurm/phase0_calibrate_cpu.sh status 20260822_phase0_cpu_v4
bash slurm/phase0_calibrate_cpu.sh show 20260822_phase0_cpu_v4
```

The one-shot `submit RUN_ID` action remains available, but the staged path is
preferred because it exposes a seed failure before allocating the matrix.

The default candidate request is 32 GiB and four hours. The guarded worst-case
reservation for seed, both candidate jobs, and report is `0.570312500` CPU
node-hours, below the three-node-hour cap. Actual charge depends on elapsed
time, while the report uses measured MaxRSS plus 30%, rounded to 2 GiB, to
recommend a smaller production memory request.

The report requires:

- exact agreement of seed, config, model, code, and E_p provenance;
- the requested exclusive thread topology;
- one and only one fixed-mu DMRG solve in every timed repetition;
- energy agreement with `serial-t1` within `1e-8` per physical site;
- density agreement within `1e-6`;
- measured MaxRSS; and
- a repeat timing range no larger than 10% of the median.

Eligible candidates are ranked by projected shared-QOS CPU node-hours per DMRG
solve, not wall time alone.

## Legacy GPU estimate

There is no synchronized matched `chi=200` GPU timing plus `sacct` record, so
the current comparison is deliberately labeled an estimate. Extrapolating the
saved legacy `chi=500` and `chi=1000` sweep logs gives approximately 35--60 s
for a six-sweep, `chi=200` fixed-mu solve. Under [NERSC's shared-QOS charging
rule](https://docs.nersc.gov/jobs/policy/#calculating-charges), one GPU is one
quarter of a Perlmutter GPU node, so that is about `0.00243--0.00417` GPU
node-hours.

Scaling the v2 `chi=64`, two-sweep CPU measurements with a `chi^2.5--chi^3`
cost model gives these pre-run ranges:

| Backend | Estimated wall time per solve | 32 GiB request | Projected 18 GiB request |
|---|---:|---:|---:|
| `blocksparse-t4` | 44--78 min | 0.052--0.092 CPU node-hours | 0.029--0.051 CPU node-hours |
| `serial-t1` | 62--109 min | 0.073--0.128 CPU node-hours | 0.040--0.071 CPU node-hours |

Thus the CPU path is provisionally about 44--190 times slower in wall time.
The raw node-hour numbers are approximately 7--53 times larger, depending on
backend and memory request. That is not a monetary or allocation-equivalent
ratio: [NERSC maintains separate CPU and GPU allocation
pools](https://docs.nersc.gov/jobs/policy/#charge-factors).

This is also not a pure hardware comparison. The CPU implementation conserves
total `S_z` and fermion-number parity; the legacy GPU implementation disables
both quantum numbers. A definitive crossover comparison requires a matched GPU
job using the same checkpoint, Hamiltonian, sweeps, convergence rule, timed
region, and `sacct` record.

For scale only, the legacy launcher requests one GPU for up to 48 hours in the
shared QOS. A segment that consumes all 48 hours costs 12 GPU node-hours. A
full legacy SCF run may contain many DMRG solves and resubmitted segments, so
that upper-bound segment cost must not be compared with the single-solve Phase
0 number.

After the run, append job IDs, terminal states, hashes, measured MaxRSS,
recommendation, and actual `sacct` charge to `docs/RUN_LOG.md`.
