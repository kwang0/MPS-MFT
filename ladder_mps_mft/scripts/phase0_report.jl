#!/usr/bin/env julia

using Printf
using TOML

length(ARGS) == 1 || error("usage: julia --project=. scripts/phase0_report.jl RUN_DIRECTORY")
run_directory = abspath(first(ARGS))
candidates_path = joinpath(run_directory, "candidates.tsv")
isfile(candidates_path) || error("missing candidate table: $candidates_path")

const MIB_PER_LOGICAL_CPU = 1952
const PHYSICAL_CORES_PER_NODE = 128
const MEMORY_MARGIN = 1.30
const MINIMUM_MEMORY_MIB = 4096
const MEMORY_ROUND_MIB = 2048

ceil_div(left::Integer, right::Integer) = div(left + right - 1, right)

function maximum_rss_kib(path)
    isfile(path) || return 0
    for line in eachline(path)
        occursin("Maximum resident set size (kbytes)", line) || continue
        return parse(Int, strip(split(line, ':'; limit=2)[2]))
    end
    return 0
end

function recommended_memory_mib(rss_kib)
    rss_mib = ceil_div(rss_kib, 1024)
    padded = ceil(Int, MEMORY_MARGIN * rss_mib)
    return max(MINIMUM_MEMORY_MIB, ceil_div(padded, MEMORY_ROUND_MIB) * MEMORY_ROUND_MIB)
end

function charged_physical_cores(logical_cpus, memory_mib)
    memory_logical = ceil_div(memory_mib, MIB_PER_LOGICAL_CPU)
    return ceil_div(max(logical_cpus, memory_logical), 2)
end

function topology_matches(metric, candidate)
    String(metric["backend"]) == candidate.backend || return false
    Int(metric["julia_threads"]) == candidate.threads || return false
    blas = Int(metric["blas_threads"])
    strided = Int(metric["strided_threads"])
    blocksparse = Bool(metric["threaded_blocksparse"])
    return if candidate.backend == "serial"
        blas == 1 && strided == 1 && !blocksparse
    elseif candidate.backend == "blocksparse"
        blas == 1 && strided == 1 && blocksparse
    elseif candidate.backend == "strided"
        blas == 1 && strided == candidate.threads && !blocksparse
    else
        blas == candidate.threads && strided == 1 && !blocksparse
    end
end

function fixed_mu_contract(metric)
    repetitions = Int(metric["repetitions"])
    solves = Int.(metric["dmrg_solves"])
    return String(metric["benchmark_kind"]) == "fixed_mu_dmrg" &&
        String(metric["timed_region"]) == "run_dmrg_ground_only" &&
        length(solves) == repetitions && all(==(1), solves) &&
        !Bool(metric["mpo_construction_timed"]) &&
        !Bool(metric["initial_mps_copy_timed"]) &&
        !Bool(metric["garbage_collection_timed"]) &&
        !Bool(metric["compile_warmup_timed"])
end

candidates = NamedTuple[]
for (index, line) in enumerate(eachline(candidates_path))
    index == 1 && continue
    isempty(strip(line)) && continue
    label, threads, backend, logical = split(line, '\t')
    push!(candidates, (; label, threads=parse(Int, threads), backend, logical=parse(Int, logical)))
end

baseline_candidate_index = findfirst(row -> row.label == "serial-t1", candidates)
baseline_candidate_index === nothing && error("candidate matrix must include serial-t1")
baseline_candidate = candidates[baseline_candidate_index]
baseline_path = joinpath(run_directory, "metrics", "serial-t1.toml")
isfile(baseline_path) || error("serial baseline did not produce $baseline_path")
baseline = TOML.parsefile(baseline_path)
Int(get(baseline, "schema_version", 0)) >= 4 || error(
    "serial baseline predates fixed-mu DMRG metric schema v4; rerun Phase 0",
)
String(baseline["status"]) == "complete" || error("serial baseline did not complete")
fixed_mu_contract(baseline) || error("serial baseline timed something other than one fixed-mu DMRG solve")
topology_matches(baseline, baseline_candidate) || error("serial baseline topology is inconsistent")

baseline_energy = Float64(baseline["representative_energy"])
baseline_density = Float64(baseline["representative_density"])
target_density = Float64(baseline["target_density"])
benchmark_mu = Float64(baseline["benchmark_chemical_potential"])
seed_mu = Float64(baseline["seed_chemical_potential"])
isapprox(seed_mu, benchmark_mu; atol=1e-12, rtol=0.0) || error(
    "serial baseline seed and benchmark chemical potentials differ",
)
baseline_seed_sha256 = String(baseline["seed_sha256"])
baseline_seed_config_sha256 = String(baseline["seed_config_sha256"])
baseline_config_sha256 = String(baseline["config_sha256"])
baseline_seed_config_sha256 == baseline_config_sha256 || error(
    "serial timing seed was not created from the timing config",
)
baseline_ep_sha256 = String(baseline["ep_source_sha256"])
baseline_model_fingerprint = String(baseline["model_fingerprint"])
baseline_implementation_sha256 = String(baseline["implementation_sha256"])
baseline_git_commit = String(baseline["git_commit"])
physical_sites = Int(baseline["physical_sites"])

energy_tolerance_per_site = parse(Float64, get(ENV, "PHASE0_ENERGY_TOL_PER_SITE", "1e-8"))
density_tolerance = parse(Float64, get(ENV, "PHASE0_DENSITY_TOL", "1e-6"))
mu_tolerance = parse(Float64, get(ENV, "PHASE0_MU_TOL", "1e-12"))
timing_range_tolerance = parse(Float64, get(ENV, "PHASE0_TIMING_RELATIVE_RANGE_MAX", "0.10"))
legacy_gpu_seconds_low = parse(Float64, get(ENV, "PHASE0_LEGACY_GPU_SECONDS_LOW", "35"))
legacy_gpu_seconds_high = parse(Float64, get(ENV, "PHASE0_LEGACY_GPU_SECONDS_HIGH", "60"))
0 < legacy_gpu_seconds_low <= legacy_gpu_seconds_high || error(
    "legacy GPU timing estimate must be a positive ordered range",
)

Float64(baseline["energy_spread"]) / physical_sites <= energy_tolerance_per_site || error(
    "serial baseline repeated inconsistent energies",
)
Float64(baseline["density_spread"]) <= density_tolerance || error(
    "serial baseline repeated inconsistent densities",
)

rows = NamedTuple[]
for candidate in candidates
    metric_path = joinpath(run_directory, "metrics", "$(candidate.label).toml")
    time_path = joinpath(run_directory, "metrics", "$(candidate.label).time")
    if !isfile(metric_path)
        push!(rows, merge(candidate, (; status="missing", valid=false, reason="missing metric",
            seconds=NaN, energy=NaN, density=NaN, energy_delta_per_site=NaN,
            density_delta=NaN, timing_relative_range=NaN, rss_gib=NaN, memory_gib=NaN,
            physical_cores=0, node_hours=Inf)))
        continue
    end
    metric = TOML.parsefile(metric_path)
    if Int(get(metric, "schema_version", 0)) < 4
        push!(rows, merge(candidate, (; status="legacy", valid=false, reason="legacy metric schema",
            seconds=NaN, energy=NaN, density=NaN, energy_delta_per_site=NaN,
            density_delta=NaN, timing_relative_range=NaN, rss_gib=NaN, memory_gib=NaN,
            physical_cores=0, node_hours=Inf)))
        continue
    end

    status = String(metric["status"])
    seconds = Float64(metric["median_seconds"])
    repeated_seconds = Float64.(metric["seconds"])
    timing_relative_range = (maximum(repeated_seconds) - minimum(repeated_seconds)) / seconds
    energy = Float64(metric["representative_energy"])
    density = Float64(metric["representative_density"])
    energy_delta = abs(energy - baseline_energy) / physical_sites
    density_delta = abs(density - baseline_density)
    energy_spread = Float64(metric["energy_spread"]) / physical_sites
    density_spread = Float64(metric["density_spread"])
    candidate_mu = Float64(metric["benchmark_chemical_potential"])

    workload_match = fixed_mu_contract(metric) &&
        isapprox(Float64(metric["target_density"]), target_density; atol=1e-12, rtol=0.0) &&
        isapprox(candidate_mu, benchmark_mu; atol=mu_tolerance, rtol=0.0) &&
        isapprox(Float64(metric["seed_chemical_potential"]), seed_mu; atol=mu_tolerance, rtol=0.0)
    provenance_match = String(metric["seed_sha256"]) == baseline_seed_sha256 &&
        String(metric["config_sha256"]) == baseline_config_sha256 &&
        String(metric["seed_config_sha256"]) == baseline_seed_config_sha256 &&
        String(metric["ep_source_sha256"]) == baseline_ep_sha256 &&
        String(metric["model_fingerprint"]) == baseline_model_fingerprint &&
        String(metric["implementation_sha256"]) == baseline_implementation_sha256 &&
        String(metric["git_commit"]) == baseline_git_commit
    topology_match = topology_matches(metric, candidate)
    rss_kib = maximum_rss_kib(time_path)
    memory_mib = rss_kib > 0 ? recommended_memory_mib(rss_kib) : 0
    physical_cores = memory_mib > 0 ? charged_physical_cores(candidate.logical, memory_mib) : 0
    node_hours = physical_cores > 0 ? seconds / 3600 * physical_cores / PHYSICAL_CORES_PER_NODE : Inf

    valid = status == "complete" && energy_delta <= energy_tolerance_per_site &&
        density_delta <= density_tolerance && energy_spread <= energy_tolerance_per_site &&
        density_spread <= density_tolerance && workload_match && provenance_match &&
        topology_match && rss_kib > 0 && isfinite(seconds) && seconds > 0 &&
        timing_relative_range <= timing_range_tolerance
    reason = status != "complete" ? status :
        energy_delta > energy_tolerance_per_site ? "energy mismatch" :
        density_delta > density_tolerance ? "density mismatch" :
        energy_spread > energy_tolerance_per_site ? "repeat energy spread" :
        density_spread > density_tolerance ? "repeat density spread" :
        !workload_match ? "fixed-mu workload mismatch" :
        !provenance_match ? "provenance mismatch" :
        !topology_match ? "thread/backend mismatch" :
        rss_kib == 0 ? "missing MaxRSS" :
        !(isfinite(seconds) && seconds > 0) ? "invalid timing" :
        timing_relative_range > timing_range_tolerance ? "unstable timing" : "valid"
    push!(rows, merge(candidate, (; status, valid, reason, seconds, energy, density,
        energy_delta_per_site=energy_delta, density_delta, timing_relative_range,
        rss_gib=rss_kib / 1048576, memory_gib=memory_mib / 1024,
        physical_cores, node_hours)))
end

valid_rows = filter(row -> row.valid, rows)
isempty(valid_rows) && error("no candidate passed the numerical-equivalence and MaxRSS gates")
sort!(valid_rows; by=row -> row.node_hours)
winner = first(valid_rows)
legacy_gpu_node_hours_low = legacy_gpu_seconds_low / 3600 / 4
legacy_gpu_node_hours_high = legacy_gpu_seconds_high / 3600 / 4
cpu_to_gpu_time_ratio_low = winner.seconds / legacy_gpu_seconds_high
cpu_to_gpu_time_ratio_high = winner.seconds / legacy_gpu_seconds_low
raw_node_hour_ratio_low = winner.node_hours / legacy_gpu_node_hours_high
raw_node_hour_ratio_high = winner.node_hours / legacy_gpu_node_hours_low

summary_path = joinpath(run_directory, "summary.csv")
open(summary_path, "w") do io
    println(io, "label,backend,julia_threads,slurm_logical_cpus,status,valid,reason,median_seconds,timing_relative_range,energy_delta_per_site,density_delta,max_rss_gib,recommended_memory_gib,charged_physical_cores,projected_node_hours_per_dmrg_solve")
    for row in rows
        @printf(io, "%s,%s,%d,%d,%s,%s,%s,%.9g,%.9g,%.9g,%.9g,%.6g,%.6g,%d,%.9g\n",
            row.label, row.backend, row.threads, row.logical, row.status, row.valid, row.reason,
            row.seconds, row.timing_relative_range, row.energy_delta_per_site, row.density_delta,
            row.rss_gib, row.memory_gib, row.physical_cores, row.node_hours)
    end
end

recommendation_env = joinpath(run_directory, "recommendation.env")
open(recommendation_env, "w") do io
    println(io, "PHASE0_RECOMMENDED_LABEL=$(winner.label)")
    println(io, "PHASE0_RECOMMENDED_BACKEND=$(winner.backend)")
    println(io, "PHASE0_RECOMMENDED_JULIA_THREADS=$(winner.threads)")
    println(io, "PHASE0_RECOMMENDED_SLURM_CPUS=$(winner.logical)")
    println(io, "PHASE0_RECOMMENDED_MEMORY_GIB=$(ceil(Int, winner.memory_gib))")
    println(io, "PHASE0_PROJECTED_NODE_HOURS_PER_SOLVE=$(winner.node_hours)")
    println(io, "PHASE0_TARGET_DENSITY=$target_density")
    println(io, "PHASE0_BENCHMARK_CHEMICAL_POTENTIAL=$benchmark_mu")
    println(io, "PHASE0_LEGACY_GPU_SECONDS_LOW=$legacy_gpu_seconds_low")
    println(io, "PHASE0_LEGACY_GPU_SECONDS_HIGH=$legacy_gpu_seconds_high")
end

recommendation_path = joinpath(run_directory, "recommendation.md")
open(recommendation_path, "w") do io
    println(io, "# Phase 0 CPU calibration recommendation")
    println(io)
    println(io, "Winner: `$(winner.label)` using the exclusive `$(winner.backend)` backend with $(winner.threads) Julia/compute threads.")
    println(io)
    println(io, "- Median fixed-mu DMRG time: `$(winner.seconds)` s for one $(baseline["nsweeps"])-sweep solve")
    println(io, "- MaxRSS: `$(round(winner.rss_gib; digits=3))` GiB")
    println(io, "- Recommended memory after 30% margin and 2 GiB rounding: `$(ceil(Int, winner.memory_gib))G`")
    println(io, "- Projected shared-QOS charge: `$(winner.node_hours)` node-hours per DMRG solve")
    println(io, "- Benchmark chemical potential / resulting density: `$benchmark_mu` / `$(winner.density)`")
    println(io, "- Repeat timing relative range: `$(winner.timing_relative_range)`")
    println(io, "- Numerical gates relative to `serial-t1`: |delta E|/(2L) <= `$(energy_tolerance_per_site)` and |delta n| <= `$(density_tolerance)`")
    println(io)
    println(io, "The timed region is exactly `run_dmrg_ground`: MPO construction, MPS copying, compilation warmup, GC, density measurement, and chemical-potential search are excluded. This is a production-scale resource calibration at chi=$(baseline["maxdim"]) and $(baseline["nsweeps"]) sweeps, not a scientific convergence or phase-ordering result.")
    println(io)
    println(io, "## Estimated comparison with the legacy GPU path")
    println(io)
    println(io, "- Estimated legacy one-GPU time for the same six-sweep chi=200 fixed-mu solve: `$(legacy_gpu_seconds_low)`--`$(legacy_gpu_seconds_high)` s")
    println(io, "- Estimated shared-QOS GPU charge: `$(legacy_gpu_node_hours_low)`--`$(legacy_gpu_node_hours_high)` GPU node-hours (one of four GPUs on a node)")
    println(io, "- Measured CPU / estimated GPU wall-time ratio: `$(cpu_to_gpu_time_ratio_low)`--`$(cpu_to_gpu_time_ratio_high)`")
    println(io, "- Raw CPU-node-hour / GPU-node-hour numerical ratio: `$(raw_node_hour_ratio_low)`--`$(raw_node_hour_ratio_high)`")
    println(io)
    println(io, "The GPU range is an extrapolation from the saved legacy chi=500 and chi=1000 sweep logs, not a matched sacct measurement. It is also not a pure hardware comparison: this CPU path conserves total S_z and fermion-number parity, whereas the legacy GPU path disables those quantum numbers. CPU and GPU node-hours are separate NERSC allocation pools and are not exchangeable cost units. A matched legacy-GPU timing job would be required for a definitive crossover claim.")
    println(io)
    println(io, "Full candidate table: `summary.csv`.")
end
println(read(recommendation_path, String))
