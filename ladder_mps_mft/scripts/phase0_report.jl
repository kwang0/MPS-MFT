#!/usr/bin/env julia

using Printf
using Statistics
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

candidates = NamedTuple[]
for (index, line) in enumerate(eachline(candidates_path))
    index == 1 && continue
    isempty(strip(line)) && continue
    label, threads, backend, logical = split(line, '\t')
    push!(candidates, (; label, threads=parse(Int, threads), backend, logical=parse(Int, logical)))
end

baseline_candidate = findfirst(row -> row.label == "serial-t1", candidates)
baseline_candidate === nothing && error("candidate matrix must include serial-t1")
baseline_path = joinpath(run_directory, "metrics", "serial-t1.toml")
isfile(baseline_path) || error("serial baseline did not produce $baseline_path")
baseline = TOML.parsefile(baseline_path)
baseline["status"] == "complete" || error("serial baseline did not complete")
baseline_energy = Float64(baseline["representative_energy"])
baseline_density = Float64(baseline["representative_density"])
baseline_seed_sha256 = String(baseline["seed_sha256"])
baseline_config_sha256 = String(baseline["config_sha256"])
baseline_ep_sha256 = String(baseline["ep_source_sha256"])
baseline_implementation_sha256 = String(baseline["implementation_sha256"])
physical_sites = Int(baseline["physical_sites"])
energy_tolerance_per_site = parse(Float64, get(ENV, "PHASE0_ENERGY_TOL_PER_SITE", "1e-8"))
density_tolerance = parse(Float64, get(ENV, "PHASE0_DENSITY_TOL", "1e-6"))

rows = NamedTuple[]
for candidate in candidates
    metric_path = joinpath(run_directory, "metrics", "$(candidate.label).toml")
    time_path = joinpath(run_directory, "metrics", "$(candidate.label).time")
    if !isfile(metric_path)
        push!(rows, merge(candidate, (; status="missing", valid=false, reason="missing metric", seconds=NaN,
            energy=NaN, density=NaN, energy_delta_per_site=NaN, density_delta=NaN,
            rss_gib=NaN, memory_gib=NaN, physical_cores=0, node_hours=Inf)))
        continue
    end
    metric = TOML.parsefile(metric_path)
    status = String(metric["status"])
    seconds = Float64(metric["median_seconds"])
    energy = Float64(metric["representative_energy"])
    density = Float64(metric["representative_density"])
    energy_delta = abs(energy - baseline_energy) / physical_sites
    density_delta = abs(density - baseline_density)
    energy_spread = Float64(metric["energy_spread"]) / physical_sites
    density_spread = Float64(metric["density_spread"])
    provenance_match = String(metric["seed_sha256"]) == baseline_seed_sha256 &&
        String(metric["config_sha256"]) == baseline_config_sha256 &&
        String(metric["ep_source_sha256"]) == baseline_ep_sha256 &&
        String(metric["implementation_sha256"]) == baseline_implementation_sha256
    topology_match = String(metric["backend"]) == candidate.backend &&
        Int(metric["julia_threads"]) == candidate.threads
    rss_kib = maximum_rss_kib(time_path)
    memory_mib = rss_kib > 0 ? recommended_memory_mib(rss_kib) : 0
    physical_cores = memory_mib > 0 ? charged_physical_cores(candidate.logical, memory_mib) : 0
    node_hours = physical_cores > 0 ? seconds / 3600 * physical_cores / PHYSICAL_CORES_PER_NODE : Inf
    valid = status == "complete" && energy_delta <= energy_tolerance_per_site &&
        density_delta <= density_tolerance && energy_spread <= energy_tolerance_per_site &&
        density_spread <= density_tolerance && provenance_match && topology_match && rss_kib > 0
    reason = status != "complete" ? status :
        energy_delta > energy_tolerance_per_site ? "energy mismatch" :
        density_delta > density_tolerance ? "density mismatch" :
        energy_spread > energy_tolerance_per_site ? "repeat energy spread" :
        density_spread > density_tolerance ? "repeat density spread" :
        !provenance_match ? "provenance mismatch" :
        !topology_match ? "thread/backend mismatch" :
        rss_kib == 0 ? "missing MaxRSS" : "valid"
    push!(rows, merge(candidate, (; status, valid, reason, seconds, energy, density,
        energy_delta_per_site=energy_delta, density_delta, rss_gib=rss_kib / 1048576,
        memory_gib=memory_mib / 1024, physical_cores, node_hours)))
end

valid_rows = filter(row -> row.valid, rows)
isempty(valid_rows) && error("no candidate passed the numerical-equivalence and MaxRSS gates")
sort!(valid_rows; by=row -> row.node_hours)
winner = first(valid_rows)

summary_path = joinpath(run_directory, "summary.csv")
open(summary_path, "w") do io
    println(io, "label,backend,julia_threads,slurm_logical_cpus,status,valid,reason,median_seconds,energy_delta_per_site,density_delta,max_rss_gib,recommended_memory_gib,charged_physical_cores,projected_node_hours_per_solve")
    for row in rows
        @printf(io, "%s,%s,%d,%d,%s,%s,%s,%.9g,%.9g,%.9g,%.6g,%.6g,%d,%.9g\n",
            row.label, row.backend, row.threads, row.logical, row.status, row.valid, row.reason,
            row.seconds, row.energy_delta_per_site, row.density_delta, row.rss_gib,
            row.memory_gib, row.physical_cores, row.node_hours)
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
end

recommendation_path = joinpath(run_directory, "recommendation.md")
open(recommendation_path, "w") do io
    println(io, "# Phase 0 CPU calibration recommendation")
    println(io)
    println(io, "Winner: `$(winner.label)` using the exclusive `$(winner.backend)` backend with $(winner.threads) Julia/compute threads.")
    println(io)
    println(io, "- Median timing-payload solve: `$(winner.seconds)` s")
    println(io, "- MaxRSS: `$(round(winner.rss_gib; digits=3))` GiB")
    println(io, "- Recommended memory after 30% margin and 2 GiB rounding: `$(ceil(Int, winner.memory_gib))G`")
    println(io, "- Projected shared-QOS charge: `$(winner.node_hours)` node-hours per timing-payload solve")
    println(io, "- Numerical gates: |delta E|/(2L) <= `$(energy_tolerance_per_site)` and |delta n| <= `$(density_tolerance)` relative to `serial-t1`")
    println(io)
    println(io, "This is a timing-only result at chi=$(baseline["maxdim"]) and $(baseline["nsweeps"]) sweeps. Run the separate chi=200 validation before choosing production resources. It does not establish scientific convergence or CPU superiority over the legacy GPU workflow.")
    if isfile(joinpath(run_directory, "metrics", "validation.toml"))
        validation = TOML.parsefile(joinpath(run_directory, "metrics", "validation.toml"))
        println(io)
        println(io, "Validation metric present: status `$(validation["status"])`, median `$(validation["median_seconds"])` s, maxdim `$(validation["maxdim"])`.")
    end
    println(io)
    println(io, "Full candidate table: `summary.csv`.")
end
println(read(recommendation_path, String))
