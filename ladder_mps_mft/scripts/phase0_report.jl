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

function target_density_error(metric, target_density)
    densities = Float64.(metric["densities"])
    isempty(densities) && return Inf
    return maximum(abs.(densities .- target_density))
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
Int(get(baseline, "schema_version", 0)) >= 3 || error(
    "serial baseline predates density-search metric schema v3; rerun Phase 0",
)
baseline["status"] == "complete" || error("serial baseline did not complete")
baseline_energy = Float64(baseline["representative_energy"])
baseline_density = Float64(baseline["representative_density"])
target_density = Float64(baseline["target_density"])
target_density_tolerance = Float64(baseline["density_target_tolerance"])
chemical_potential = Float64(baseline["representative_chemical_potential"])
seed_chemical_potential = Float64(baseline["seed_chemical_potential"])
baseline_mu_evaluations = Int.(baseline["mu_evaluations"])
length(unique(baseline_mu_evaluations)) == 1 || error(
    "serial baseline repeated different numbers of density-search evaluations",
)
baseline_mu_evaluation_count = first(baseline_mu_evaluations)
all(Bool.(baseline["mu_density_converged"])) || error(
    "serial baseline contains a failed density search",
)
baseline_target_error = target_density_error(baseline, target_density)
stored_baseline_target_error = Float64(baseline["maximum_density_error_to_target"])
isapprox(stored_baseline_target_error, baseline_target_error; atol=1e-12, rtol=1e-12) || error(
    "serial baseline stored/recomputed target-density errors disagree",
)
Bool(baseline["seed_mu_density_converged"]) || error("serial baseline used an untargeted seed")
Float64(baseline["seed_density_error"]) <= target_density_tolerance || error(
    "serial baseline seed missed target density",
)
baseline_target_error <= target_density_tolerance || error(
    "serial baseline density error $baseline_target_error exceeds target tolerance $target_density_tolerance",
)
baseline_seed_sha256 = String(baseline["seed_sha256"])
baseline_config_sha256 = String(baseline["config_sha256"])
baseline_seed_config_sha256 = String(baseline["seed_config_sha256"])
baseline_seed_config_sha256 == baseline_config_sha256 || error(
    "serial baseline seed was not created from the timing config",
)
baseline_ep_sha256 = String(baseline["ep_source_sha256"])
baseline_model_fingerprint = String(baseline["model_fingerprint"])
baseline_implementation_sha256 = String(baseline["implementation_sha256"])
baseline_git_commit = String(baseline["git_commit"])
physical_sites = Int(baseline["physical_sites"])
energy_tolerance_per_site = parse(Float64, get(ENV, "PHASE0_ENERGY_TOL_PER_SITE", "1e-8"))
density_tolerance = parse(Float64, get(ENV, "PHASE0_DENSITY_TOL", "1e-6"))
mu_tolerance = parse(Float64, get(ENV, "PHASE0_MU_TOL", "1e-8"))
timing_range_tolerance = parse(Float64, get(ENV, "PHASE0_TIMING_RELATIVE_RANGE_MAX", "0.10"))
Float64(baseline["energy_spread"]) / physical_sites <= energy_tolerance_per_site || error(
    "serial baseline repeated inconsistent energies",
)
Float64(baseline["density_spread"]) <= density_tolerance || error(
    "serial baseline repeated inconsistent densities",
)
Float64(baseline["chemical_potential_spread"]) <= mu_tolerance || error(
    "serial baseline repeated inconsistent chemical potentials",
)

rows = NamedTuple[]
for candidate in candidates
    metric_path = joinpath(run_directory, "metrics", "$(candidate.label).toml")
    time_path = joinpath(run_directory, "metrics", "$(candidate.label).time")
    if !isfile(metric_path)
        push!(rows, merge(candidate, (; status="missing", valid=false, reason="missing metric", seconds=NaN,
            energy=NaN, density=NaN, energy_delta_per_site=NaN, density_delta=NaN,
            target_density_error=Inf, chemical_potential=NaN, chemical_potential_delta=Inf,
            mu_evaluations=0, timing_relative_range=NaN,
            rss_gib=NaN, memory_gib=NaN, physical_cores=0, node_hours=Inf)))
        continue
    end
    metric = TOML.parsefile(metric_path)
    if Int(get(metric, "schema_version", 0)) < 3
        push!(rows, merge(candidate, (; status="legacy", valid=false, reason="legacy metric schema", seconds=NaN,
            energy=NaN, density=NaN, energy_delta_per_site=NaN, density_delta=NaN,
            target_density_error=Inf, chemical_potential=NaN, chemical_potential_delta=Inf,
            mu_evaluations=0, timing_relative_range=NaN,
            rss_gib=NaN, memory_gib=NaN, physical_cores=0, node_hours=Inf)))
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
    candidate_target_density = Float64(metric["target_density"])
    candidate_target_tolerance = Float64(metric["density_target_tolerance"])
    candidate_mu = Float64(metric["representative_chemical_potential"])
    candidate_mu_spread = Float64(metric["chemical_potential_spread"])
    candidate_seed_mu = Float64(metric["seed_chemical_potential"])
    candidate_mu_evaluations = Int.(metric["mu_evaluations"])
    repeated_evaluation_count = length(unique(candidate_mu_evaluations)) == 1
    candidate_mu_evaluation_count = repeated_evaluation_count ? first(candidate_mu_evaluations) : 0
    chemical_potential_delta = abs(candidate_mu - chemical_potential)
    candidate_target_error = target_density_error(metric, target_density)
    stored_target_error = Float64(metric["maximum_density_error_to_target"])
    target_contract_match = isapprox(candidate_target_density, target_density; atol=1e-12, rtol=0.0) &&
        isapprox(candidate_target_tolerance, target_density_tolerance; atol=1e-12, rtol=0.0) &&
        isapprox(candidate_seed_mu, seed_chemical_potential; atol=1e-12, rtol=0.0) &&
        isapprox(stored_target_error, candidate_target_error; atol=1e-12, rtol=1e-12) &&
        all(Bool.(metric["mu_density_converged"])) &&
        Bool(metric["seed_mu_density_converged"]) &&
        Float64(metric["seed_density_error"]) <= target_density_tolerance
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
        density_spread <= density_tolerance && candidate_target_error <= target_density_tolerance &&
        chemical_potential_delta <= mu_tolerance && candidate_mu_spread <= mu_tolerance &&
        repeated_evaluation_count && candidate_mu_evaluation_count == baseline_mu_evaluation_count &&
        target_contract_match && provenance_match && topology_match && rss_kib > 0 &&
        isfinite(seconds) && seconds > 0 && timing_relative_range <= timing_range_tolerance
    reason = status != "complete" ? status :
        energy_delta > energy_tolerance_per_site ? "energy mismatch" :
        density_delta > density_tolerance ? "density mismatch" :
        energy_spread > energy_tolerance_per_site ? "repeat energy spread" :
        density_spread > density_tolerance ? "repeat density spread" :
        candidate_target_error > target_density_tolerance ? "target density miss" :
        chemical_potential_delta > mu_tolerance ? "chemical-potential mismatch" :
        candidate_mu_spread > mu_tolerance ? "repeat chemical-potential spread" :
        !repeated_evaluation_count ? "repeat density-search path mismatch" :
        candidate_mu_evaluation_count != baseline_mu_evaluation_count ? "density-search path mismatch" :
        !target_contract_match ? "target-density contract mismatch" :
        !provenance_match ? "provenance mismatch" :
        !topology_match ? "thread/backend mismatch" :
        rss_kib == 0 ? "missing MaxRSS" :
        !(isfinite(seconds) && seconds > 0) ? "invalid timing" :
        timing_relative_range > timing_range_tolerance ? "unstable timing" : "valid"
    push!(rows, merge(candidate, (; status, valid, reason, seconds, energy, density,
        energy_delta_per_site=energy_delta, density_delta, target_density_error=candidate_target_error,
        chemical_potential=candidate_mu, chemical_potential_delta,
        mu_evaluations=candidate_mu_evaluation_count, timing_relative_range, rss_gib=rss_kib / 1048576,
        memory_gib=memory_mib / 1024, physical_cores, node_hours)))
end

valid_rows = filter(row -> row.valid, rows)
isempty(valid_rows) && error("no candidate passed the numerical-equivalence and MaxRSS gates")
sort!(valid_rows; by=row -> row.node_hours)
winner = first(valid_rows)

summary_path = joinpath(run_directory, "summary.csv")
open(summary_path, "w") do io
    println(io, "label,backend,julia_threads,slurm_logical_cpus,status,valid,reason,median_seconds,timing_relative_range,mu_evaluations,energy_delta_per_site,density_delta,target_density,max_density_error_to_target,chemical_potential,chemical_potential_delta,max_rss_gib,recommended_memory_gib,charged_physical_cores,projected_node_hours_per_density_search")
    for row in rows
        @printf(io, "%s,%s,%d,%d,%s,%s,%s,%.9g,%.9g,%d,%.9g,%.9g,%.9g,%.9g,%.12g,%.9g,%.6g,%.6g,%d,%.9g\n",
            row.label, row.backend, row.threads, row.logical, row.status, row.valid, row.reason,
            row.seconds, row.timing_relative_range, row.mu_evaluations, row.energy_delta_per_site,
            row.density_delta, target_density, row.target_density_error, row.chemical_potential,
            row.chemical_potential_delta, row.rss_gib,
            row.memory_gib, row.physical_cores, row.node_hours)
    end
end

validation_path = joinpath(run_directory, "metrics", "validation.toml")
validation = isfile(validation_path) ? TOML.parsefile(validation_path) : nothing
validation_target_error = validation === nothing ? Inf : target_density_error(validation, target_density)
validation_mu = validation === nothing ? NaN : Float64(validation["representative_chemical_potential"])
validation_mu_evaluations = validation === nothing ? Int[] : Int.(validation["mu_evaluations"])
validation_rss_kib = validation === nothing ? 0 : maximum_rss_kib(
    joinpath(run_directory, "metrics", "validation.time"),
)
validation_stored_target_error = validation === nothing ? Inf :
    Float64(validation["maximum_density_error_to_target"])
validation_accepted = validation !== nothing && Int(get(validation, "schema_version", 0)) >= 3 &&
    String(validation["status"]) == "complete" &&
    isapprox(Float64(validation["target_density"]), target_density; atol=1e-12, rtol=0.0) &&
    isapprox(Float64(validation["seed_chemical_potential"]), seed_chemical_potential; atol=1e-12, rtol=0.0) &&
    isapprox(validation_stored_target_error, validation_target_error; atol=1e-12, rtol=1e-12) &&
    validation_target_error <= Float64(validation["density_target_tolerance"]) &&
    all(Bool.(validation["mu_density_converged"])) &&
    String(validation["seed_sha256"]) == baseline_seed_sha256 &&
    String(validation["seed_config_sha256"]) == baseline_seed_config_sha256 &&
    String(validation["ep_source_sha256"]) == baseline_ep_sha256 &&
    String(validation["model_fingerprint"]) == baseline_model_fingerprint &&
    String(validation["implementation_sha256"]) == baseline_implementation_sha256 &&
    String(validation["git_commit"]) == baseline_git_commit &&
    topology_matches(validation, winner) && validation_rss_kib > 0

recommendation_env = joinpath(run_directory, "recommendation.env")
open(recommendation_env, "w") do io
    println(io, "PHASE0_RECOMMENDED_LABEL=$(winner.label)")
    println(io, "PHASE0_RECOMMENDED_BACKEND=$(winner.backend)")
    println(io, "PHASE0_RECOMMENDED_JULIA_THREADS=$(winner.threads)")
    println(io, "PHASE0_RECOMMENDED_SLURM_CPUS=$(winner.logical)")
    println(io, "PHASE0_RECOMMENDED_MEMORY_GIB=$(ceil(Int, winner.memory_gib))")
    println(io, "PHASE0_PROJECTED_NODE_HOURS_PER_SOLVE=$(winner.node_hours)")
    println(io, "PHASE0_TARGET_DENSITY=$target_density")
    println(io, "PHASE0_SEED_CHEMICAL_POTENTIAL=$seed_chemical_potential")
    println(io, "PHASE0_RECOMMENDED_CHEMICAL_POTENTIAL=$chemical_potential")
    println(io, "PHASE0_VALIDATION_PRESENT=$(validation !== nothing)")
    println(io, "PHASE0_VALIDATION_ACCEPTED=$validation_accepted")
end

recommendation_path = joinpath(run_directory, "recommendation.md")
open(recommendation_path, "w") do io
    println(io, "# Phase 0 CPU calibration recommendation")
    println(io)
    println(io, "Winner: `$(winner.label)` using the exclusive `$(winner.backend)` backend with $(winner.threads) Julia/compute threads.")
    println(io)
    println(io, "- Median density-targeted timing payload: `$(winner.seconds)` s ($(winner.mu_evaluations) DMRG evaluation(s))")
    println(io, "- MaxRSS: `$(round(winner.rss_gib; digits=3))` GiB")
    println(io, "- Recommended memory after 30% margin and 2 GiB rounding: `$(ceil(Int, winner.memory_gib))G`")
    println(io, "- Projected shared-QOS charge: `$(winner.node_hours)` node-hours per density-targeting call")
    println(io, "- Target density / achieved maximum error: `$target_density` / `$(winner.target_density_error)`")
    println(io, "- Seed / converged chemical potential: `$seed_chemical_potential` / `$chemical_potential`")
    println(io, "- Repeat timing relative range: `$(winner.timing_relative_range)`")
    println(io, "- Timing-stability gate: relative range <= `$(timing_range_tolerance)`")
    println(io, "- Numerical gates: |delta E|/(2L) <= `$(energy_tolerance_per_site)`, |delta n| <= `$(density_tolerance)`, and |delta mu| <= `$(mu_tolerance)` relative to `serial-t1`; every repetition must satisfy |n-n_target| <= `$(target_density_tolerance)` and follow the same density-search path")
    println(io)
    println(io, "This is a timing-only result at chi=$(baseline["maxdim"]) and $(baseline["nsweeps"]) sweeps. Run the separate chi=200 validation before choosing production resources. It does not establish scientific convergence or CPU superiority over the legacy GPU workflow.")
    if validation !== nothing
        println(io)
        println(io, "Validation metric present: status `$(validation["status"])`, median `$(validation["median_seconds"])` s, maxdim `$(validation["maxdim"])`, converged mu `$validation_mu`, and $(join(validation_mu_evaluations, ",")) density-search evaluation(s).")
        println(io, "Validation target-density error: `$validation_target_error`; MaxRSS: `$(validation_rss_kib / 1048576)` GiB (accepted: `$validation_accepted`).")
    end
    println(io)
    println(io, "Full candidate table: `summary.csv`.")
end
println(read(recommendation_path, String))
