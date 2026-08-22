#!/usr/bin/env julia

using LadderMPSMFT
using HDF5
using ITensorMPS
using Random
using Statistics
using TOML

length(ARGS) == 4 || error(
    "usage: julia --project=. scripts/phase0_payload.jl CONFIG.toml OUTPUT.toml LABEL BACKEND",
)
config_path, output_path, label, backend_raw = ARGS
backend = Symbol(lowercase(backend_raw))
backend in (:serial, :blocksparse, :strided, :blas) || error("unknown backend: $backend")
ispath(output_path) && error("refusing to overwrite Phase 0 metric: $output_path")

settings = load_settings(config_path)
current_git_commit = LadderMPSMFT._read_git("rev-parse", "HEAD")
current_implementation_sha256 = LadderMPSMFT.implementation_fingerprint()
current_ep_source_sha256 = LadderMPSMFT.sha256_file(settings.model.ep_source)
threads = Threads.nthreads()
runtime = if backend == :blocksparse
    RuntimeSettings(blas_threads=1, strided_threads=1, threaded_blocksparse=true)
elseif backend == :strided
    RuntimeSettings(blas_threads=1, strided_threads=threads, threaded_blocksparse=false)
elseif backend == :blas
    RuntimeSettings(blas_threads=threads, strided_threads=1, threaded_blocksparse=false)
else
    RuntimeSettings(blas_threads=1, strided_threads=1, threaded_blocksparse=false)
end
threading = configure_threading!(runtime)

function compile_warmup()
    model = ModelSettings(
        L=2,
        U=2.0,
        t0=1.0,
        tp=0.01,
        density=1.0,
        r_range=1,
        geometry=:square,
        ep=0.2,
        ep_signed=-0.2,
        ep_source="phase0_compile_warmup",
    )
    sites = LadderMPSMFT.make_sites(model)
    fields = initial_fields(model; seed=:zero)
    psi = productMPS(sites, density_product_state(2 * model.L, model.density; rng=MersenneTwister(1)))
    dmrg = DMRGSettings(nsweeps=1, maxdim=4, cutoff=1e-6, energy_tol=0.0, output_level=0, max_time_seconds=300.0)
    find_mu_for_density(
        sites,
        model,
        fields,
        0.0,
        dmrg;
        psi_init=psi,
        rng=MersenneTwister(1),
        deadline=time() + 300,
    )
    return nothing
end

get(ENV, "PHASE0_COMPILE_WARMUP", "1") == "1" && compile_warmup()

seed_path = get(ENV, "PHASE0_SEED_STATE", "")
isempty(seed_path) && error("PHASE0_SEED_STATE is required for a provenance-matched timing payload")
isfile(seed_path) || error("PHASE0_SEED_STATE does not exist: $seed_path")
loaded = h5open(seed_path, "r") do file
    Int(read(file, "schema_version")) >= 2 || error(
        "Phase 0 seed predates density-targeted schema v2; submit a new calibration run",
    )
    read(file, "model_fingerprint") == LadderMPSMFT.model_fingerprint(settings.model) ||
        error("Phase 0 seed model fingerprint differs from the payload model")
    seed_ep_source_sha256 = String(read(file, "ep_source_sha256"))
    seed_ep_source_sha256 == current_ep_source_sha256 || error(
        "Phase 0 seed E_p registry fingerprint differs from the payload registry",
    )
    seed_git_commit = String(read(file, "git_commit"))
    seed_git_commit == current_git_commit || error(
        "Phase 0 seed git commit differs from the payload commit",
    )
    seed_implementation_sha256 = String(read(file, "implementation_sha256"))
    seed_implementation_sha256 == current_implementation_sha256 || error(
        "Phase 0 seed implementation fingerprint differs from the payload implementation",
    )
    target_density = Float64(read(file, "target_density"))
    isapprox(target_density, settings.model.density; atol=1e-12, rtol=0.0) || error(
        "Phase 0 seed target density $target_density differs from config target $(settings.model.density)",
    )
    seed_density = Float64(read(file, "density"))
    seed_density_error = abs(seed_density - target_density)
    Bool(read(file, "mu_density_converged")) || error("Phase 0 seed did not pass density targeting")
    seed_density_error <= settings.dmrg.mu_density_tol || error(
        "Phase 0 seed density error $seed_density_error exceeds payload tolerance $(settings.dmrg.mu_density_tol)",
    )
    return (
        psi=read(file, "psi", MPS),
        fields=FieldState(
            read(file, "fields/alpha"),
            read(file, "fields/beta"),
            read(file, "fields/mu_cdw"),
        ),
        chemical_potential=Float64(read(file, "chemical_potential")),
        target_density,
        seed_density,
        seed_density_error,
        seed_mu_search_status=String(read(file, "mu_search_status")),
        seed_mu_evaluations=Int(read(file, "mu_evaluations")),
        seed_config_sha256=String(read(file, "config_sha256")),
    )
end
seed_psi = loaded.psi
fields = loaded.fields
seed_chemical_potential = loaded.chemical_potential
sites = siteinds(seed_psi)
seed_source = abspath(seed_path)
repetitions = parse(Int, get(ENV, "PHASE0_REPETITIONS", "3"))
repetitions >= 1 || error("PHASE0_REPETITIONS must be positive")

seconds = Float64[]
energies = Float64[]
densities = Float64[]
chemical_potentials = Float64[]
mu_evaluations = Int[]
mu_search_statuses = String[]
mu_density_converged = Bool[]
bond_dimensions = Int[]
timed_out = Bool[]
for _ in 1:repetitions
    GC.gc()
    result_ref = Ref{Any}()
    elapsed = @elapsed result_ref[] = find_mu_for_density(
        sites,
        settings.model,
        fields,
        seed_chemical_potential,
        settings.dmrg;
        psi_init=copy(seed_psi),
        rng=MersenneTwister(settings.run.random_seed),
        deadline=time() + settings.dmrg.max_time_seconds,
    )
    result = result_ref[]
    push!(seconds, elapsed)
    push!(energies, result.energy)
    push!(densities, result.density)
    push!(chemical_potentials, result.mu)
    push!(mu_evaluations, result.evaluations)
    push!(mu_search_statuses, String(result.status))
    push!(mu_density_converged, result.converged)
    push!(bond_dimensions, maxlinkdim(result.psi))
    push!(timed_out, result.timed_out)
end
density_errors = abs.(densities .- loaded.target_density)
status = any(timed_out) ? "time_limit" :
    all(mu_density_converged) ? "complete" : "density_failure"

metric = Dict{String,Any}(
    "schema_version" => 3,
    "label" => label,
    "backend" => String(backend),
    "status" => status,
    "repetitions" => repetitions,
    "seconds" => seconds,
    "median_seconds" => median(seconds),
    "minimum_seconds" => minimum(seconds),
    "energies" => energies,
    "representative_energy" => median(energies),
    "energy_spread" => maximum(energies) - minimum(energies),
    "densities" => densities,
    "representative_density" => median(densities),
    "density_spread" => maximum(densities) - minimum(densities),
    "density_errors_to_target" => density_errors,
    "maximum_density_error_to_target" => maximum(density_errors),
    "target_density" => loaded.target_density,
    "density_target_tolerance" => settings.dmrg.mu_density_tol,
    "chemical_potentials" => chemical_potentials,
    "representative_chemical_potential" => median(chemical_potentials),
    "chemical_potential_spread" => maximum(chemical_potentials) - minimum(chemical_potentials),
    "seed_chemical_potential" => seed_chemical_potential,
    "mu_evaluations" => mu_evaluations,
    "mu_search_statuses" => mu_search_statuses,
    "mu_density_converged" => mu_density_converged,
    "maximum_bond_dimensions" => bond_dimensions,
    "timed_out" => timed_out,
    "L" => settings.model.L,
    "physical_sites" => 2 * settings.model.L,
    "maxdim" => settings.dmrg.maxdim,
    "nsweeps" => settings.dmrg.nsweeps,
    "geometry" => String(settings.model.geometry),
    "model_fingerprint" => LadderMPSMFT.model_fingerprint(settings.model),
    "config_path" => abspath(config_path),
    "config_sha256" => LadderMPSMFT.sha256_file(abspath(config_path)),
    "ep_source_sha256" => LadderMPSMFT.sha256_file(settings.model.ep_source),
    "seed_source" => seed_source,
    "seed_sha256" => LadderMPSMFT.sha256_file(seed_path),
    "seed_density" => loaded.seed_density,
    "seed_density_error" => loaded.seed_density_error,
    "seed_mu_search_status" => loaded.seed_mu_search_status,
    "seed_mu_evaluations" => loaded.seed_mu_evaluations,
    "seed_mu_density_converged" => true,
    "seed_config_sha256" => loaded.seed_config_sha256,
    "git_commit" => current_git_commit,
    "implementation_sha256" => current_implementation_sha256,
    "hostname" => get(ENV, "HOSTNAME", "unknown"),
    "slurm_job_id" => get(ENV, "SLURM_JOB_ID", ""),
    "julia_threads" => threading.julia,
    "blas_threads" => threading.blas,
    "strided_threads" => threading.strided,
    "threaded_blocksparse" => threading.blocksparse,
)
mkpath(dirname(abspath(output_path)))
temporary = tempname(dirname(abspath(output_path)))
open(temporary, "w") do io
    TOML.print(io, metric; sorted=true)
end
mv(temporary, abspath(output_path))
println("metric_path=$(abspath(output_path))")
