#!/usr/bin/env julia

using HDF5
using ITensorMPS
using LadderMPSMFT
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
const COMPATIBLE_PHASE0_SEED_COMMITS = Set([
    # Phase 0 v1.3.1 changes only staged Slurm dependency handling and seed
    # lineage recording. Reusing the completed v1.3.0 warm seed is safe after
    # its model, config, E_p registry, and immutable file hash are verified.
    "38697d803a7a15218cd54b9df1507a41fa76587a",
])
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
    psi = productMPS(
        sites,
        density_product_state(2 * model.L, model.density; rng=MersenneTwister(1)),
    )
    hamiltonian = build_mf_mpo(sites, model, fields, 0.0)
    dmrg = DMRGSettings(
        nsweeps=1,
        maxdim=4,
        cutoff=1e-6,
        energy_tol=0.0,
        output_level=0,
        max_time_seconds=300.0,
    )
    run_dmrg_ground(
        sites,
        hamiltonian,
        model.density,
        dmrg;
        psi_init=psi,
        rng=MersenneTwister(1),
        deadline=time() + 300,
    )
    return nothing
end

# Compilation is deliberately outside the timed region.
get(ENV, "PHASE0_COMPILE_WARMUP", "1") == "1" && compile_warmup()

seed_path = get(ENV, "PHASE0_SEED_STATE", "")
isempty(seed_path) && error("PHASE0_SEED_STATE is required for a provenance-matched timing payload")
isfile(seed_path) || error("PHASE0_SEED_STATE does not exist: $seed_path")
loaded = h5open(seed_path, "r") do file
    Int(read(file, "schema_version")) >= 3 || error(
        "Phase 0 seed predates fixed-mu DMRG schema v3; submit a new calibration run",
    )
    String(read(file, "benchmark_kind")) == "fixed_mu_dmrg" || error(
        "Phase 0 seed is not a fixed-mu DMRG benchmark seed",
    )
    read(file, "model_fingerprint") == LadderMPSMFT.model_fingerprint(settings.model) ||
        error("Phase 0 seed model fingerprint differs from the payload model")
    String(read(file, "ep_source_sha256")) == current_ep_source_sha256 || error(
        "Phase 0 seed E_p registry fingerprint differs from the payload registry",
    )
    seed_git_commit = String(read(file, "git_commit"))
    seed_implementation_sha256 = String(read(file, "implementation_sha256"))
    seed_implementation_sha256 == current_implementation_sha256 ||
        seed_git_commit in COMPATIBLE_PHASE0_SEED_COMMITS || error(
            "Phase 0 seed implementation fingerprint differs from the payload implementation",
        )
    benchmark_mu = Float64(read(file, "chemical_potential"))
    isapprox(benchmark_mu, settings.model.mu_initial; atol=1e-12, rtol=0.0) || error(
        "Phase 0 seed mu $benchmark_mu differs from config mu $(settings.model.mu_initial)",
    )
    target_density = Float64(read(file, "target_density"))
    isapprox(target_density, settings.model.density; atol=1e-12, rtol=0.0) || error(
        "Phase 0 seed target density $target_density differs from config target $(settings.model.density)",
    )
    return (
        psi=read(file, "psi", MPS),
        fields=FieldState(
            read(file, "fields/alpha"),
            read(file, "fields/beta"),
            read(file, "fields/mu_cdw"),
        ),
        benchmark_mu,
        target_density,
        seed_density=Float64(read(file, "density")),
        seed_config_sha256=String(read(file, "config_sha256")),
        seed_git_commit,
        seed_implementation_sha256,
    )
end

seed_psi = loaded.psi
sites = siteinds(seed_psi)
# MPO construction is configuration-independent setup and is not timed.
hamiltonian = build_mf_mpo(sites, settings.model, loaded.fields, loaded.benchmark_mu)
seed_source = abspath(seed_path)
repetitions = parse(Int, get(ENV, "PHASE0_REPETITIONS", "3"))
repetitions >= 1 || error("PHASE0_REPETITIONS must be positive")

seconds = Float64[]
energies = Float64[]
densities = Float64[]
bond_dimensions = Int[]
timed_out = Bool[]
energy_converged = Bool[]
for _ in 1:repetitions
    # Copying the common initial MPS and forcing GC are also outside timing.
    psi_initial = copy(seed_psi)
    GC.gc()
    result_ref = Ref{Any}()
    elapsed = @elapsed result_ref[] = run_dmrg_ground(
        sites,
        hamiltonian,
        settings.model.density,
        settings.dmrg;
        psi_init=psi_initial,
        rng=MersenneTwister(settings.run.random_seed),
        deadline=time() + settings.dmrg.max_time_seconds,
    )
    result = result_ref[]
    push!(seconds, elapsed)
    push!(energies, result.energy)
    push!(densities, LadderMPSMFT.average_density(result.psi))
    push!(bond_dimensions, maxlinkdim(result.psi))
    push!(timed_out, result.timed_out)
    push!(energy_converged, result.energy_converged)
end
status = any(timed_out) ? "time_limit" : "complete"

metric = Dict{String,Any}(
    "schema_version" => 4,
    "benchmark_kind" => "fixed_mu_dmrg",
    "label" => label,
    "backend" => String(backend),
    "status" => status,
    "repetitions" => repetitions,
    "dmrg_solves" => fill(1, repetitions),
    "seconds" => seconds,
    "median_seconds" => median(seconds),
    "minimum_seconds" => minimum(seconds),
    "energies" => energies,
    "representative_energy" => median(energies),
    "energy_spread" => maximum(energies) - minimum(energies),
    "densities" => densities,
    "representative_density" => median(densities),
    "density_spread" => maximum(densities) - minimum(densities),
    "target_density" => loaded.target_density,
    "benchmark_chemical_potential" => loaded.benchmark_mu,
    "maximum_bond_dimensions" => bond_dimensions,
    "timed_out" => timed_out,
    "energy_converged" => energy_converged,
    "timed_region" => "run_dmrg_ground_only",
    "mpo_construction_timed" => false,
    "initial_mps_copy_timed" => false,
    "garbage_collection_timed" => false,
    "compile_warmup_timed" => false,
    "L" => settings.model.L,
    "physical_sites" => 2 * settings.model.L,
    "maxdim" => settings.dmrg.maxdim,
    "nsweeps" => settings.dmrg.nsweeps,
    "geometry" => String(settings.model.geometry),
    "model_fingerprint" => LadderMPSMFT.model_fingerprint(settings.model),
    "config_path" => abspath(config_path),
    "config_sha256" => LadderMPSMFT.sha256_file(abspath(config_path)),
    "ep_source_sha256" => current_ep_source_sha256,
    "seed_source" => seed_source,
    "seed_sha256" => LadderMPSMFT.sha256_file(seed_path),
    "seed_density" => loaded.seed_density,
    "seed_chemical_potential" => loaded.benchmark_mu,
    "seed_config_sha256" => loaded.seed_config_sha256,
    "seed_git_commit" => loaded.seed_git_commit,
    "seed_implementation_sha256" => loaded.seed_implementation_sha256,
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
