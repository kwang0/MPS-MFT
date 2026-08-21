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
    hamiltonian = build_mf_mpo(sites, model, fields, 0.0)
    psi = productMPS(sites, density_product_state(2 * model.L, model.density; rng=MersenneTwister(1)))
    dmrg = DMRGSettings(nsweeps=1, maxdim=4, cutoff=1e-6, energy_tol=0.0, output_level=0, max_time_seconds=300.0)
    run_dmrg_ground(sites, hamiltonian, model.density, dmrg; psi_init=psi, deadline=time() + 300)
    return nothing
end

get(ENV, "PHASE0_COMPILE_WARMUP", "1") == "1" && compile_warmup()

seed_path = get(ENV, "PHASE0_SEED_STATE", "")
seed_source = "deterministic_product_state"
if !isempty(seed_path)
    isfile(seed_path) || error("PHASE0_SEED_STATE does not exist: $seed_path")
    loaded = h5open(seed_path, "r") do file
        read(file, "model_fingerprint") == LadderMPSMFT.model_fingerprint(settings.model) ||
            error("Phase 0 seed model fingerprint differs from the payload model")
        return (
            psi=read(file, "psi", MPS),
            fields=FieldState(
                read(file, "fields/alpha"),
                read(file, "fields/beta"),
                read(file, "fields/mu_cdw"),
            ),
        )
    end
    seed_psi = loaded.psi
    fields = loaded.fields
    sites = siteinds(seed_psi)
    seed_source = abspath(seed_path)
else
    rng = MersenneTwister(settings.run.random_seed)
    sites = LadderMPSMFT.make_sites(settings.model)
    fields = initial_fields(
        settings.model;
        seed=settings.run.initial_seed,
        amplitude=settings.run.initial_amplitude,
        rng,
    )
    seed_psi = productMPS(
        sites,
        density_product_state(2 * settings.model.L, settings.model.density; rng=MersenneTwister(settings.run.random_seed)),
    )
end
hamiltonian = build_mf_mpo(sites, settings.model, fields, settings.model.mu_initial)
repetitions = parse(Int, get(ENV, "PHASE0_REPETITIONS", "3"))
repetitions >= 1 || error("PHASE0_REPETITIONS must be positive")

seconds = Float64[]
energies = Float64[]
densities = Float64[]
bond_dimensions = Int[]
timed_out = Bool[]
for _ in 1:repetitions
    GC.gc()
    result_ref = Ref{Any}()
    elapsed = @elapsed result_ref[] = run_dmrg_ground(
        sites,
        hamiltonian,
        settings.model.density,
        settings.dmrg;
        psi_init=copy(seed_psi),
        rng=MersenneTwister(settings.run.random_seed),
        deadline=time() + settings.dmrg.max_time_seconds,
    )
    result = result_ref[]
    push!(seconds, elapsed)
    push!(energies, result.energy)
    push!(densities, LadderMPSMFT.average_density(result.psi))
    push!(bond_dimensions, maxlinkdim(result.psi))
    push!(timed_out, result.timed_out)
end

metric = Dict{String,Any}(
    "schema_version" => 1,
    "label" => label,
    "backend" => String(backend),
    "status" => any(timed_out) ? "time_limit" : "complete",
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
    "maximum_bond_dimensions" => bond_dimensions,
    "timed_out" => timed_out,
    "L" => settings.model.L,
    "physical_sites" => 2 * settings.model.L,
    "maxdim" => settings.dmrg.maxdim,
    "nsweeps" => settings.dmrg.nsweeps,
    "geometry" => String(settings.model.geometry),
    "config_path" => abspath(config_path),
    "config_sha256" => LadderMPSMFT.sha256_file(abspath(config_path)),
    "ep_source_sha256" => LadderMPSMFT.sha256_file(settings.model.ep_source),
    "seed_source" => seed_source,
    "seed_sha256" => isfile(seed_path) ? LadderMPSMFT.sha256_file(seed_path) : "",
    "git_commit" => LadderMPSMFT._read_git("rev-parse", "HEAD"),
    "implementation_sha256" => LadderMPSMFT.implementation_fingerprint(),
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
