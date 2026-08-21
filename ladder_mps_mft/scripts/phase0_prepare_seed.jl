#!/usr/bin/env julia

using HDF5
using ITensorMPS
using LadderMPSMFT
using Random

length(ARGS) == 2 || error(
    "usage: julia --project=. scripts/phase0_prepare_seed.jl CONFIG.toml SEED_STATE.h5",
)
config_path, output_path = abspath.(ARGS)
ispath(output_path) && error("refusing to overwrite immutable Phase 0 seed: $output_path")
settings = load_settings(config_path)
runtime = RuntimeSettings(blas_threads=1, strided_threads=1, threaded_blocksparse=true)
threading = configure_threading!(runtime)
rng = MersenneTwister(settings.run.random_seed)
sites = LadderMPSMFT.make_sites(settings.model)
fields = initial_fields(
    settings.model;
    seed=settings.run.initial_seed,
    amplitude=settings.run.initial_amplitude,
    rng,
)
psi0 = productMPS(
    sites,
    density_product_state(2 * settings.model.L, settings.model.density; rng=MersenneTwister(settings.run.random_seed)),
)
hamiltonian = build_mf_mpo(sites, settings.model, fields, settings.model.mu_initial)
result = run_dmrg_ground(
    sites,
    hamiltonian,
    settings.model.density,
    settings.dmrg;
    psi_init=psi0,
    rng=MersenneTwister(settings.run.random_seed),
    deadline=time() + settings.dmrg.max_time_seconds,
)
result.timed_out && error("Phase 0 seed DMRG reached its time limit")
mkpath(dirname(output_path))
temporary = tempname(dirname(output_path))
h5open(temporary, "w") do file
    file["schema_version"] = 1
    file["artifact_kind"] = "phase0_timing_seed"
    file["scientific_state"] = false
    file["psi"] = result.psi
    fields_group = create_group(file, "fields")
    fields_group["alpha"] = fields.alpha
    fields_group["beta"] = fields.beta
    fields_group["mu_cdw"] = fields.mu_cdw
    file["energy"] = result.energy
    file["density"] = LadderMPSMFT.average_density(result.psi)
    file["maximum_bond_dimension"] = maxlinkdim(result.psi)
    file["model_fingerprint"] = LadderMPSMFT.model_fingerprint(settings.model)
    file["config_sha256"] = LadderMPSMFT.sha256_file(config_path)
    file["ep_source_sha256"] = LadderMPSMFT.sha256_file(settings.model.ep_source)
    file["git_commit"] = LadderMPSMFT._read_git("rev-parse", "HEAD")
    file["implementation_sha256"] = LadderMPSMFT.implementation_fingerprint()
    file["julia_threads"] = threading.julia
    file["threaded_blocksparse"] = threading.blocksparse
end
mv(temporary, output_path)
println("seed_path=$output_path")
println("seed_sha256=$(LadderMPSMFT.sha256_file(output_path))")
